#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
一键流程：
1) 调用视觉总流程脚本 run_full_pipeline.py，得到目标在 L515 camera_color_optical_frame 下的 3D 坐标
2) 机械臂运动前先张开夹爪
3) 调用 move_link6_to_camera_point.py，把目标点传给机械臂运动脚本
4) 到位后闭合夹爪
5) 闭合后让 link6 保持当前姿态，回收到指定 xyz

设计原则：
- 视觉链路和机械臂链路尽量复用现有、已验证脚本，减少重复实现和引入新误差。
- 不在本脚本里重复做手眼变换、符号翻转、TCP offset 补偿；这些交给 move_link6_to_camera_point.py。
- 夹爪控制优先支持两种方式：
    A. 直接执行用户提供的 shell 命令（最稳，适配你的真实接口）
    B. 若未提供命令，则尝试用 MoveIt 的 gripper group + named target("open"/"close")
- 最后的回收动作不改变当前 link6 姿态，只修改其在 base_link 下的 xyz。

运行前提：
- 机械臂节点已打开
- MoveIt demo.launch 已打开
- 若使用 MoveIt 夹爪方式，gripper group 和 named target 已配置好

典型用法：
source ~/catkin_ws/devel/setup.bash
rosrun piper detect_and_grasp_pipeline.py --target-class "cell phone"

若夹爪不能直接用 MoveIt named target，建议显式给命令：
rosrun piper detect_and_grasp_pipeline.py \
  --target-class "cell phone" \
  --gripper-mode command \
  --gripper-open-cmd  "python3 /path/to/open_gripper.py" \
  --gripper-close-cmd "python3 /path/to/close_gripper.py"
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import rospy
import moveit_commander


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).absolute()


def run_cmd(cmd: List[str], desc: str, cwd: Optional[Path] = None) -> None:
    print(f"\n[STEP] {desc}")
    print("[CMD ] " + " ".join(shlex.quote(x) for x in cmd) + "\n")

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{desc} 失败，返回码: {ret}")


def choose_arm_group(robot, preferred_group=None) -> str:
    group_names = robot.get_group_names()
    rospy.loginfo("MoveIt 可用 group: %s", group_names)

    if preferred_group and preferred_group in group_names:
        return preferred_group

    priority = [
        preferred_group,
        "piper_arm",
        "arm",
        "manipulator",
        "piper",
    ]
    for name in priority:
        if name and name in group_names:
            return name

    for name in group_names:
        lname = name.lower()
        if "gripper" not in lname and "hand" not in lname:
            return name

    if group_names:
        return group_names[0]

    raise RuntimeError("没有找到任何 MoveIt arm group")


def move_link6_to_base_xyz_keep_current_orientation(args, x: float, y: float, z: float):
    """
    抓取完成后回收到一个 base_link 下的固定 xyz。
    姿态保持为“当前姿态”，也就是闭合夹爪后的 link6 姿态不变。
    """
    print("\n[STEP] 闭合夹爪后回收到指定位置（保持当前姿态）")
    rospy.loginfo("回收目标位置（%s）: [%.6f, %.6f, %.6f] m", args.base_frame, x, y, z)

    robot = moveit_commander.RobotCommander()
    group_name = choose_arm_group(robot, args.group)
    rospy.loginfo("回收动作使用 MoveIt arm group: %s", group_name)
    group = moveit_commander.MoveGroupCommander(group_name)

    group.set_pose_reference_frame(args.base_frame)
    group.set_planning_time(args.planning_time)
    group.set_num_planning_attempts(args.num_attempts)
    group.set_max_velocity_scaling_factor(args.vel_scale)
    group.set_max_acceleration_scaling_factor(args.acc_scale)
    group.set_goal_position_tolerance(args.pos_tol)
    group.set_goal_orientation_tolerance(args.ori_tol)
    group.allow_replanning(True)

    try:
        group.set_end_effector_link(args.ee_link)
    except Exception:
        pass

    current_pose = group.get_current_pose(args.ee_link).pose
    rospy.loginfo(
        "当前 link6 位姿（保持其方向不变）:\n"
        "  position = [%.6f, %.6f, %.6f] m\n"
        "  quaternion(xyzw) = [%.6f, %.6f, %.6f, %.6f]",
        current_pose.position.x,
        current_pose.position.y,
        current_pose.position.z,
        current_pose.orientation.x,
        current_pose.orientation.y,
        current_pose.orientation.z,
        current_pose.orientation.w,
    )

    target_pose = current_pose
    target_pose.position.x = float(x)
    target_pose.position.y = float(y)
    target_pose.position.z = float(z)

    rospy.loginfo(
        "回收目标 link6 位姿（%s）:\n"
        "  position = [%.6f, %.6f, %.6f] m\n"
        "  quaternion(xyzw) = [%.6f, %.6f, %.6f, %.6f]",
        args.base_frame,
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z,
        target_pose.orientation.x,
        target_pose.orientation.y,
        target_pose.orientation.z,
        target_pose.orientation.w,
    )

    if args.dry_run_arm:
        rospy.logwarn("dry-run-arm 模式：回收动作只打印，不执行。")
        return

    group.set_start_state_to_current_state()
    group.set_pose_target(target_pose, args.ee_link)
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    if not success:
        raise RuntimeError(
            "回收动作 MoveIt 规划/执行失败。\n"
            "建议先检查该 xyz 是否在工作空间内，以及当前抓取后姿态在该位置是否可达。"
        )


class GripperController:
    def __init__(self, args):
        self.args = args
        self.robot = None
        self.group = None

    def _choose_gripper_group(self, robot, preferred_group=None) -> str:
        group_names = robot.get_group_names()
        rospy.loginfo("MoveIt 可用 group: %s", group_names)

        if preferred_group and preferred_group in group_names:
            return preferred_group

        priority = [
            preferred_group,
            "gripper",
            "hand",
            "piper_gripper",
            "piper_hand",
        ]
        for name in priority:
            if name and name in group_names:
                return name

        for name in group_names:
            lname = name.lower()
            if "gripper" in lname or "hand" in lname:
                return name

        raise RuntimeError(
            "没有找到 gripper/hand 类 MoveIt group。"
            "请改用 --gripper-mode command 并提供 open/close 命令，"
            "或者显式指定 --gripper-group。"
        )

    def _ensure_moveit_group(self):
        if self.group is not None:
            return
        self.robot = moveit_commander.RobotCommander()
        group_name = self._choose_gripper_group(self.robot, self.args.gripper_group)
        rospy.loginfo("使用夹爪 MoveIt group: %s", group_name)
        self.group = moveit_commander.MoveGroupCommander(group_name)
        self.group.set_planning_time(self.args.gripper_planning_time)
        self.group.set_num_planning_attempts(self.args.gripper_num_attempts)
        self.group.set_max_velocity_scaling_factor(self.args.gripper_vel_scale)
        self.group.set_max_acceleration_scaling_factor(self.args.gripper_acc_scale)
        self.group.allow_replanning(True)

    def _run_shell_command(self, command: str, desc: str):
        print(f"\n[STEP] {desc}")
        print(f"[CMD ] {command}\n")
        ret = subprocess.call(["bash", "-lc", command])
        if ret != 0:
            raise RuntimeError(f"{desc} 失败，返回码: {ret}")

    def _run_moveit_named(self, target_name: str, desc: str):
        self._ensure_moveit_group()

        named_targets = []
        if hasattr(self.group, "get_named_targets"):
            try:
                named_targets = list(self.group.get_named_targets())
            except Exception:
                named_targets = []

        rospy.loginfo("夹爪 group 的 named targets: %s", named_targets)
        if named_targets and target_name not in named_targets:
            raise RuntimeError(
                f"夹爪 MoveIt group 不包含 named target '{target_name}'。"
                f"当前可用 targets: {named_targets}\n"
                "你可以改 --gripper-open-target / --gripper-close-target，"
                "或者改用 --gripper-mode command。"
            )

        print(f"\n[STEP] {desc}")
        print(f"[INFO] MoveIt named target = {target_name}\n")
        self.group.set_named_target(target_name)
        success = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
        if not success:
            raise RuntimeError(f"{desc} 失败：MoveIt 执行未成功")

    def open(self):
        if self.args.gripper_mode == "skip":
            rospy.logwarn("已跳过夹爪张开步骤")
            return

        if self.args.gripper_mode == "command":
            if not self.args.gripper_open_cmd:
                raise RuntimeError("gripper_mode=command 但未提供 --gripper-open-cmd")
            self._run_shell_command(self.args.gripper_open_cmd, "张开夹爪")
            return

        if self.args.gripper_mode == "moveit_named":
            self._run_moveit_named(self.args.gripper_open_target, "张开夹爪")
            return

        if self.args.gripper_open_cmd and self.args.gripper_close_cmd:
            self._run_shell_command(self.args.gripper_open_cmd, "张开夹爪")
        else:
            self._run_moveit_named(self.args.gripper_open_target, "张开夹爪")

    def close(self):
        if self.args.gripper_mode == "skip":
            rospy.logwarn("已跳过夹爪闭合步骤")
            return

        if self.args.gripper_mode == "command":
            if not self.args.gripper_close_cmd:
                raise RuntimeError("gripper_mode=command 但未提供 --gripper-close-cmd")
            self._run_shell_command(self.args.gripper_close_cmd, "闭合夹爪")
            return

        if self.args.gripper_mode == "moveit_named":
            self._run_moveit_named(self.args.gripper_close_target, "闭合夹爪")
            return

        if self.args.gripper_open_cmd and self.args.gripper_close_cmd:
            self._run_shell_command(self.args.gripper_close_cmd, "闭合夹爪")
        else:
            self._run_moveit_named(self.args.gripper_close_target, "闭合夹爪")


def load_position_from_latest_json(latest_json: Path):
    if not latest_json.is_file():
        raise FileNotFoundError(f"未找到 latest_result.json: {latest_json}")

    data = json.loads(latest_json.read_text(encoding="utf-8"))
    pos = data.get("position_cam", {})
    x = float(pos["x"])
    y = float(pos["y"])
    z = float(pos["z"])
    return data, x, y, z


def build_detection_cmd(args, run_pipeline_script: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(run_pipeline_script),
        "--target-class", args.target_class,
        "--selection-rule", args.selection_rule,
        "--base-outdir", str(args.base_outdir),
        "--run-id", args.run_id,
        "--conf", str(args.det_conf),
        "--imgsz", str(args.imgsz),
        "--fx", str(args.fx),
        "--fy", str(args.fy),
        "--cx", str(args.cx),
        "--cy", str(args.cy),
        "--depth-scale", str(args.depth_scale),
        "--min-depth-raw", str(args.min_depth_raw),
        "--max-depth-raw", str(args.max_depth_raw),
    ]

    if args.allow_fallback:
        cmd.append("--allow-fallback")
    if args.skip_launch:
        cmd.append("--skip-launch")

    cmd.extend(["--ros-setup", args.ros_setup])
    cmd.extend(["--ws-setup", args.ws_setup])
    cmd.extend(["--conda-sh", args.conda_sh])
    cmd.extend(["--conda-env", args.conda_env])
    cmd.extend(["--capture-script", args.capture_script])
    cmd.extend(["--yolo-script", args.yolo_script])
    cmd.extend(["--estimate-script", args.estimate_script])
    cmd.extend(["--color-topic", args.color_topic])
    cmd.extend(["--depth-topic", args.depth_topic])
    cmd.extend(["--launch-pkg", args.launch_pkg])
    cmd.extend(["--launch-file", args.launch_file])
    cmd.extend(["--timeout", str(args.timeout)])
    cmd.extend(["--startup-delay", str(args.startup_delay)])
    cmd.extend(["--model", args.model])

    return cmd


def build_move_cmd(args, move_script: Path, x: float, y: float, z: float) -> List[str]:
    cmd = [
        sys.executable,
        str(move_script),
        "--x", f"{x:.9f}",
        "--y", f"{y:.9f}",
        "--z", f"{z:.9f}",
        "--base-frame", args.base_frame,
        "--camera-frame", args.camera_frame,
        "--ee-link", args.ee_link,
        "--planning-time", str(args.planning_time),
        "--num-attempts", str(args.num_attempts),
        "--vel-scale", str(args.vel_scale),
        "--acc-scale", str(args.acc_scale),
        "--pos-tol", str(args.pos_tol),
        "--ori-tol", str(args.ori_tol),
        "--tf-timeout", str(args.tf_timeout),
    ]
    if args.group:
        cmd.extend(["--group", args.group])
    if args.prefer_x_up:
        cmd.append("--prefer-x-up")
    if args.dry_run_arm:
        cmd.append("--dry-run")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="一键执行：识别 -> 张开夹爪 -> 机械臂到点 -> 闭合夹爪 -> 回收到指定位置")

    parser.add_argument(
        "--run-pipeline-script",
        default="/home/songfei/catkin_ws/src/milk_vision_yolo11/scripts/run_full_pipeline.py",
        help="视觉总流程脚本 run_full_pipeline.py 的绝对路径",
    )
    parser.add_argument(
        "--move-script",
        default="/home/songfei/catkin_ws/src/piper_ros/src/piper/scripts/move_link6_to_camera_point.py",
        help="机械臂运动脚本 move_link6_to_camera_point.py 的绝对路径",
    )

    parser.add_argument("--target-class", default="cell phone")
    parser.add_argument("--selection-rule", default="largest_area", choices=["largest_area", "highest_conf", "first"])
    parser.add_argument("--allow-fallback", action="store_true")
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--base-outdir", default="/home/songfei/catkin_ws/src/milk_vision_yolo11/pipeline_runs")

    parser.add_argument("--ros-setup", default="/opt/ros/noetic/setup.bash")
    parser.add_argument("--ws-setup", default="~/catkin_ws/devel/setup.bash")
    parser.add_argument("--conda-sh", default="~/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="milk_yolo11")
    parser.add_argument("--capture-script", default="/home/songfei/catkin_ws/src/l515_snapshot_tool/capture_l515_once.py")
    parser.add_argument("--yolo-script", default="/home/songfei/catkin_ws/src/milk_vision_yolo11/scripts/run_yolo11_seg_demo.py")
    parser.add_argument("--estimate-script", default="/home/songfei/catkin_ws/src/milk_vision_yolo11/scripts/estimate_position_from_mask_depth.py")
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--launch-pkg", default="robocup_executive")
    parser.add_argument("--launch-file", default="l515.launch")
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--startup-delay", type=float, default=8.0)
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--fx", type=float, default=905.534912109375)
    parser.add_argument("--fy", type=float, default=905.7691650390625)
    parser.add_argument("--cx", type=float, default=656.93017578125)
    parser.add_argument("--cy", type=float, default=356.93890380859375)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth-raw", type=int, default=300)
    parser.add_argument("--max-depth-raw", type=int, default=4000)

    parser.add_argument("--group", default=None)
    parser.add_argument("--base-frame", default="base_link")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--ee-link", default="link6")
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--vel-scale", type=float, default=0.2)
    parser.add_argument("--acc-scale", type=float, default=0.2)
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--ori-tol", type=float, default=0.05235987755982989)
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    parser.add_argument("--prefer-x-up", action="store_true")
    parser.add_argument("--dry-run-arm", action="store_true", help="只打印机械臂目标位姿，不实际运动")

    parser.add_argument("--gripper-mode", default="auto", choices=["auto", "moveit_named", "command", "skip"])
    parser.add_argument("--gripper-group", default=None)
    parser.add_argument("--gripper-open-target", default="open")
    parser.add_argument("--gripper-close-target", default="close")
    parser.add_argument("--gripper-open-cmd", default="")
    parser.add_argument("--gripper-close-cmd", default="")
    parser.add_argument("--gripper-planning-time", type=float, default=3.0)
    parser.add_argument("--gripper-num-attempts", type=int, default=5)
    parser.add_argument("--gripper-vel-scale", type=float, default=0.5)
    parser.add_argument("--gripper-acc-scale", type=float, default=0.5)
    parser.add_argument("--pause-after-open", type=float, default=0.5)
    parser.add_argument("--pause-before-close", type=float, default=0.2)

    # 回收动作参数：保持当前姿态，只改 link6 的 xyz
    parser.add_argument("--retreat-x", type=float, default=0.045)
    parser.add_argument("--retreat-y", type=float, default=0.0)
    parser.add_argument("--retreat-z", type=float, default=0.346)
    parser.add_argument("--pause-after-close", type=float, default=0.2)
    parser.add_argument("--skip-retreat", action="store_true", help="跳过最后的回收动作")

    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node("detect_and_grasp_pipeline", anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)

    run_pipeline_script = expand_path(args.run_pipeline_script)
    move_script = expand_path(args.move_script)
    args.base_outdir = str(expand_path(args.base_outdir))

    for p in [run_pipeline_script, move_script]:
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

    latest_json = expand_path(args.base_outdir) / "latest_result.json"

    detect_cmd = build_detection_cmd(args, run_pipeline_script)
    run_cmd(detect_cmd, "执行视觉总流程（拍照 -> 分割 -> 3D 反推）")

    result_data, x_cam, y_cam, z_cam = load_position_from_latest_json(latest_json)
    print("\n" + "=" * 80)
    print("[VISION RESULT]")
    print(f"target_class = {result_data.get('target_class_requested')}")
    print(f"selected_class = {result_data.get('selected_detection', {}).get('class_name')}")
    print(f"position_cam  = [{x_cam:.6f}, {y_cam:.6f}, {z_cam:.6f}] m")
    print(f"latest_json   = {latest_json}")
    print("=" * 80)

    gripper = GripperController(args)
    gripper.open()
    if args.pause_after_open > 0:
        rospy.sleep(args.pause_after_open)

    move_cmd = build_move_cmd(args, move_script, x_cam, y_cam, z_cam)
    run_cmd(move_cmd, "驱动机械臂到目标点")

    if args.pause_before_close > 0:
        rospy.sleep(args.pause_before_close)

    gripper.close()

    if args.pause_after_close > 0:
        rospy.sleep(args.pause_after_close)

    if not args.skip_retreat:
        move_link6_to_base_xyz_keep_current_orientation(
            args,
            x=args.retreat_x,
            y=args.retreat_y,
            z=args.retreat_z,
        )
    else:
        rospy.logwarn("已跳过最后的回收动作")

    print("\n" + "=" * 80)
    print("[DONE] 全流程完成：识别 -> 张开夹爪 -> 机械臂到点 -> 闭合夹爪 -> 回收到指定位置")
    print("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(str(e))
        sys.exit(1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
功能：
给定一个“在 L515 相机坐标系 camera_color_optical_frame 下”的目标点，
在当前机械臂姿态处读取 base_link -> link6，结合手眼标定结果，
把该目标点先换算到 base_link 坐标系，再生成一个 link6 的目标位姿，
最后通过 MoveIt 驱动机械臂运动过去。

核心假设：
1) 当前输入的相机系目标点，是“当前这一刻”相机看到的目标点；
   也就是说，它会先被固化到 base_link 坐标系，然后再做一次性运动。
2) 你给出的 easy_handeye 标定结果默认解释为：camera_color_optical_frame -> link6
   （即：link6 在 camera_color_optical_frame 下的位姿表达）。
   如果你后续核验发现方向相反，把 CALIB_IS_CAMERA_TO_LINK6 改成 False 即可。
3) 真实末端 TCP 相对 link6 原点，沿 link6 的 +Z 方向前伸 9.5 cm。
   因此如果“TCP 要到目标点”，那么 link6 原点实际应位于：
       p_link6 = p_tcp_target - R_base_link6 * [0, 0, 0.095]^T
4) 希望 link6 的 Z 轴水平朝前，这里默认令：
       z_link6(base) = +x_base
   并默认让 link6 的 X 轴尽量朝下（更贴近很多机械臂腕部水平前伸的姿态）。

示例：
rosrun piper move_link6_to_camera_point.py --x 0.10 --y -0.03 --z 0.45
"""

import sys
import math
import argparse
import numpy as np

import rospy
import tf2_ros
import geometry_msgs.msg
import moveit_commander
from geometry_msgs.msg import Pose
from tf.transformations import (
    translation_matrix,
    quaternion_matrix,
    quaternion_from_matrix,
    translation_from_matrix,
    inverse_matrix,
    concatenate_matrices,
)


# =========================
# 这里是你的手眼标定常量
# =========================
CALIB_TRANSLATION = np.array([
    -0.1323284028220638,
    -0.006644526876749131,
    -0.0035613163419305266,
], dtype=float)

CALIB_QUATERNION_XYZW = np.array([
    -0.09512232494891511,
     0.07523262610165826,
    -0.7016308019558558,
     0.702143869169947,
], dtype=float)

# True: 以上标定值表示 ^camera T_link6
# False: 以上标定值表示 ^link6 T_camera
CALIB_IS_CAMERA_TO_LINK6 = True

# link6 原点到真实末端 TCP 的偏移：沿 link6 +Z 方向 9.5 cm
TCP_OFFSET_ALONG_LINK6_Z_M = 0.13

# 默认坐标系命名
DEFAULT_BASE_FRAME = "base_link"
DEFAULT_CAMERA_FRAME = "camera_color_optical_frame"
DEFAULT_EE_LINK = "link6"


def normalize(v, eps=1e-12):
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("零向量无法归一化")
    return v / n


def make_transform_xyzquat(translation_xyz, quat_xyzw):
    T = concatenate_matrices(
        translation_matrix(translation_xyz),
        quaternion_matrix(quat_xyzw),
    )
    return T


def pose_to_matrix(pose_msg):
    t = [
        pose_msg.position.x,
        pose_msg.position.y,
        pose_msg.position.z,
    ]
    q = [
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
        pose_msg.orientation.w,
    ]
    return make_transform_xyzquat(t, q)


def matrix_to_pose(T):
    pose = Pose()
    t = translation_from_matrix(T)
    q = quaternion_from_matrix(T)
    pose.position.x = float(t[0])
    pose.position.y = float(t[1])
    pose.position.z = float(t[2])
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def tfmsg_to_matrix(tf_msg):
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    return make_transform_xyzquat(
        [t.x, t.y, t.z],
        [q.x, q.y, q.z, q.w],
    )


def build_desired_rotation_base_to_link6(tool_forward_in_base,
                                         prefer_x_down=True):
    """
    构造 R_base_link6，使 link6 的 Z 轴指向 tool_forward_in_base。
    其余绕 Z 的自由度用“X 轴尽量朝下/朝上”来定。

    返回 3x3 旋转矩阵，列向量分别是 link6 的 x/y/z 轴在 base 下的表达。
    """
    z_axis = normalize(np.array(tool_forward_in_base, dtype=float))

    if prefer_x_down:
        x_ref = np.array([0.0, 0.0, -1.0], dtype=float)  # base -Z
    else:
        x_ref = np.array([0.0, 0.0, 1.0], dtype=float)   # base +Z

    # 把 x_ref 投影到垂直 z_axis 的平面上，作为 link6 的 x 轴优选方向
    x_axis = x_ref - np.dot(x_ref, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        # 极端退化时，选一个备用参考轴
        fallback = np.array([0.0, 1.0, 0.0], dtype=float)
        x_axis = fallback - np.dot(fallback, z_axis) * z_axis
    x_axis = normalize(x_axis)

    # 保证右手系：x × y = z  ==>  y = z × x
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.column_stack((x_axis, y_axis, z_axis))
    return R


def choose_move_group(robot, preferred_group=None):
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

    # 尽量避开 gripper 类 group
    for name in group_names:
        lname = name.lower()
        if "gripper" not in lname and "hand" not in lname:
            return name

    if group_names:
        return group_names[0]

    raise RuntimeError("没有找到任何 MoveIt group")


def main():
    rospy.init_node("move_link6_to_camera_point", anonymous=False)
    moveit_commander.roscpp_initialize(sys.argv)

    parser = argparse.ArgumentParser(description="给定相机系目标点，驱动 link6/TCP 到该位置")
    parser.add_argument("--x", type=float, required=True, help="目标点在 camera_color_optical_frame 下的 x (m)")
    parser.add_argument("--y", type=float, required=True, help="目标点在 camera_color_optical_frame 下的 y (m)")
    parser.add_argument("--z", type=float, required=True, help="目标点在 camera_color_optical_frame 下的 z (m)")
    parser.add_argument("--group", type=str, default=None, help="MoveIt 机械臂 group 名称；不填则自动选择")
    parser.add_argument("--base-frame", type=str, default=DEFAULT_BASE_FRAME)
    parser.add_argument("--camera-frame", type=str, default=DEFAULT_CAMERA_FRAME)
    parser.add_argument("--ee-link", type=str, default=DEFAULT_EE_LINK)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--num-attempts", type=int, default=10)
    parser.add_argument("--vel-scale", type=float, default=0.2)
    parser.add_argument("--acc-scale", type=float, default=0.2)
    parser.add_argument("--pos-tol", type=float, default=0.005)
    parser.add_argument("--ori-tol", type=float, default=math.radians(3.0))
    parser.add_argument("--tf-timeout", type=float, default=2.0)
    parser.add_argument("--prefer-x-up", action="store_true",
                        help="默认让 link6 的 X 轴尽量朝下；加这个参数则改为尽量朝上")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算并打印结果，不执行运动")
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    target_cam = np.array([-args.x, -args.y, args.z], dtype=float)
    base_frame = args.base_frame
    camera_frame = args.camera_frame
    ee_link = args.ee_link

    rospy.loginfo("输入目标点（相机系 %s）: [%.6f, %.6f, %.6f] m",
                  camera_frame, target_cam[0], target_cam[1], target_cam[2])

    # -------------------------
    # 1) 组装手眼标定矩阵
    # -------------------------
    T_given = make_transform_xyzquat(CALIB_TRANSLATION, CALIB_QUATERNION_XYZW)
    if CALIB_IS_CAMERA_TO_LINK6:
        T_cam_link6 = T_given
        T_link6_cam = inverse_matrix(T_cam_link6)
    else:
        T_link6_cam = T_given
        T_cam_link6 = inverse_matrix(T_link6_cam)

    rospy.loginfo("手眼外参默认解释: %s",
                  "^camera T_link6" if CALIB_IS_CAMERA_TO_LINK6 else "^link6 T_camera")

    # -------------------------
    # 2) 读取当前 base_link -> link6
    # -------------------------
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.5)

    try:
        tf_bl_l6 = tf_buffer.lookup_transform(
            base_frame,
            ee_link,
            rospy.Time(0),
            timeout=rospy.Duration(args.tf_timeout),
        )
    except Exception as e:
        raise RuntimeError(
            "读取当前 TF(base_link -> link6) 失败。请确认机械臂节点/robot_state_publisher 已正常发布 TF。\n"
            "原始报错: {}".format(e)
        )

    T_base_link6_current = tfmsg_to_matrix(tf_bl_l6)
    T_base_camera_current = np.dot(T_base_link6_current, T_link6_cam)

    p_target_base = np.dot(T_base_camera_current[:3, :3], target_cam) + T_base_camera_current[:3, 3]
    rospy.loginfo("目标点换算到 %s 下: [%.6f, %.6f, %.6f] m",
                  base_frame, p_target_base[0], p_target_base[1], p_target_base[2])

    # -------------------------
    # 3) 构造期望 link6 姿态
    #    要求：link6 的 z 轴水平朝前
    #    这里默认“前”取 base_link 的 +X
    # -------------------------
    forward_in_base = np.array([1.0, 0.0, 0.0], dtype=float)
    R_base_link6_des = build_desired_rotation_base_to_link6(
        tool_forward_in_base=forward_in_base,
        prefer_x_down=(not args.prefer_x_up),
    )

    # TCP 相对 link6 原点：沿 link6 +Z 前伸 0.095 m
    tcp_offset_in_link6 = np.array([0.0, 0.0, TCP_OFFSET_ALONG_LINK6_Z_M], dtype=float)
    tcp_offset_in_base = np.dot(R_base_link6_des, tcp_offset_in_link6)

    # 关键：如果 TCP 要命中 p_target_base，则 link6 原点需要“后退”这段偏移
    p_link6_des_base = p_target_base - tcp_offset_in_base

    T_base_link6_des = np.eye(4)
    T_base_link6_des[:3, :3] = R_base_link6_des
    T_base_link6_des[:3, 3] = p_link6_des_base
    target_pose = matrix_to_pose(T_base_link6_des)

    q = target_pose.orientation
    rospy.loginfo(
        "目标 link6 位姿（%s）:\n"
        "  position = [%.6f, %.6f, %.6f] m\n"
        "  quaternion(xyzw) = [%.6f, %.6f, %.6f, %.6f]",
        base_frame,
        target_pose.position.x,
        target_pose.position.y,
        target_pose.position.z,
        q.x, q.y, q.z, q.w,
    )

    if args.dry_run:
        rospy.logwarn("dry-run 模式：只计算，不执行运动。")
        return

    # -------------------------
    # 4) MoveIt 执行
    # -------------------------
    robot = moveit_commander.RobotCommander()
    group_name = choose_move_group(robot, args.group)
    rospy.loginfo("使用 MoveIt group: %s", group_name)
    group = moveit_commander.MoveGroupCommander(group_name)

    group.set_pose_reference_frame(base_frame)
    group.set_planning_time(args.planning_time)
    group.set_num_planning_attempts(args.num_attempts)
    group.set_max_velocity_scaling_factor(args.vel_scale)
    group.set_max_acceleration_scaling_factor(args.acc_scale)
    group.set_goal_position_tolerance(args.pos_tol)
    group.set_goal_orientation_tolerance(args.ori_tol)
    group.allow_replanning(True)

    try:
        group.set_end_effector_link(ee_link)
    except Exception:
        # 某些版本没有这个接口或 end effector 未配置，不强制报错
        pass

    current_pose = group.get_current_pose(ee_link).pose
    rospy.loginfo("当前 link6 位置（%s）: [%.6f, %.6f, %.6f] m",
                  base_frame,
                  current_pose.position.x,
                  current_pose.position.y,
                  current_pose.position.z)

    group.set_start_state_to_current_state()
    group.set_pose_target(target_pose, ee_link)

    rospy.loginfo("开始规划并执行...")
    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()

    if not success:
        raise RuntimeError(
            "MoveIt 规划/执行失败。\n"
            "建议先用 --dry-run 看目标位姿是否合理；\n"
            "也可以把速度调小，或确认该目标是否超出工作空间/存在碰撞。"
        )

    rospy.loginfo("执行完成。建议在 RViz / 实机上核验末端是否准确到位。")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(str(e))
        sys.exit(1)

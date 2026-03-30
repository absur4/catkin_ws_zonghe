#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1 + MoveIt 主逻辑：眼在手上（eye-on-hand）抓取执行节点

功能概述
--------
1. 读取“手眼标定结果”（默认假定为: T_ee_cam，即 camera -> ee_link/link6）
2. 读取“抓姿生成模块”给出的目标抓姿（此处先用手动输入参数代替）
   默认假定输入抓姿是：T_cam_grasp，即“目标抓取TCP/抓手中心”在 camera frame 下的位姿
3. 结合当前机械臂姿态，计算目标抓取位姿在 base_link 下的目标 ee_link 位姿
4. 控制 MoveIt 执行：
   - 张开夹爪（可选）
   - 到达 pre-grasp
   - 笛卡尔直线逼近 grasp
   - 闭合夹爪（可选）
   - 直线撤回到 pre-grasp
   - 抬升

重要坐标变换约定
----------------
本文统一使用 T_a_b 表示："把 b 坐标系中的点/姿态，变换到 a 坐标系下" 的 4x4 齐次矩阵。

若：
- 当前末端位姿      = T_base_ee_current
- 手眼标定结果      = T_ee_cam    (默认假定：你给出的标定结果是 camera -> link6)
- 感知抓姿输入      = T_cam_grasp (抓取TCP/抓手中心在相机坐标系下的目标位姿)
- 工具偏移          = T_ee_tcp    (ee_link -> TCP；若 link6 就是抓取参考点，则取单位阵)

则：
- 当前相机位姿      = T_base_cam = T_base_ee_current @ T_ee_cam
- 目标抓取TCP位姿   = T_base_grasp = T_base_cam @ T_cam_grasp
- 目标末端ee位姿    = T_base_ee_target = T_base_grasp @ inv(T_ee_tcp)

使用前务必确认
--------------
1. 你当前的手眼标定结果，确实对应 camera -> link6，而不是 link6 -> camera。
   若方向相反，把参数 ~handeye_result_is_ee_from_cam 改为 False 即可。
2. 你的“抓姿生成”输出，到底是“抓手中心/TCP”的位姿，还是“link6”的位姿。
   - 如果输出就是 link6 位姿，则把 ~ee_to_tcp_xyz 设为 [0,0,0]，~ee_to_tcp_quat 设为单位四元数。
   - 如果输出是抓手中心/TCP 位姿，则正确填写 ee_link 到 TCP 的静态偏移。
3. approach_axis_in_tcp 必须符合你的夹爪前进方向约定。
   例如很多夹爪是 TCP 的 +Z 或 -Z 方向为逼近方向；这个一定要按你的抓手模型确认。
4. 首次运行建议：
   - ask_before_execute=True
   - velocity / acceleration scaling 调低
   - 先只做 dry-run（看日志和目标位姿是否合理）

示例运行
--------
rosrun your_pkg eye_on_hand_grasp_main.py \
  _move_group:=arm \
  _base_frame:=base_link \
  _ee_link:=link6 \
  _camera_frame:=camera_color_optical_frame \
  _handeye_result_is_ee_from_cam:=True \
  _handeye_xyz:='[-0.1323284028220638, -0.006644526876749131, -0.0035613163419305266]' \
  _handeye_quat:='[-0.09512232494891511, 0.07523262610165826, -0.7016308019558558, 0.702143869169947]' \
  _grasp_xyz:='[0.10, 0.00, 0.35]' \
  _grasp_quat:='[0.0, 0.0, 0.0, 1.0]'

备注
----
- 代码默认基于 ROS1 (rospy + moveit_commander)。
- 夹爪控制部分由于 Piper 的具体接口可能与你系统实际不一致，因此做成“可选参数驱动”。
  如果不配置 gripper_command_topic，就只移动机械臂，不实际发夹爪命令。
"""

import sys
import copy
import math
import numpy as np

import rospy
import tf2_ros
import moveit_commander

from geometry_msgs.msg import Pose, PoseStamped
from std_msgs.msg import Float64
from tf.transformations import (
    quaternion_matrix,
    quaternion_from_matrix,
    translation_matrix,
    inverse_matrix,
)


def normalize_quaternion(q):
    """归一化四元数 [x, y, z, w]。"""
    q = np.array(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("四元数范数过小，无法归一化。")
    return (q / n).tolist()


def pose_to_matrix_from_xyz_quat(xyz, quat_xyzw):
    """由平移 + 四元数构造 4x4 齐次矩阵。"""
    quat_xyzw = normalize_quaternion(quat_xyzw)
    T = quaternion_matrix(quat_xyzw)
    T[0:3, 3] = np.array(xyz, dtype=np.float64)
    return T


def pose_msg_to_matrix(pose_msg):
    """geometry_msgs/Pose -> 4x4 矩阵。"""
    xyz = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    quat = [
        pose_msg.orientation.x,
        pose_msg.orientation.y,
        pose_msg.orientation.z,
        pose_msg.orientation.w,
    ]
    return pose_to_matrix_from_xyz_quat(xyz, quat)


def matrix_to_pose_msg(T):
    """4x4 矩阵 -> geometry_msgs/Pose。"""
    pose = Pose()
    pose.position.x = float(T[0, 3])
    pose.position.y = float(T[1, 3])
    pose.position.z = float(T[2, 3])

    q = quaternion_from_matrix(T)
    q = normalize_quaternion(q)
    pose.orientation.x = float(q[0])
    pose.orientation.y = float(q[1])
    pose.orientation.z = float(q[2])
    pose.orientation.w = float(q[3])
    return pose


def transform_stamped_to_matrix(tf_msg):
    """geometry_msgs/TransformStamped -> 4x4 矩阵。"""
    xyz = [
        tf_msg.transform.translation.x,
        tf_msg.transform.translation.y,
        tf_msg.transform.translation.z,
    ]
    quat = [
        tf_msg.transform.rotation.x,
        tf_msg.transform.rotation.y,
        tf_msg.transform.rotation.z,
        tf_msg.transform.rotation.w,
    ]
    return pose_to_matrix_from_xyz_quat(xyz, quat)


def matrix_to_xyz_quat(T):
    """4x4 齐次矩阵 -> (xyz, quat_xyzw)。"""
    xyz = T[0:3, 3].astype(np.float64).tolist()
    quat = normalize_quaternion(quaternion_from_matrix(T))
    return xyz, quat


def make_local_translation(dx, dy, dz):
    """局部平移矩阵。右乘到 T_base_xxx 上时，表示沿 xxx 局部坐标轴平移。"""
    return translation_matrix((dx, dy, dz))


def make_world_translation(dx, dy, dz):
    """世界/基坐标系平移矩阵。左乘到 T_base_xxx 上时，表示沿 base 坐标轴平移。"""
    return translation_matrix((dx, dy, dz))


class EyeOnHandGraspExecutor(object):
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("eye_on_hand_grasp_executor", anonymous=False)

        # -------------------------
        # 基本参数
        # -------------------------
        self.move_group_name = rospy.get_param("~move_group", "arm")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.ee_link = rospy.get_param("~ee_link", "link6")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_color_optical_frame")

        self.planning_time = rospy.get_param("~planning_time", 5.0)
        self.num_planning_attempts = rospy.get_param("~num_planning_attempts", 10)
        self.max_velocity_scaling = rospy.get_param("~max_velocity_scaling", 0.15)
        self.max_acceleration_scaling = rospy.get_param("~max_acceleration_scaling", 0.15)
        self.goal_position_tolerance = rospy.get_param("~goal_position_tolerance", 0.003)
        self.goal_orientation_tolerance = rospy.get_param("~goal_orientation_tolerance", 0.01)

        self.ask_before_execute = rospy.get_param("~ask_before_execute", True)
        self.open_gripper_before_start = rospy.get_param("~open_gripper_before_start", True)

        # -------------------------
        # 手眼标定参数
        # 默认假定你给的是 T_ee_cam（camera -> ee_link/link6）
        # 若你实际给的是 T_cam_ee，则把 handeye_result_is_ee_from_cam 设为 False
        # -------------------------
        self.handeye_result_is_ee_from_cam = rospy.get_param("~handeye_result_is_ee_from_cam", True)
        self.handeye_xyz = rospy.get_param(
            "~handeye_xyz",
            [-0.1323284028220638, -0.006644526876749131, -0.0035613163419305266],
        )
        self.handeye_quat = rospy.get_param(
            "~handeye_quat",
            [-0.09512232494891511, 0.07523262610165826, -0.7016308019558558, 0.702143869169947],
        )

        # -------------------------
        # ee_link -> TCP 的静态偏移
        # 若抓姿输出就是 link6 位姿，则保持为单位阵即可
        # -------------------------
        self.ee_to_tcp_xyz = rospy.get_param("~ee_to_tcp_xyz", [0.0, 0.0, 0.0])
        self.ee_to_tcp_quat = rospy.get_param("~ee_to_tcp_quat", [0.0, 0.0, 0.0, 1.0])

        # -------------------------
        # 抓取流程参数
        # approach_axis_in_tcp: TCP 局部坐标系中的“逼近方向”单位向量
        # pregrasp 的位置 = grasp 沿该方向反向退回 pregrasp_distance
        # -------------------------
        self.approach_axis_in_tcp = np.array(
            rospy.get_param("~approach_axis_in_tcp", [0.0, 0.0, 1.0]),
            dtype=np.float64,
        )
        axis_norm = np.linalg.norm(self.approach_axis_in_tcp)
        if axis_norm < 1e-12:
            raise ValueError("~approach_axis_in_tcp 不能是零向量。")
        self.approach_axis_in_tcp = self.approach_axis_in_tcp / axis_norm

        self.pregrasp_distance = rospy.get_param("~pregrasp_distance", 0.08)
        self.eef_step = rospy.get_param("~eef_step", 0.005)
        self.jump_threshold = rospy.get_param("~jump_threshold", 0.0)
        self.min_cartesian_fraction = rospy.get_param("~min_cartesian_fraction", 0.99)
        self.lift_distance = rospy.get_param("~lift_distance", 0.08)

        # -------------------------
        # 手动输入的抓姿（占位）
        # 这里假定抓姿是 T_cam_grasp，即“目标TCP位姿在相机坐标系下的表示”
        # -------------------------
        self.grasp_xyz = rospy.get_param("~grasp_xyz", [0.10, 0.00, 0.35])
        self.grasp_quat = rospy.get_param("~grasp_quat", [0.0, 0.0, 0.0, 1.0])

        # -------------------------
        # 可选：夹爪控制
        # 若不提供 gripper_command_topic，则默认只打印日志，不真正发夹爪控制命令
        # -------------------------
        self.gripper_command_topic = rospy.get_param("~gripper_command_topic", "")
        self.gripper_open_value = rospy.get_param("~gripper_open_value", 0.08)
        self.gripper_close_value = rospy.get_param("~gripper_close_value", 0.0)
        self.gripper_command_sleep = rospy.get_param("~gripper_command_sleep", 1.0)
        self.gripper_pub = None
        if self.gripper_command_topic:
            self.gripper_pub = rospy.Publisher(self.gripper_command_topic, Float64, queue_size=10)
            rospy.loginfo("夹爪控制已启用，topic: %s", self.gripper_command_topic)
        else:
            rospy.logwarn("未配置 ~gripper_command_topic，夹爪开合将只输出日志，不发实际命令。")

        # -------------------------
        # TF 和 MoveIt 初始化
        # -------------------------
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group = moveit_commander.MoveGroupCommander(self.move_group_name)

        self.move_group.set_planning_time(self.planning_time)
        self.move_group.set_num_planning_attempts(self.num_planning_attempts)
        self.move_group.set_max_velocity_scaling_factor(self.max_velocity_scaling)
        self.move_group.set_max_acceleration_scaling_factor(self.max_acceleration_scaling)
        self.move_group.set_goal_position_tolerance(self.goal_position_tolerance)
        self.move_group.set_goal_orientation_tolerance(self.goal_orientation_tolerance)
        self.move_group.set_pose_reference_frame(self.base_frame)
        try:
            self.move_group.set_end_effector_link(self.ee_link)
        except Exception as e:
            rospy.logwarn("set_end_effector_link(%s) 失败，将继续使用 MoveIt 默认 ee_link。错误: %s", self.ee_link, str(e))

        rospy.sleep(1.0)

        # 预构建固定变换
        self.T_handeye_raw = pose_to_matrix_from_xyz_quat(self.handeye_xyz, self.handeye_quat)
        self.T_ee_cam = self.T_handeye_raw if self.handeye_result_is_ee_from_cam else inverse_matrix(self.T_handeye_raw)
        self.T_ee_tcp = pose_to_matrix_from_xyz_quat(self.ee_to_tcp_xyz, self.ee_to_tcp_quat)

        rospy.loginfo("Eye-on-hand 抓取执行节点初始化完成。")
        self.print_configuration_summary()

    def print_configuration_summary(self):
        rospy.loginfo("========== 配置摘要 ==========")
        rospy.loginfo("move_group                  : %s", self.move_group_name)
        rospy.loginfo("base_frame                  : %s", self.base_frame)
        rospy.loginfo("ee_link                     : %s", self.ee_link)
        rospy.loginfo("camera_frame                : %s", self.camera_frame)
        rospy.loginfo("handeye_result_is_ee_from_cam : %s", str(self.handeye_result_is_ee_from_cam))
        rospy.loginfo("handeye xyz                 : %s", str(self.handeye_xyz))
        rospy.loginfo("handeye quat                : %s", str(self.handeye_quat))
        rospy.loginfo("ee_to_tcp xyz               : %s", str(self.ee_to_tcp_xyz))
        rospy.loginfo("ee_to_tcp quat              : %s", str(self.ee_to_tcp_quat))
        rospy.loginfo("grasp xyz (camera frame)    : %s", str(self.grasp_xyz))
        rospy.loginfo("grasp quat (camera frame)   : %s", str(self.grasp_quat))
        rospy.loginfo("approach_axis_in_tcp        : %s", str(self.approach_axis_in_tcp.tolist()))
        rospy.loginfo("pregrasp_distance           : %.4f m", self.pregrasp_distance)
        rospy.loginfo("lift_distance               : %.4f m", self.lift_distance)
        rospy.loginfo("============================")

    def confirm(self, message):
        if not self.ask_before_execute:
            return True
        try:
            user_input = input("%s [y/N]: " % message).strip().lower()
        except EOFError:
            return False
        return user_input in ["y", "yes"]

    def get_current_T_base_ee(self):
        """从 TF 获取当前 T_base_ee。"""
        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_link,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            raise RuntimeError("获取 TF %s <- %s 失败: %s" % (self.base_frame, self.ee_link, str(e)))
        return transform_stamped_to_matrix(tf_msg)

    def get_manual_grasp_T_cam_grasp(self):
        """读取手动输入抓姿，返回 T_cam_grasp。"""
        return pose_to_matrix_from_xyz_quat(self.grasp_xyz, self.grasp_quat)

    def compute_grasp_related_targets(self):
        """
        计算抓取相关目标位姿。

        返回:
            T_base_grasp     : 目标 TCP/grasp 在 base 下的位姿
            T_base_ee_target : 目标 ee_link 在 base 下的位姿
            T_base_ee_pre    : pre-grasp 的 ee_link 位姿
            pose_grasp_ee    : geometry_msgs/Pose
            pose_pre_ee      : geometry_msgs/Pose
        """
        T_base_ee_current = self.get_current_T_base_ee()
        T_cam_grasp = self.get_manual_grasp_T_cam_grasp()

        # 当前相机位姿（由当前末端位姿 + 手眼标定得到）
        T_base_cam = np.dot(T_base_ee_current, self.T_ee_cam)

        # 目标 TCP/grasp 在 base 下的位姿
        T_base_grasp = np.dot(T_base_cam, T_cam_grasp)

        # 目标 ee_link 位姿
        T_base_ee_target = np.dot(T_base_grasp, inverse_matrix(self.T_ee_tcp))

        # pre-grasp：在 TCP 局部坐标下，沿逼近方向反向退回 pregrasp_distance
        pre_offset_local = -self.pregrasp_distance * self.approach_axis_in_tcp
        T_grasp_pre = make_local_translation(
            pre_offset_local[0],
            pre_offset_local[1],
            pre_offset_local[2],
        )
        T_base_pregrasp = np.dot(T_base_grasp, T_grasp_pre)
        T_base_ee_pre = np.dot(T_base_pregrasp, inverse_matrix(self.T_ee_tcp))

        pose_grasp_ee = matrix_to_pose_msg(T_base_ee_target)
        pose_pre_ee = matrix_to_pose_msg(T_base_ee_pre)
        return T_base_grasp, T_base_ee_target, T_base_ee_pre, pose_grasp_ee, pose_pre_ee

    def pose_to_pretty_string(self, pose_msg, name="pose"):
        q = [
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ]
        xyz = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
        s = (
            "%s:\n"
            "  position  = [%.6f, %.6f, %.6f]\n"
            "  quaternion= [%.6f, %.6f, %.6f, %.6f]"
            % (name, xyz[0], xyz[1], xyz[2], q[0], q[1], q[2], q[3])
        )
        return s

    def _extract_plan(self, plan_result):
        """
        兼容 MoveIt Python API 的不同返回风格。
        Noetic 常见返回: (success, plan, planning_time, error_code)
        也有版本直接返回 RobotTrajectory。
        """
        if isinstance(plan_result, tuple):
            if len(plan_result) >= 2:
                success = bool(plan_result[0])
                plan = plan_result[1]
                return success, plan
            return False, None

        plan = plan_result
        if plan is None:
            return False, None

        # 尽量通过关节轨迹点数判断
        try:
            success = len(plan.joint_trajectory.points) > 0
        except Exception:
            success = plan is not None
        return success, plan

    def plan_and_execute_pose(self, target_pose, description="target_pose"):
        """MoveIt 常规规划到某个 pose 并执行。"""
        rospy.loginfo("开始规划到 %s ...", description)
        rospy.loginfo("\n%s", self.pose_to_pretty_string(target_pose, description))

        self.move_group.set_start_state_to_current_state()
        self.move_group.clear_pose_targets()
        self.move_group.set_pose_target(target_pose, self.ee_link)

        plan_result = self.move_group.plan()
        success, plan = self._extract_plan(plan_result)
        if not success or plan is None:
            self.move_group.clear_pose_targets()
            raise RuntimeError("MoveIt 规划失败：%s" % description)

        if not self.confirm("确认执行到 %s 吗？" % description):
            self.move_group.clear_pose_targets()
            raise RuntimeError("用户取消执行：%s" % description)

        ok = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

        if not ok:
            raise RuntimeError("执行失败：%s" % description)
        rospy.loginfo("执行成功：%s", description)

    def cartesian_execute_to_pose(self, target_pose, description="cartesian_target"):
        """从当前位置直线（笛卡尔）运动到目标 pose。"""
        rospy.loginfo("开始笛卡尔路径规划到 %s ...", description)
        rospy.loginfo("\n%s", self.pose_to_pretty_string(target_pose, description))

        self.move_group.set_start_state_to_current_state()
        waypoints = [copy.deepcopy(target_pose)]
        plan, fraction = self.move_group.compute_cartesian_path(
            waypoints,
            self.eef_step,
            self.jump_threshold,
        )
        rospy.loginfo("笛卡尔路径 fraction = %.4f", fraction)
        if fraction < self.min_cartesian_fraction:
            raise RuntimeError(
                "笛卡尔路径覆盖率不足（fraction=%.4f < %.4f），拒绝执行：%s"
                % (fraction, self.min_cartesian_fraction, description)
            )

        if not self.confirm("确认执行笛卡尔运动到 %s 吗？" % description):
            raise RuntimeError("用户取消执行：%s" % description)

        ok = self.move_group.execute(plan, wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()
        if not ok:
            raise RuntimeError("笛卡尔执行失败：%s" % description)
        rospy.loginfo("笛卡尔执行成功：%s", description)

    def publish_gripper(self, value, description="gripper"):
        if self.gripper_pub is None:
            rospy.logwarn("[%s] 未配置实际夹爪控制 topic，仅日志输出，目标值=%.6f", description, value)
            rospy.sleep(self.gripper_command_sleep)
            return

        msg = Float64()
        msg.data = float(value)
        rospy.loginfo("发布夹爪命令 [%s] -> %.6f", description, msg.data)
        self.gripper_pub.publish(msg)
        rospy.sleep(self.gripper_command_sleep)

    def open_gripper(self):
        if not self.confirm("确认执行张开夹爪吗？"):
            raise RuntimeError("用户取消执行：open gripper")
        self.publish_gripper(self.gripper_open_value, description="open")

    def close_gripper(self):
        if not self.confirm("确认执行闭合夹爪吗？"):
            raise RuntimeError("用户取消执行：close gripper")
        self.publish_gripper(self.gripper_close_value, description="close")

    def compute_lift_pose_from_current(self):
        """基于当前 ee 位姿，在 base 坐标系下沿 +Z 抬升。"""
        current_pose = self.move_group.get_current_pose(self.ee_link).pose
        T_base_ee = pose_msg_to_matrix(current_pose)
        T_lift = np.dot(make_world_translation(0.0, 0.0, self.lift_distance), T_base_ee)
        return matrix_to_pose_msg(T_lift)

    def execute_once(self):
        rospy.loginfo("开始执行一次抓取流程。")

        # 1) 计算目标位姿
        T_base_grasp, T_base_ee_target, T_base_ee_pre, pose_grasp_ee, pose_pre_ee = self.compute_grasp_related_targets()

        grasp_xyz, grasp_quat = matrix_to_xyz_quat(T_base_grasp)
        rospy.loginfo(
            "目标 TCP/grasp 在 base 下位姿: xyz=%s quat=%s",
            str([round(v, 6) for v in grasp_xyz]),
            str([round(v, 6) for v in grasp_quat]),
        )
        rospy.loginfo("\n%s", self.pose_to_pretty_string(pose_pre_ee, "pre_grasp_ee"))
        rospy.loginfo("\n%s", self.pose_to_pretty_string(pose_grasp_ee, "grasp_ee"))

        # 2) 夹爪张开（可选）
        if self.open_gripper_before_start:
            self.open_gripper()

        # 3) 先运动到 pre-grasp
        self.plan_and_execute_pose(pose_pre_ee, description="pre_grasp")

        # 4) 直线逼近 grasp
        self.cartesian_execute_to_pose(pose_grasp_ee, description="grasp")

        # 5) 闭合夹爪
        self.close_gripper()

        # 6) 直线撤回到 pre-grasp
        self.cartesian_execute_to_pose(pose_pre_ee, description="retreat_to_pre_grasp")

        # 7) 抬升
        lift_pose = self.compute_lift_pose_from_current()
        self.plan_and_execute_pose(lift_pose, description="lift")

        rospy.loginfo("抓取流程执行完成。")


def main():
    try:
        executor = EyeOnHandGraspExecutor()
        executor.execute_once()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr("程序异常退出: %s", str(e))
        raise
    finally:
        try:
            moveit_commander.roscpp_shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()

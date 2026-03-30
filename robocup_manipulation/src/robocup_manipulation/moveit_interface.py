#!/usr/bin/env python3
"""
MoveIt接口封装
提供简化的机械臂控制接口
"""
import rospy
import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
import sys


class MoveItInterface:
    def __init__(self, group_name="piper_arm"):
        """
        初始化MoveIt接口
        Args:
            group_name: MoveGroup名称（默认"piper_arm"）
        """
        # 初始化moveit_commander
        moveit_commander.roscpp_initialize(sys.argv)

        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group_name)

        # 设置规划参数
        self.group.set_planning_time(5.0)
        self.group.set_max_velocity_scaling_factor(0.5)
        self.group.set_max_acceleration_scaling_factor(0.5)
        self.group.set_num_planning_attempts(10)

        rospy.loginfo(f"MoveIt接口已初始化")
        rospy.loginfo(f"  Planning Frame: {self.group.get_planning_frame()}")
        rospy.loginfo(f"  End Effector Link: {self.group.get_end_effector_link()}")
        rospy.loginfo(f"  Available Groups: {self.robot.get_group_names()}")

    def move_to_pose(self, target_pose, cartesian=False, wait=True):
        """
        移动到目标位姿
        Args:
            target_pose: geometry_msgs/Pose
            cartesian: 是否使用笛卡尔路径规划
            wait: 是否等待执行完成
        Returns:
            bool: 执行是否成功
        """
        rospy.loginfo(f"移动到目标位姿 (cartesian={cartesian})")

        if cartesian:
            # 笛卡尔路径规划
            waypoints = [target_pose]
            (plan, fraction) = self.group.compute_cartesian_path(
                waypoints,
                0.01,  # eef_step
                0.0    # jump_threshold
            )

            if fraction < 0.9:
                rospy.logwarn(f"笛卡尔路径规划不完整: {fraction*100:.1f}%")
                return False

            success = self.group.execute(plan, wait=wait)
        else:
            # 普通规划
            self.group.set_pose_target(target_pose)
            success = self.group.go(wait=wait)
            self.group.stop()
            self.group.clear_pose_targets()

        if success:
            rospy.loginfo("  ✓ 移动成功")
        else:
            rospy.logwarn("  ✗ 移动失败")

        return success

    def move_to_named_target(self, target_name):
        """
        移动到预定义姿态
        Args:
            target_name: 姿态名称（如"home", "transport"）
        Returns:
            bool: 执行是否成功
        """
        rospy.loginfo(f"移动到预定义姿态: {target_name}")

        try:
            self.group.set_named_target(target_name)
            success = self.group.go(wait=True)
            self.group.stop()

            if success:
                rospy.loginfo(f"  ✓ 成功到达 {target_name}")
            else:
                rospy.logwarn(f"  ✗ 未能到达 {target_name}")

            return success
        except Exception as e:
            rospy.logerr(f"移动到 {target_name} 失败: {e}")
            return False

    def get_current_pose(self):
        """获取当前末端位姿"""
        return self.group.get_current_pose().pose

    def get_current_joints(self):
        """获取当前关节角度"""
        return self.group.get_current_joint_values()

    def stop(self):
        """停止当前运动"""
        self.group.stop()
        rospy.loginfo("机械臂已停止")

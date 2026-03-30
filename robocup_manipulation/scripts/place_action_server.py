#!/usr/bin/env python3
"""
放置动作服务器
实现完整的放置流程：Approach → Open → Retract
"""
import rospy
import actionlib
from robocup_msgs.msg import PlaceObjectAction, PlaceObjectResult, PlaceObjectFeedback
from robocup_manipulation.moveit_interface import MoveItInterface
from robocup_manipulation.gripper_controller import GripperController
from geometry_msgs.msg import Pose
import copy


class PlaceActionServer:
    def __init__(self):
        rospy.init_node('place_action_server')

        # 初始化MoveIt和夹爪
        try:
            self.moveit = MoveItInterface(group_name="piper_arm")
            self.gripper = GripperController()
        except Exception as e:
            rospy.logerr(f"初始化失败: {e}")
            self.moveit = None
            self.gripper = None

        # 创建动作服务器
        self.server = actionlib.SimpleActionServer(
            '/place_object',
            PlaceObjectAction,
            execute_cb=self.execute_place,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo("✓ 放置动作服务器已启动")

    def execute_place(self, goal):
        """执行放置动作"""
        rospy.loginfo("=" * 50)
        rospy.loginfo("开始放置物体")
        rospy.loginfo("=" * 50)

        result = PlaceObjectResult()
        feedback = PlaceObjectFeedback()

        if self.moveit is None or self.gripper is None:
            result.success = False
            result.message = "MoveIt或夹爪未初始化"
            self.server.set_aborted(result)
            return

        # 如果没有提供有效目标位姿，使用当前位置下方
        if self._is_zero_pose(goal.target_pose):
            rospy.logwarn("未提供放置位姿，使用当前位置")
            current_pose = self.moveit.get_current_pose()
            target_pose = copy.deepcopy(current_pose)
            target_pose.position.z -= 0.15  # 下降15cm
        else:
            target_pose = goal.target_pose

        # 1. Approach - 移动到放置位置上方
        feedback.current_phase = "接近放置位置"
        feedback.progress = 0.3
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 1/4: 接近放置位置...")

        # 先移动到放置位置上方
        pre_place = copy.deepcopy(target_pose)
        pre_place.position.z += 0.15  # 抬高15cm

        if not self.moveit.move_to_pose(pre_place, cartesian=False):
            result.success = False
            result.message = "移动到放置位置上方失败"
            self.server.set_aborted(result)
            return

        # 2. 下降到放置位置
        feedback.current_phase = "下降到放置位置"
        feedback.progress = 0.6
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 2/4: 下降到放置位置...")

        if not self.moveit.move_to_pose(target_pose, cartesian=True):
            rospy.logwarn("笛卡尔路径失败，尝试普通规划...")
            if not self.moveit.move_to_pose(target_pose, cartesian=False):
                result.success = False
                result.message = "移动到放置位置失败"
                self.server.set_aborted(result)
                return

        # 3. Open Gripper - 释放物体
        feedback.current_phase = "释放物体"
        feedback.progress = 0.8
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 3/4: 打开夹爪释放物体...")
        self.gripper.open_gripper(width=0.07)
        rospy.sleep(0.5)  # 等待物体释放

        # 4. Retract - 抬起并返回home姿态
        feedback.current_phase = "返回初始姿态"
        feedback.progress = 0.9
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 4/4: 返回初始姿态...")

        # 先抬高一点
        post_place = copy.deepcopy(target_pose)
        post_place.position.z += 0.15
        self.moveit.move_to_pose(post_place, cartesian=True)

        # 返回home姿态
        self.moveit.move_to_named_target("home")

        # 成功
        feedback.progress = 1.0
        self.server.publish_feedback(feedback)

        result.success = True
        result.message = "放置成功"
        result.placed_pose = target_pose

        rospy.loginfo("=" * 50)
        rospy.loginfo("✓ 放置成功完成！")
        rospy.loginfo("=" * 50)

        self.server.set_succeeded(result)

    @staticmethod
    def _is_zero_pose(pose):
        return (
            pose is None or
            (
                pose.position.x == 0.0 and
                pose.position.y == 0.0 and
                pose.position.z == 0.0 and
                pose.orientation.x == 0.0 and
                pose.orientation.y == 0.0 and
                pose.orientation.z == 0.0 and
                pose.orientation.w == 0.0
            )
        )


if __name__ == '__main__':
    try:
        server = PlaceActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("放置动作服务器已停止")

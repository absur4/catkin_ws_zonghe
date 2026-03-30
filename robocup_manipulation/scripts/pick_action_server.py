#!/usr/bin/env python3
"""
抓取动作服务器
实现完整的抓取流程：Approach → Open → Grasp → Close → Lift → Retract
"""
import rospy
import actionlib
from robocup_msgs.msg import PickObjectAction, PickObjectResult, PickObjectFeedback
from robocup_msgs.srv import ComputeGraspPose, ComputeGraspPoseRequest
from robocup_manipulation.moveit_interface import MoveItInterface
from robocup_manipulation.gripper_controller import GripperController
from geometry_msgs.msg import Pose
import copy


class PickActionServer:
    def __init__(self):
        rospy.init_node('pick_action_server')

        # 初始化MoveIt和夹爪
        try:
            self.moveit = MoveItInterface(group_name="piper_arm")
            self.gripper = GripperController()
        except Exception as e:
            rospy.logerr(f"初始化失败: {e}")
            self.moveit = None
            self.gripper = None

        # 抓取姿态计算服务
        rospy.loginfo("等待抓取姿态计算服务...")
        # Note: 服务可能在运行时才可用，这里不阻塞

        # 创建动作服务器
        self.server = actionlib.SimpleActionServer(
            '/pick_object',
            PickObjectAction,
            execute_cb=self.execute_pick,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo("✓ 抓取动作服务器已启动")

    def compute_grasp_pose(self, target_object, grasp_strategy):
        """计算抓取姿态"""
        try:
            rospy.wait_for_service('/compute_grasp_pose', timeout=5.0)
            grasp_pose_srv = rospy.ServiceProxy('/compute_grasp_pose', ComputeGraspPose)

            req = ComputeGraspPoseRequest()
            req.target_object = target_object
            req.grasp_type = grasp_strategy

            resp = grasp_pose_srv(req)
            return resp.grasp_pose if resp.success else None

        except Exception as e:
            rospy.logerr(f"抓取姿态计算失败: {e}")
            # 返回简化的抓取姿态
            from robocup_msgs.msg import GraspPose
            grasp = GraspPose()

            # 使用物体位置作为抓取位置
            grasp.grasp = target_object.pose

            # 计算pre_grasp（抬高15cm）
            grasp.pre_grasp = copy.deepcopy(grasp.grasp)
            grasp.pre_grasp.position.z += 0.15

            # 计算post_grasp（抬高10cm）
            grasp.post_grasp = copy.deepcopy(grasp.grasp)
            grasp.post_grasp.position.z += 0.10

            grasp.gripper_width = 0.05
            grasp.approach_direction = "top"

            return grasp

    def execute_pick(self, goal):
        """执行抓取动作"""
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"开始抓取: {goal.target_object.class_name}")
        rospy.loginfo("=" * 50)

        result = PickObjectResult()
        feedback = PickObjectFeedback()

        if self.moveit is None or self.gripper is None:
            result.success = False
            result.message = "MoveIt或夹爪未初始化"
            self.server.set_aborted(result)
            return

        # 1. 计算抓取姿态
        feedback.current_phase = "计算抓取姿态"
        feedback.progress = 0.1
        self.server.publish_feedback(feedback)

        grasp_pose = self.compute_grasp_pose(goal.target_object, goal.grasp_strategy)

        if grasp_pose is None:
            result.success = False
            result.message = "抓取姿态计算失败"
            self.server.set_aborted(result)
            return

        rospy.loginfo("✓ 抓取姿态已计算")

        # 2. Approach - 移动到预抓取位姿
        feedback.current_phase = "接近物体"
        feedback.progress = 0.3
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 1/6: 接近物体...")
        if not self.moveit.move_to_pose(grasp_pose.pre_grasp, cartesian=False):
            result.success = False
            result.message = "移动到预抓取位姿失败"
            self.server.set_aborted(result)
            return

        # 3. Open Gripper
        feedback.current_phase = "打开夹爪"
        feedback.progress = 0.4
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 2/6: 打开夹爪...")
        self.gripper.open_gripper(width=grasp_pose.gripper_width + 0.02)
        rospy.sleep(0.5)  # 等待夹爪稳定

        # 4. Grasp - 移动到抓取位姿（笛卡尔路径）
        feedback.current_phase = "执行抓取"
        feedback.progress = 0.6
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 3/6: 下降到抓取位置...")
        if not self.moveit.move_to_pose(grasp_pose.grasp, cartesian=True):
            rospy.logwarn("笛卡尔路径失败，尝试普通规划...")
            if not self.moveit.move_to_pose(grasp_pose.grasp, cartesian=False):
                result.success = False
                result.message = "移动到抓取位姿失败"
                self.server.set_aborted(result)
                return

        # 5. Close Gripper
        feedback.current_phase = "闭合夹爪"
        feedback.progress = 0.7
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 4/6: 闭合夹爪...")

        # 根据物体类别选择力度
        if "food" in goal.target_object.category or "food" in goal.target_object.class_name.lower():
            effort = 40.0  # 轻柔抓取
        else:
            effort = 80.0  # 正常抓取

        self.gripper.close_gripper(effort=effort)
        rospy.sleep(1.0)  # 等待夹爪稳定

        # 6. Lift - 抬升
        feedback.current_phase = "抬升物体"
        feedback.progress = 0.85
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 5/6: 抬升物体...")
        if not self.moveit.move_to_pose(grasp_pose.post_grasp, cartesian=True):
            rospy.logwarn("抬升失败，尝试继续")

        # 7. Retract - 返回运输姿态
        feedback.current_phase = "返回运输姿态"
        feedback.progress = 0.95
        self.server.publish_feedback(feedback)

        rospy.loginfo("步骤 6/6: 返回运输姿态...")
        self.moveit.move_to_named_target("transport")

        # 成功
        feedback.progress = 1.0
        self.server.publish_feedback(feedback)

        result.success = True
        result.message = "抓取成功"
        result.picked_pose = grasp_pose.grasp

        rospy.loginfo("=" * 50)
        rospy.loginfo("✓ 抓取成功完成！")
        rospy.loginfo("=" * 50)

        self.server.set_succeeded(result)


if __name__ == '__main__':
    try:
        server = PickActionServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("抓取动作服务器已停止")

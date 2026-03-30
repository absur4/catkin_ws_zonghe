#!/usr/bin/env python3
"""
抓取姿态计算服务
根据物体信息计算抓取姿态（pre-grasp, grasp, post-grasp）
"""
import rospy
import copy
from robocup_msgs.srv import ComputeGraspPose, ComputeGraspPoseResponse
from robocup_msgs.msg import GraspPose
from geometry_msgs.msg import Pose
import tf.transformations as tf_trans


class GraspPoseComputer:
    def __init__(self):
        rospy.init_node('grasp_pose_computer')

        # 加载参数
        self.approach_distance = rospy.get_param('~approach_distance', 0.15)
        self.lift_height = rospy.get_param('~lift_height', 0.10)
        self.default_gripper_width = rospy.get_param('~default_gripper_width', 0.05)

        # 物体尺寸估计（简化版）
        self.object_sizes = {
            'cup': 0.04,
            'plate': 0.03,
            'bowl': 0.045,
            'spoon': 0.02,
            'fork': 0.02,
            'knife': 0.02,
            'glass': 0.04,
            'bottle': 0.035,
            'apple': 0.04,
            'banana': 0.03,
            'default': 0.04
        }

        # 创建服务
        self.service = rospy.Service('/compute_grasp_pose', ComputeGraspPose, self.handle_compute)
        rospy.loginfo("✓ 抓取姿态计算服务已启动")

    def estimate_gripper_width(self, object_name):
        """根据物体名称估计合适的夹爪宽度"""
        name_lower = object_name.lower()
        for obj_type, width in self.object_sizes.items():
            if obj_type in name_lower:
                return width + 0.02  # 留出2cm余量
        return self.default_gripper_width + 0.02

    def compute_approach_direction(self, grasp_type):
        """计算接近方向"""
        if grasp_type == "top":
            # 从上方接近，z轴向下
            return "top"
        elif grasp_type == "side":
            # 从侧面接近
            return "side"
        else:
            # 自动选择：默认从上方
            return "top"

    def handle_compute(self, req):
        """处理抓取姿态计算请求"""
        target_obj = req.target_object
        grasp_type = req.grasp_type if req.grasp_type else "auto"

        rospy.loginfo(f"计算抓取姿态: {target_obj.class_name}, 策略: {grasp_type}")

        # 创建响应
        resp = ComputeGraspPoseResponse()
        grasp_pose = GraspPose()

        # 1. 抓取位姿 = 物体位姿
        grasp_pose.grasp = copy.deepcopy(target_obj.pose)

        # 如果物体姿态没有设置旋转，设置默认姿态（夹爪垂直向下）
        if grasp_pose.grasp.orientation.w == 0 and \
           grasp_pose.grasp.orientation.x == 0 and \
           grasp_pose.grasp.orientation.y == 0 and \
           grasp_pose.grasp.orientation.z == 0:
            # 设置夹爪垂直向下的四元数 (绕Y轴旋转90度)
            q = tf_trans.quaternion_from_euler(0, 1.5708, 0)  # 90度
            grasp_pose.grasp.orientation.x = q[0]
            grasp_pose.grasp.orientation.y = q[1]
            grasp_pose.grasp.orientation.z = q[2]
            grasp_pose.grasp.orientation.w = q[3]

        # 2. 预抓取位姿（在抓取位姿上方）
        grasp_pose.pre_grasp = copy.deepcopy(grasp_pose.grasp)
        grasp_pose.pre_grasp.position.z += self.approach_distance

        # 3. 抓取后位姿（抓取后抬升）
        grasp_pose.post_grasp = copy.deepcopy(grasp_pose.grasp)
        grasp_pose.post_grasp.position.z += self.lift_height

        # 4. 夹爪宽度
        grasp_pose.gripper_width = self.estimate_gripper_width(target_obj.class_name)

        # 5. 接近方向
        grasp_pose.approach_direction = self.compute_approach_direction(grasp_type)

        # 6. 抓取质量评分（简化版，实际应该用GraspNet）
        grasp_pose.quality_score = 0.8

        resp.grasp_pose = grasp_pose
        resp.success = True
        resp.message = f"成功计算抓取姿态，夹爪宽度: {grasp_pose.gripper_width:.3f}m"

        rospy.loginfo(f"  抓取位置: ({grasp_pose.grasp.position.x:.2f}, "
                      f"{grasp_pose.grasp.position.y:.2f}, "
                      f"{grasp_pose.grasp.position.z:.2f})")
        rospy.loginfo(f"  预抓取高度: {grasp_pose.pre_grasp.position.z:.2f}m")
        rospy.loginfo(f"  夹爪宽度: {grasp_pose.gripper_width:.3f}m")

        return resp


if __name__ == '__main__':
    try:
        computer = GraspPoseComputer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("抓取姿态计算服务已停止")

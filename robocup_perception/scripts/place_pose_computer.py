#!/usr/bin/env python3
"""
放置姿态计算服务
根据目标层与已有物体布局计算放置位姿
"""
import rospy
from geometry_msgs.msg import Pose
from robocup_msgs.srv import ComputePlacePose, ComputePlacePoseResponse


class PlacePoseComputer:
    def __init__(self):
        rospy.init_node('place_pose_computer')

        # 放置参数（米）
        self.place_height_offset = rospy.get_param('~place_height_offset', 0.03)
        self.min_step = rospy.get_param('~min_step', 0.08)
        self.max_columns = rospy.get_param('~max_columns', 3)
        self.row_spacing = rospy.get_param('~row_spacing', 0.10)
        self.default_width = rospy.get_param('~default_object_width', 0.08)
        self.default_depth = rospy.get_param('~default_object_depth', 0.08)

        # 当没有目标层信息时的兜底位置（机器人前方桌面）
        self.fallback_x = rospy.get_param('~fallback_place_pose/x', 0.55)
        self.fallback_y = rospy.get_param('~fallback_place_pose/y', 0.00)
        self.fallback_z = rospy.get_param('~fallback_place_pose/z', 0.75)

        self.service = rospy.Service('/compute_place_pose', ComputePlacePose, self.handle_compute)
        rospy.loginfo("✓ 放置姿态计算服务已启动")

    @staticmethod
    def _is_zero_pose(pose):
        return (
            pose.position.x == 0.0 and
            pose.position.y == 0.0 and
            pose.position.z == 0.0 and
            pose.orientation.x == 0.0 and
            pose.orientation.y == 0.0 and
            pose.orientation.z == 0.0 and
            pose.orientation.w == 0.0
        )

    def _default_pose(self):
        pose = Pose()
        pose.position.x = self.fallback_x
        pose.position.y = self.fallback_y
        pose.position.z = self.fallback_z
        pose.orientation.w = 1.0
        return pose

    def _compute_on_layer(self, target_layer, existing_count, object_width, object_depth):
        pose = Pose()
        pose.position = target_layer.center_pose.position
        pose.orientation = target_layer.center_pose.orientation

        if self._is_zero_pose(target_layer.center_pose):
            return self._default_pose()

        # 避免与已放置物体重叠：按行列网格偏移
        step_x = max(object_width, self.min_step)
        step_y = max(object_depth, self.row_spacing)
        column = existing_count % self.max_columns
        row = existing_count // self.max_columns
        centered_col = column - (self.max_columns // 2)

        pose.position.x += centered_col * step_x
        pose.position.y += row * step_y
        pose.position.z += max(target_layer.height * 0.5, self.place_height_offset)

        if pose.orientation.w == 0.0 and pose.orientation.x == 0.0 and \
           pose.orientation.y == 0.0 and pose.orientation.z == 0.0:
            pose.orientation.w = 1.0

        return pose

    def handle_compute(self, req):
        resp = ComputePlacePoseResponse()

        existing_count = len(req.existing_objects.objects)
        object_width = req.object_width if req.object_width > 0 else self.default_width
        object_depth = req.object_depth if req.object_depth > 0 else self.default_depth

        if self._is_zero_pose(req.target_layer.center_pose):
            resp.place_pose = self._default_pose()
            resp.success = True
            resp.message = "目标层为空，已使用默认放置位姿"
        else:
            resp.place_pose = self._compute_on_layer(
                req.target_layer,
                existing_count,
                object_width,
                object_depth
            )
            resp.success = True
            resp.message = f"已基于目标层计算放置位姿（existing={existing_count}）"

        rospy.loginfo(
            f"放置位姿: ({resp.place_pose.position.x:.2f}, "
            f"{resp.place_pose.position.y:.2f}, {resp.place_pose.position.z:.2f})"
        )
        return resp


if __name__ == '__main__':
    try:
        PlacePoseComputer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("放置姿态计算服务已停止")

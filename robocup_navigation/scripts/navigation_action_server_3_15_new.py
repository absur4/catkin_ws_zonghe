#!/usr/bin/env python3
"""
虚拟导航动作服务器
不启动真实导航，仅模拟 /navigate_to_location 的成功/失败流程
"""
import os
import time
import yaml
import rospy
import actionlib
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationResult, NavigateToLocationFeedback


class MockNavigationActionServer:
    def __init__(self):
        rospy.init_node('navigation_action_server_mock')

        # 加载预定义位置（用于打印与返回位姿）
        locations_file = rospy.get_param('~locations_file', '')
        if not locations_file:
            locations_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'locations.yaml'
            )

        self.locations = {}
        try:
            if os.path.exists(locations_file):
                with open(locations_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.locations = config.get('locations', {})
                    rospy.loginfo(f"[MOCK NAV] 已加载 {len(self.locations)} 个预定义位置")
        except Exception as e:
            rospy.logwarn(f"[MOCK NAV] 加载位置配置失败: {e}")

        # 模拟配置
        self.fake_travel_time = rospy.get_param('~fake_travel_time', 2.0)
        self.accept_unknown = rospy.get_param('~accept_unknown', True)

        # 创建导航动作服务器
        self.server = actionlib.SimpleActionServer(
            '/navigate_to_location',
            NavigateToLocationAction,
            execute_cb=self.execute_navigation,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo("✓ 虚拟导航动作服务器已启动 (/navigate_to_location)")

    def execute_navigation(self, goal):
        location_name = goal.target_location
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"[MOCK NAV] 开始导航到: {location_name}")

        feedback = NavigateToLocationFeedback()
        result = NavigateToLocationResult()

        # 目标位姿（若已配置）
        final_pose = Pose()
        if location_name in self.locations:
            loc = self.locations[location_name]
            final_pose.position.x = loc['x']
            final_pose.position.y = loc['y']
            final_pose.position.z = 0.0
            quat = tf_trans.quaternion_from_euler(0, 0, loc['theta'])
            final_pose.orientation.x = quat[0]
            final_pose.orientation.y = quat[1]
            final_pose.orientation.z = quat[2]
            final_pose.orientation.w = quat[3]
        else:
            rospy.logwarn(f"[MOCK NAV] 未知位置: {location_name}")
            if not self.accept_unknown:
                result.success = False
                self.server.set_aborted(result)
                return

        # 模拟“导航中”
        start_time = time.time()
        while time.time() - start_time < self.fake_travel_time and not rospy.is_shutdown():
            if self.server.is_preempt_requested():
                rospy.loginfo("[MOCK NAV] 导航被上层取消")
                result.success = False
                self.server.set_preempted(result)
                return
            feedback.current_status = "导航中(虚拟)..."
            self.server.publish_feedback(feedback)
            rospy.sleep(0.2)

        result.success = True
        result.final_pose = final_pose
        rospy.loginfo(f"[MOCK NAV] ✓ 已到达 {location_name} (虚拟)")
        self.server.set_succeeded(result)


if __name__ == '__main__':
    server = MockNavigationActionServer()
    rospy.spin()

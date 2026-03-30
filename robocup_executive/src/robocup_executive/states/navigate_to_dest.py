#!/usr/bin/env python3
"""
NavigateToDest State - 导航到目的地状态
"""
import rospy
import smach
import actionlib
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal


class NavigateToDest(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['arrived', 'navigation_failed', 'fatal_error'],
            input_keys=['destination']
        )

        # 创建导航动作客户端
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )

    def execute(self, userdata):
        rospy.loginfo("========== 导航到目的地 ==========")

        if 'destination' not in userdata:
            rospy.logerr("未指定目的地")
            return 'fatal_error'

        destination = userdata['destination']
        rospy.loginfo(f"目的地: {destination}")

        # 等待动作服务器
        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("导航动作服务器未响应")
            return 'navigation_failed'

        # 创建目标
        goal = NavigateToLocationGoal()
        goal.target_location = destination

        rospy.loginfo(f"发送导航目标: {destination}")
        self.nav_client.send_goal(goal)

        # 等待结果（最多120秒）
        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))

        if not finished:
            rospy.logerr("导航超时")
            self.nav_client.cancel_goal()
            return 'navigation_failed'

        result = self.nav_client.get_result()

        if result and result.success:
            rospy.loginfo(f"✓ 成功到达 {destination}")
            return 'arrived'
        else:
            rospy.logwarn(f"✗ 导航到 {destination} 失败")
            return 'navigation_failed'

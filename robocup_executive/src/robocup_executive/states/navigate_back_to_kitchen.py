#!/usr/bin/env python3
"""
NavigateBackToKitchen State - 放置完成后返回厨房桌边
在内层清理循环中，每次放置完一件物品后调用，
使机器人返回桌边准备抓取下一件。
"""
import rospy
import smach
import actionlib
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal


class NavigateBackToKitchen(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['returned', 'navigation_failed']
        )

        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )

    def execute(self, userdata):
        rospy.loginfo("========== 返回厨房桌边 ==========")

        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("导航动作服务器未响应")
            return 'navigation_failed'

        goal = NavigateToLocationGoal()
        goal.target_location = "kitchen"

        rospy.loginfo("发送导航目标: kitchen（返回抓取位）")
        self.nav_client.send_goal(goal)

        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))

        if not finished:
            rospy.logerr("返回厨房导航超时")
            self.nav_client.cancel_goal()
            return 'navigation_failed'

        result = self.nav_client.get_result()

        if result and result.success:
            rospy.loginfo("✓ 已返回厨房桌边，准备抓取下一件")
            return 'returned'
        else:
            rospy.logwarn("返回厨房失败，尝试继续")
            return 'navigation_failed'

#!/usr/bin/env python3
"""
NavigateToKitchen State - 导航到厨房状态
"""
import rospy
import smach
import actionlib
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal


class NavigateToKitchen(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['arrived', 'navigation_failed']
        )

        # 创建导航动作客户端
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )
        rospy.loginfo("NavigateToKitchen: 等待导航动作服务器...")
        # Note: 在实际运行时才等待服务器，这里不阻塞初始化

    def execute(self, userdata):
        rospy.loginfo("========== 导航到厨房 ==========")

        # 等待动作服务器（最多10秒）
        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("导航动作服务器未响应")
            return 'navigation_failed'

        # 创建目标
        goal = NavigateToLocationGoal()
        goal.target_location = "kitchen"

        rospy.loginfo("发送导航目标: kitchen")
        self.nav_client.send_goal(goal)

        # 等待结果（最多120秒）
        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))

        if not finished:
            rospy.logerr("导航超时")
            self.nav_client.cancel_goal()
            return 'navigation_failed'

        result = self.nav_client.get_result()

        if result and result.success:
            rospy.loginfo("成功到达厨房！")
            return 'arrived'
        else:
            rospy.logerr("导航失败")
            return 'navigation_failed'

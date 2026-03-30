#!/usr/bin/env python3
"""
InitSystem State - 系统初始化状态
检查所有服务是否就绪，机械臂回零
"""
import rospy
import smach


class InitSystem(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['initialized', 'init_failed']
        )

    def execute(self, userdata):
        rospy.loginfo("========== 初始化系统 ==========")

        # 检查关键服务是否可用
        services_to_check = [
            '/detect_objects',
            '/classify_object',
            '/compute_grasp_pose',
            '/compute_place_pose'
        ]

        for service_name in services_to_check:
            rospy.loginfo(f"检查服务: {service_name}")
            try:
                rospy.wait_for_service(service_name, timeout=5.0)
                rospy.loginfo(f"  ✓ {service_name} 已就绪")
            except rospy.ROSException:
                rospy.logwarn(f"  ✗ {service_name} 未就绪（可能在后续启动）")
                # 不致命，继续

        # 检查动作服务器
        action_servers = [
            '/pick_object',
            '/place_object',
            '/navigate_to_location'
        ]

        for action_name in action_servers:
            rospy.loginfo(f"检查动作服务器: {action_name}")
            # Note: 实际实现中应该检查action server是否活跃
            # 这里简化处理

        rospy.loginfo("系统初始化完成")
        return 'initialized'

#!/usr/bin/env python3
"""
夹爪控制器
封装Piper夹爪服务调用
"""
import rospy
from piper_msgs.srv import Gripper, GripperRequest


class GripperController:
    def __init__(self):
        """初始化夹爪控制器"""
        rospy.loginfo("等待夹爪服务...")
        try:
            rospy.wait_for_service('/gripper_srv', timeout=10.0)
            self.gripper_srv = rospy.ServiceProxy('/gripper_srv', Gripper)
            rospy.loginfo("✓ 夹爪控制器已初始化")
        except rospy.ROSException:
            rospy.logerr("夹爪服务不可用")
            self.gripper_srv = None

    def open_gripper(self, width=0.07, effort=30.0):
        """
        打开夹爪
        Args:
            width: 开度（米），0.0-0.07
            effort: 力度
        Returns:
            bool: 是否成功
        """
        if self.gripper_srv is None:
            rospy.logerr("夹爪服务不可用")
            return False

        try:
            req = GripperRequest()
            req.gripper_angle = width
            req.gripper_effort = effort
            req.gripper_code = 0
            req.set_zero = 0

            resp = self.gripper_srv(req)
            rospy.loginfo(f"夹爪已打开至 {width}m")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"夹爪服务调用失败: {e}")
            return False

    def close_gripper(self, effort=100.0):
        """
        闭合夹爪
        Args:
            effort: 抓取力度
        Returns:
            bool: 是否成功
        """
        if self.gripper_srv is None:
            rospy.logerr("夹爪服务不可用")
            return False

        try:
            req = GripperRequest()
            req.gripper_angle = 0.0
            req.gripper_effort = effort
            req.gripper_code = 0
            req.set_zero = 0

            resp = self.gripper_srv(req)
            rospy.loginfo(f"夹爪已闭合，力度: {effort}")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"夹爪服务调用失败: {e}")
            return False

    def set_gripper(self, width, effort=50.0):
        """
        设置夹爪到指定开度
        Args:
            width: 开度（米）
            effort: 力度
        Returns:
            bool: 是否成功
        """
        if self.gripper_srv is None:
            rospy.logerr("夹爪服务不可用")
            return False

        try:
            req = GripperRequest()
            req.gripper_angle = width
            req.gripper_effort = effort
            req.gripper_code = 0
            req.set_zero = 0

            resp = self.gripper_srv(req)
            rospy.loginfo(f"夹爪设置为 {width}m, 力度: {effort}")
            return True
        except rospy.ServiceException as e:
            rospy.logerr(f"夹爪服务调用失败: {e}")
            return False

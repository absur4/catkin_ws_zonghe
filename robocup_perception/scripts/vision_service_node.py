#!/usr/bin/env python3
"""
视觉服务主节点 - 初始化视觉模型并管理所有感知服务
"""
import rospy

if __name__ == '__main__':
    rospy.init_node('vision_service_node')
    rospy.loginfo("Vision service node started (placeholder)")
    rospy.spin()

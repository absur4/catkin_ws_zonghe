#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import time
import random
import numpy as np
from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlRequest
from tf.transformations import quaternion_from_euler, quaternion_from_matrix

def call_joint_moveit_ctrl_arm(joint_states, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_arm")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_arm", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        request.joint_states = joint_states
        request.gripper = 0.0
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_arm")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_arm, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

def call_joint_moveit_ctrl_gripper(gripper_position, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_gripper")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_gripper", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        request.joint_states = [0.0] * 6
        request.gripper = gripper_position
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_gripper")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_gripper, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")

def convert_endpose(endpose):
    if len(endpose) == 6:
        x, y, z, roll, pitch, yaw = endpose
        qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)
        return [x, y, z, qx, qy, qz, qw]

    elif len(endpose) == 7:
        return endpose  # 直接返回四元数

    else:
        raise ValueError("Invalid endpose format! Must be 6 (Euler) or 7 (Quaternion) values.")

def call_joint_moveit_ctrl_endpose(endpose, max_velocity=0.5, max_acceleration=0.5):
    rospy.wait_for_service("joint_moveit_ctrl_endpose")
    try:
        moveit_service = rospy.ServiceProxy("joint_moveit_ctrl_endpose", JointMoveitCtrl)
        request = JointMoveitCtrlRequest()
        
        request.joint_states = [0.0] * 6  # 填充6个关节状态
        request.gripper = 0.0
        request.max_velocity = max_velocity
        request.max_acceleration = max_acceleration
        request.joint_endpose = convert_endpose(endpose)  # 自动转换

        response = moveit_service(request)
        if response.status:
            rospy.loginfo("Successfully executed joint_moveit_ctrl_endpose")
        else:
            rospy.logwarn(f"Failed to execute joint_moveit_ctrl_endpose, error code: {response.error_code}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {str(e)}")


def main():
    rospy.init_node("piper_rotate_and_grasp_with_home", anonymous=True)
    
    # ==========================================
    # === 核心修改：提取矩阵平移位置，强制姿态为水平向前 ===
    # 视觉矩阵位置：X=0.62981397, Y=-0.11709712, Z=0.33634236
    # 强制姿态：Roll=0.0, Pitch=1.5708 (向下倾斜90度使Z轴水平), Yaw=0.0
    # ==========================================
    pre_grasp_pose = [
        0.49981397, 
        0.002709712, 
        0.08634236, 
        0.0, 1.5708, 0.0
    ]
    
    rospy.loginfo("开始执行任务...")
    rospy.loginfo(f"计算出的抓取位姿 (X, Y, Z, Roll, Pitch, Yaw): {pre_grasp_pose}")

    # 步骤 1: 张开夹爪
    rospy.loginfo("步骤 1: 张开夹爪")
    call_joint_moveit_ctrl_gripper(0.035) 
    time.sleep(1)

    # 步骤 2: 执行抓取位置和姿态
    rospy.loginfo("步骤 2: 移动至指定位姿...")
    call_joint_moveit_ctrl_endpose(pre_grasp_pose, max_velocity=0.3)
    time.sleep(1)

    rospy.loginfo("任务完成！")

if __name__ == "__main__":
    main()
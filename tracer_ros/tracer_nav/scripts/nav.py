#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# ================= 配置区域：在这里修改你的目标坐标 =================
GOAL_X = -2.0  # 目标 X 坐标 (米)
GOAL_Y = 3.5  # 目标 Y 坐标 (米)
GOAL_W = 1.0  # 目标朝向 (四元数 w，1.0 表示保持起始朝向)
# ===============================================================

def move_to_goal(x, y, w):
    # 1. 初始化 ROS 节点
    rospy.init_node('simple_navigation_script', anonymous=True)

    # 2. 创建 move_base 的 Action 客户端
    # 这里的 'move_base' 必须与你的 launch 文件中的节点名称一致 
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    rospy.loginfo("等待 move_base 服务器响应...")
    wait = client.wait_for_server(rospy.Duration(5.0))
    if not wait:
        rospy.logerr("未能连接到 move_base 服务器，请检查 move_base 是否启动！")
        return

    # 3. 构建目标点
    goal = MoveBaseGoal()
    # 这里的 frame_id 必须与 global_costmap 中的 global_frame 一致，通常是 'map' 
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    # 设置具体位置
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    # 设置朝向 (这里简单处理，只设置 w 轴)
    goal.target_pose.pose.orientation.w = w

    # 4. 发送目标点
    rospy.loginfo("正在发送目标坐标: X=%s, Y=%s", x, y)
    client.send_goal(goal)

    # 5. 等待执行结果
    rospy.loginfo("正在前往目的地...")
    client.wait_for_result()

    # 6. 判断最终状态
    if client.get_state() == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo("恭喜！底盘已成功到达目的地。")
    else:
        rospy.loginfo("导航失败，请检查代价地图是否有障碍物阻挡。")

if __name__ == '__main__':
    try:
        move_to_goal(GOAL_X, GOAL_Y, GOAL_W)
    except rospy.ROSInterruptException:
        rospy.loginfo("导航脚本已停止。")
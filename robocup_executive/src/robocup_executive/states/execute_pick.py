#!/usr/bin/env python3
"""
ExecutePick State - 执行抓取状态
"""
import rospy
import smach
import actionlib
from robocup_msgs.msg import PickObjectAction, PickObjectGoal


class ExecutePick(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['pick_succeeded', 'pick_failed', 'fatal_error'],
            input_keys=['selected_object', 'object_category', 'objects_picked_count', 'failed_objects'],
            output_keys=['grasp_pose', 'objects_picked_count', 'failed_objects']
        )

        # 创建动作客户端
        self.pick_client = actionlib.SimpleActionClient(
            '/pick_object',
            PickObjectAction
        )
        rospy.loginfo("ExecutePick: 等待pick_object动作服务器...")

    def execute(self, userdata):
        rospy.loginfo("========== 执行抓取 ==========")

        if 'selected_object' not in userdata:
            rospy.logerr("未选择目标物品")
            return 'fatal_error'

        obj = userdata['selected_object']
        rospy.loginfo(f"目标物品: {obj.class_name}")

        # 等待动作服务器
        if not self.pick_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("抓取动作服务器未响应")
            return 'fatal_error'

        # 构建目标
        goal = PickObjectGoal()
        goal.target_object = obj
        goal.grasp_strategy = "adaptive"

        rospy.loginfo("发送抓取目标...")
        self.pick_client.send_goal(goal)

        # 等待结果（最多60秒）
        finished = self.pick_client.wait_for_result(timeout=rospy.Duration(60.0))

        if not finished:
            rospy.logerr("抓取动作超时")
            self.pick_client.cancel_goal()
            failed = list(userdata.get('failed_objects', []))
            failed.append(obj.class_name)
            userdata['failed_objects'] = failed
            return 'pick_failed'

        result = self.pick_client.get_result()

        if result and result.success:
            rospy.loginfo("✓ 抓取成功！")
            userdata['grasp_pose'] = result.picked_pose
            userdata['objects_picked_count'] = userdata.get('objects_picked_count', 0) + 1
            return 'pick_succeeded'
        else:
            msg = result.message if result else "未知错误"
            rospy.logwarn(f"✗ 抓取失败: {msg}")
            failed = list(userdata.get('failed_objects', []))
            failed.append(obj.class_name)
            userdata['failed_objects'] = failed
            return 'pick_failed'

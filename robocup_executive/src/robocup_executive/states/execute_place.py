#!/usr/bin/env python3
"""
ExecutePlace State - 执行放置状态
"""
import rospy
import smach
import actionlib
from std_msgs.msg import String
from robocup_msgs.msg import PlaceObjectAction, PlaceObjectGoal


class ExecutePlace(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['place_succeeded', 'place_failed', 'fatal_error'],
            input_keys=['place_pose', 'selected_object', 'objects_placed_count', 'destination', 'failed_objects'],
            output_keys=['objects_placed_count', 'failed_objects']
        )

        # 创建动作客户端
        self.place_client = actionlib.SimpleActionClient(
            '/place_object',
            PlaceObjectAction
        )
        rospy.loginfo("ExecutePlace: 等待place_object动作服务器...")

        # TTS 发布者（规则书 Rule #4：洗碗机门由 referee 打开）
        self.tts_pub = rospy.Publisher('/tts/say', String, queue_size=1)

    def execute(self, userdata):
        rospy.loginfo("========== 执行放置 ==========")

        destination = userdata.get('destination', '')

        # 规则书 Rule #4：洗碗机默认关闭，需通知 referee 开门
        if destination == 'dishwasher':
            rospy.logwarn("[洗碗机门] 请 referee 打开洗碗机门")
            self.tts_pub.publish(String(data="Please open the dishwasher door."))
            rospy.sleep(5.0)  # 等待 referee 响应

        # 等待动作服务器
        if not self.place_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("放置动作服务器未响应")
            return 'fatal_error'

        # 构建目标
        goal = PlaceObjectGoal()

        if 'place_pose' in userdata and userdata['place_pose']:
            goal.target_pose = userdata['place_pose']
        else:
            rospy.logwarn("未计算放置姿态，使用当前位置")
            # 在实际实现中，这里应该有一个默认的放置姿态

        # 区分洗碗机与普通平面放置策略（合法值："gentle" / "fast"）
        goal.place_strategy = "fast" if destination == 'dishwasher' else "gentle"

        rospy.loginfo("发送放置目标...")
        self.place_client.send_goal(goal)

        # 等待结果（最多60秒）
        finished = self.place_client.wait_for_result(timeout=rospy.Duration(60.0))

        if not finished:
            rospy.logerr("放置动作超时")
            self.place_client.cancel_goal()
            selected = userdata.get('selected_object', None)
            if selected is not None:
                failed = list(userdata.get('failed_objects', []))
                failed.append(selected.class_name)
                userdata['failed_objects'] = failed
            return 'place_failed'

        result = self.place_client.get_result()

        if result and result.success:
            rospy.loginfo("✓ 放置成功！")

            # 更新统计
            if 'objects_placed_count' not in userdata:
                userdata['objects_placed_count'] = 0
            userdata['objects_placed_count'] += 1

            return 'place_succeeded'
        else:
            msg = result.message if result else "未知错误"
            rospy.logwarn(f"✗ 放置失败: {msg}")
            selected = userdata.get('selected_object', None)
            if selected is not None:
                failed = list(userdata.get('failed_objects', []))
                failed.append(selected.class_name)
                userdata['failed_objects'] = failed
            return 'place_failed'

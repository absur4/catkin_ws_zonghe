#!/usr/bin/env python3
"""
SelectTarget State - 选择下一个目标物品状态
"""
import rospy
import smach
from robocup_msgs.srv import ClassifyObject, ClassifyObjectRequest


class SelectTarget(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['target_selected', 'no_more_objects', 'failed'],
            input_keys=['objects_to_pick', 'current_object_index'],
            output_keys=['selected_object', 'object_category', 'destination', 'current_object_index']
        )

    def execute(self, userdata):
        rospy.loginfo("========== 选择目标物品 ==========")

        # 检查是否还有物品待处理
        if 'objects_to_pick' not in userdata or len(userdata['objects_to_pick']) == 0:
            rospy.loginfo("没有更多物品需要处理")
            return 'no_more_objects'

        # 获取下一个物品
        if 'current_object_index' not in userdata:
            userdata['current_object_index'] = 0

        index = userdata['current_object_index']

        if index >= len(userdata['objects_to_pick']):
            rospy.loginfo("所有物品已处理完毕")
            return 'no_more_objects'

        selected_obj = userdata['objects_to_pick'][index]
        rospy.loginfo(f"选择物品 [{index + 1}/{len(userdata['objects_to_pick'])}]: {selected_obj.class_name}")

        # 调用分类服务确定目的地
        try:
            rospy.wait_for_service('/classify_object', timeout=5.0)
            classify_srv = rospy.ServiceProxy('/classify_object', ClassifyObject)

            req = ClassifyObjectRequest()
            req.object_name = selected_obj.class_name

            resp = classify_srv(req)

            if resp.success:
                rospy.loginfo(f"  类别: {resp.category}")
                rospy.loginfo(f"  目的地: {resp.destination}")

                userdata['selected_object'] = selected_obj
                userdata['object_category'] = resp.category
                userdata['destination'] = resp.destination
                userdata['current_object_index'] = index + 1  # 为下次准备

                return 'target_selected'
            else:
                rospy.logwarn("分类失败，跳过此物品")
                userdata['current_object_index'] = index + 1
                return 'failed'

        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"分类服务调用失败: {e}")
            userdata['current_object_index'] = index + 1  # 跳过此物品，避免无限重试
            return 'failed'

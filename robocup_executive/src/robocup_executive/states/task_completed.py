#!/usr/bin/env python3
"""
TaskCompleted State - 任务完成状态
显示统计信息并结束任务
"""
import rospy
import smach


class TaskCompleted(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['done'],
            input_keys=['objects_picked_count', 'objects_placed_count', 'failed_objects']
        )

    def execute(self, userdata):
        rospy.loginfo("=" * 60)
        rospy.loginfo(" " * 15 + "任务完成！")
        rospy.loginfo("=" * 60)

        # 显示统计信息
        picked = userdata.get('objects_picked_count', 0)
        placed = userdata.get('objects_placed_count', 0)
        failed = userdata.get('failed_objects', [])

        rospy.loginfo(f"抓取物品数: {picked}")
        rospy.loginfo(f"放置物品数: {placed}")

        if len(failed) > 0:
            rospy.loginfo(f"失败物品数: {len(failed)}")
            rospy.loginfo(f"失败物品列表: {', '.join(failed)}")

        rospy.loginfo("=" * 60)

        return 'done'

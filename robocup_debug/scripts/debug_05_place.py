#!/usr/bin/env python3
"""
debug_05_place.py — place_object 动作全流程调试脚本

先决条件：
  /place_object 动作服务器 + 机械臂（假设机器人已手持物体）

用法：
  rosrun robocup_debug debug_05_place.py --x 0.45 --y 0.0 --z 0.55
  rosrun robocup_debug debug_05_place.py --x 0.45 --y 0.0 --z 0.55 --strategy dishwasher

功能：
  - 目标位姿通过 --x --y --z 指定（必填，防止意外放置）
  - 策略通过 --strategy 指定（默认 gentle）
  - 实时打印 feedback，完成后打印 placed_pose
  - dishwasher 策略时检验 /tts/say 消息是否发出
"""
import sys
import argparse
import rospy
import actionlib
from std_msgs.msg import String
from geometry_msgs.msg import Pose
from robocup_msgs.msg import PlaceObjectAction, PlaceObjectGoal

VALID_STRATEGIES = ['gentle', 'fast', 'dishwasher', 'breakfast_layout']


class PlaceDebugger:
    def __init__(self, x, y, z, strategy):
        self.x = x
        self.y = y
        self.z = z
        self.strategy = strategy

        self.place_client = actionlib.SimpleActionClient('/place_object', PlaceObjectAction)

        self.tts_received = None
        if strategy == 'dishwasher':
            rospy.Subscriber('/tts/say', String, self._tts_cb)

    def _tts_cb(self, msg):
        self.tts_received = msg.data
        rospy.loginfo(f"  [TTS] 收到消息: \"{msg.data}\"")

    def _feedback_cb(self, feedback):
        rospy.loginfo(
            f"  [FEEDBACK] phase={feedback.current_phase:<15} "
            f"progress={feedback.progress:.0%}"
        )

    def run(self):
        rospy.loginfo(
            f"目标放置位置: ({self.x}, {self.y}, {self.z})m  策略: {self.strategy}"
        )

        rospy.loginfo("等待 /place_object 动作服务器（超时 15s）...")
        if not self.place_client.wait_for_server(timeout=rospy.Duration(15.0)):
            rospy.logerr("/place_object 动作服务器未响应")
            rospy.logerr("请先运行：rosrun robocup_manipulation place_action_server.py")
            sys.exit(1)
        rospy.loginfo("/place_object 服务器已就绪 ✓")

        target_pose = Pose()
        target_pose.position.x = self.x
        target_pose.position.y = self.y
        target_pose.position.z = self.z
        target_pose.orientation.w = 1.0

        goal = PlaceObjectGoal()
        goal.target_pose = target_pose
        goal.place_strategy = self.strategy

        if self.strategy == 'dishwasher':
            rospy.loginfo("洗碗机策略：等待 /tts/say 消息（请确认 ExecutePlace 会发送 TTS）...")
            rospy.sleep(0.5)

        rospy.loginfo(f"\n发送 PlaceObjectGoal: strategy={self.strategy}")
        rospy.loginfo("--- feedback 流 ---")
        self.place_client.send_goal(goal, feedback_cb=self._feedback_cb)

        finished = self.place_client.wait_for_result(timeout=rospy.Duration(90.0))

        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        rospy.loginfo("  PlaceObject 最终结果")
        rospy.loginfo("=" * 60)

        if not finished:
            rospy.logerr("动作超时（90s）")
            self.place_client.cancel_goal()
            return

        result = self.place_client.get_result()
        if result is None:
            rospy.logerr("未收到结果")
            return

        rospy.loginfo(f"success : {result.success}")
        rospy.loginfo(f"message : {result.message}")
        if result.success:
            p = result.placed_pose.position
            q = result.placed_pose.orientation
            rospy.loginfo(
                f"placed_pose: pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  "
                f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
            )

        if self.strategy == 'dishwasher':
            rospy.loginfo("")
            if self.tts_received:
                rospy.loginfo(f"TTS 验证: ✓  收到: \"{self.tts_received}\"")
            else:
                rospy.logwarn("TTS 验证: ✗  未收到 /tts/say 消息（Rule #4 洗碗机通知）")

        rospy.loginfo("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='debug_05_place: place_object 动作调试')
    parser.add_argument('--x', type=float, required=True, help='目标 x 坐标（米）')
    parser.add_argument('--y', type=float, required=True, help='目标 y 坐标（米）')
    parser.add_argument('--z', type=float, required=True, help='目标 z 坐标（米）')
    parser.add_argument('--strategy', default='gentle',
                        choices=VALID_STRATEGIES,
                        help='放置策略（默认 gentle）')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node('debug_05_place', anonymous=True)
    rospy.loginfo("== debug_05_place 启动 ==")

    debugger = PlaceDebugger(args.x, args.y, args.z, args.strategy)
    debugger.run()
    rospy.loginfo("== debug_05_place 完成 ==")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

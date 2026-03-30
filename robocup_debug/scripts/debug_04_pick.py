#!/usr/bin/env python3
"""
debug_04_pick.py — pick_object 动作全流程调试脚本

先决条件：
  /pick_object 动作服务器 + 机械臂 + /detect_objects + 相机

用法：
  rosrun robocup_debug debug_04_pick.py              # 自动取最高置信度物体
  rosrun robocup_debug debug_04_pick.py --class cup  # 指定类别

功能：
  1. 先检测场景，选置信度最高的物体（或按 --class 指定）
  2. 发送真实 PickObjectGoal（grasp_strategy="adaptive"）
  3. 实时打印 feedback：current_phase + progress
  4. 完成后打印 success、message、picked_pose
"""
import sys
import argparse
import rospy
import actionlib
from sensor_msgs.msg import Image
from robocup_msgs.msg import PickObjectAction, PickObjectGoal
from robocup_msgs.srv import DetectObjects, DetectObjectsRequest

TARGET_CLASSES = [
    'cup', 'plate', 'bowl', 'spoon', 'fork', 'knife', 'glass', 'mug',
    'apple', 'banana', 'bread', 'bottle'
]
IMG_TIMEOUT = 10.0


class PickDebugger:
    def __init__(self, class_filter=None):
        self.class_filter = class_filter
        self.rgb_image = None
        self.depth_image = None
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self._depth_cb)

        self.pick_client = actionlib.SimpleActionClient('/pick_object', PickObjectAction)

    def _rgb_cb(self, msg):
        self.rgb_image = msg

    def _depth_cb(self, msg):
        self.depth_image = msg

    def _wait_for_images(self):
        rospy.loginfo(f"等待相机图像（超时 {IMG_TIMEOUT}s）...")
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(IMG_TIMEOUT)
        while (self.rgb_image is None or self.depth_image is None) and rospy.Time.now() < deadline:
            rate.sleep()
        if self.rgb_image is None or self.depth_image is None:
            rospy.logerr("相机图像超时")
            sys.exit(1)

    def _detect_target(self):
        rospy.loginfo("等待 /detect_objects 服务...")
        try:
            rospy.wait_for_service('/detect_objects', timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("/detect_objects 不可用")
            sys.exit(1)

        srv = rospy.ServiceProxy('/detect_objects', DetectObjects)
        req = DetectObjectsRequest()
        classes = [self.class_filter] if self.class_filter else TARGET_CLASSES
        req.target_classes = classes
        req.rgb_image = self.rgb_image
        req.depth_image = self.depth_image
        req.confidence_threshold = 0.3

        rospy.loginfo(f"检测目标类别: {classes}")
        try:
            resp = srv(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"检测失败: {e}")
            sys.exit(1)

        if not resp.success or len(resp.detected_objects.objects) == 0:
            rospy.logwarn("未检测到目标物体")
            sys.exit(0)

        objects = resp.detected_objects.objects
        rospy.loginfo(f"检测到 {len(objects)} 个物体:")
        for o in objects:
            rospy.loginfo(
                f"  {o.class_name:<12} confidence={o.confidence:.2f}  "
                f"pos=({o.pose.position.x:.3f}, {o.pose.position.y:.3f}, {o.pose.position.z:.3f})"
            )

        best = max(objects, key=lambda o: o.confidence)
        rospy.loginfo(f"\n选择: {best.class_name} (confidence={best.confidence:.2f})")
        return best

    def _feedback_cb(self, feedback):
        rospy.loginfo(
            f"  [FEEDBACK] phase={feedback.current_phase:<15} "
            f"progress={feedback.progress:.0%}"
        )

    def run(self):
        self._wait_for_images()
        target = self._detect_target()

        rospy.loginfo("\n等待 /pick_object 动作服务器（超时 15s）...")
        if not self.pick_client.wait_for_server(timeout=rospy.Duration(15.0)):
            rospy.logerr("/pick_object 动作服务器未响应")
            rospy.logerr("请先运行：rosrun robocup_manipulation pick_action_server.py")
            sys.exit(1)
        rospy.loginfo("/pick_object 服务器已就绪 ✓")

        goal = PickObjectGoal()
        goal.target_object = target
        goal.grasp_strategy = "adaptive"

        rospy.loginfo(f"\n发送 PickObjectGoal: target={target.class_name}, strategy=adaptive")
        rospy.loginfo("--- feedback 流 ---")
        self.pick_client.send_goal(goal, feedback_cb=self._feedback_cb)

        finished = self.pick_client.wait_for_result(timeout=rospy.Duration(90.0))

        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        rospy.loginfo("  PickObject 最终结果")
        rospy.loginfo("=" * 60)

        if not finished:
            rospy.logerr("动作超时（90s）")
            self.pick_client.cancel_goal()
            return

        result = self.pick_client.get_result()
        if result is None:
            rospy.logerr("未收到结果")
            return

        rospy.loginfo(f"success : {result.success}")
        rospy.loginfo(f"message : {result.message}")
        if result.success:
            p = result.picked_pose.position
            q = result.picked_pose.orientation
            rospy.loginfo(
                f"picked_pose: pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  "
                f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
            )
        rospy.loginfo("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='debug_04_pick: pick_object 动作调试')
    parser.add_argument('--class', dest='obj_class', default=None,
                        help='指定要抓取的物品类别（不指定则取最高置信度）')
    # rospy 可能传入 __name 等参数，需过滤
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node('debug_04_pick', anonymous=True)
    rospy.loginfo("== debug_04_pick 启动 ==")
    if args.obj_class:
        rospy.loginfo(f"指定类别: {args.obj_class}")

    debugger = PickDebugger(class_filter=args.obj_class)
    debugger.run()
    rospy.loginfo("== debug_04_pick 完成 ==")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

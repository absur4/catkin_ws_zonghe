#!/usr/bin/env python3
"""
debug_02_perception.py — 物体检测服务调试脚本

先决条件：
  object_detector_server.py + 相机节点已启动

功能：
  1. 等待相机图像（超时 10s）
  2. 调用真实 /detect_objects 服务
  3. 打印每个检测结果（class_name、confidence、base_link 坐标）
  4. 验证 pose.position.z 是否在合理范围（0.3~1.2m）
"""
import sys
import rospy
from sensor_msgs.msg import Image
from robocup_msgs.srv import DetectObjects, DetectObjectsRequest

# 检测目标类别（从 task_config.yaml 对齐）
TARGET_CLASSES = [
    'cup', 'plate', 'bowl', 'spoon', 'fork', 'knife', 'glass', 'mug',
    'apple', 'banana', 'bread', 'milk', 'juice', 'bottle',
    'wrapper', 'tissue', 'napkin', 'paper'
]

Z_MIN = 0.3   # base_link 坐标 z 合理下限（米）
Z_MAX = 1.2   # base_link 坐标 z 合理上限（米）
IMG_TIMEOUT = 10.0


class PerceptionDebugger:
    def __init__(self):
        self.rgb_image = None
        self.depth_image = None
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self._depth_cb)

    def _rgb_cb(self, msg):
        self.rgb_image = msg

    def _depth_cb(self, msg):
        self.depth_image = msg

    def wait_for_images(self, timeout=IMG_TIMEOUT):
        rospy.loginfo(f"等待相机图像（超时 {timeout}s）...")
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while (self.rgb_image is None or self.depth_image is None) and rospy.Time.now() < deadline:
            rate.sleep()
        if self.rgb_image is None:
            rospy.logerr("/camera/color/image_raw 超时，未接收到 RGB 图像")
            return False
        if self.depth_image is None:
            rospy.logerr("/camera/aligned_depth_to_color/image_raw 超时，未接收到深度图像")
            return False
        rospy.loginfo("相机图像就绪 ✓")
        return True

    def run(self):
        if not self.wait_for_images():
            sys.exit(1)

        rospy.loginfo("等待 /detect_objects 服务...")
        try:
            rospy.wait_for_service('/detect_objects', timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("/detect_objects 服务不可用（超时 10s）")
            rospy.logerr("请先运行：rosrun robocup_perception object_detector_server.py")
            sys.exit(1)

        srv = rospy.ServiceProxy('/detect_objects', DetectObjects)
        req = DetectObjectsRequest()
        req.target_classes = TARGET_CLASSES
        req.rgb_image = self.rgb_image
        req.depth_image = self.depth_image
        req.confidence_threshold = 0.3

        rospy.loginfo(f"调用 /detect_objects，目标类别: {len(TARGET_CLASSES)} 个...")
        try:
            resp = srv(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"/detect_objects 调用失败: {e}")
            sys.exit(1)

        rospy.loginfo("")
        rospy.loginfo("=" * 70)
        rospy.loginfo("  /detect_objects 结果")
        rospy.loginfo("=" * 70)

        if not resp.success:
            rospy.logwarn(f"服务返回失败: {resp.message}")
            return

        objects = resp.detected_objects.objects
        rospy.loginfo(f"检测到 {len(objects)} 个物体\n")

        anomaly_count = 0
        for i, obj in enumerate(objects):
            x = obj.pose.position.x
            y = obj.pose.position.y
            z = obj.pose.position.z

            z_ok = Z_MIN <= z <= Z_MAX
            z_flag = "✓" if z_ok else f"⚠ 超范围（期望 {Z_MIN}~{Z_MAX}m）"
            if not z_ok:
                anomaly_count += 1

            rospy.loginfo(
                f"[{i+1}] {obj.class_name:<12} "
                f"confidence={obj.confidence:.2f}  "
                f"base_link=({x:.3f}, {y:.3f}, {z:.3f})m  {z_flag}"
            )

        rospy.loginfo("")
        if anomaly_count == 0:
            rospy.loginfo("所有物体坐标均在合理范围 ✓（TF 变换正常）")
        else:
            rospy.logwarn(
                f"{anomaly_count} 个物体坐标超出范围，"
                "请检查 camera_color_optical_frame→base_link TF 变换"
            )

        rospy.loginfo(f"\n服务消息: {resp.message}")
        rospy.loginfo("=" * 70)

        # 返回检测结果供其他脚本复用
        return resp.detected_objects


def main():
    rospy.init_node('debug_02_perception', anonymous=True)
    rospy.loginfo("== debug_02_perception 启动 ==")
    debugger = PerceptionDebugger()
    debugger.run()
    rospy.loginfo("== debug_02_perception 完成 ==")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

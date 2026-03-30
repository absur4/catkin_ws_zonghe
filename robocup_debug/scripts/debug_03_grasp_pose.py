#!/usr/bin/env python3
"""
debug_03_grasp_pose.py — 抓取姿态计算服务调试脚本

先决条件：
  /compute_grasp_pose 服务 + /detect_objects 服务 + 相机已启动

功能：
  1. 调用 /detect_objects，取置信度最高的物体
  2. 用该真实检测结果调用 /compute_grasp_pose
  3. 打印 pre_grasp、grasp、post_grasp 三点坐标和 gripper_width
"""
import sys
import rospy
from sensor_msgs.msg import Image
from robocup_msgs.srv import DetectObjects, DetectObjectsRequest
from robocup_msgs.srv import ComputeGraspPose, ComputeGraspPoseRequest

TARGET_CLASSES = [
    'cup', 'plate', 'bowl', 'spoon', 'fork', 'knife', 'glass', 'mug',
    'apple', 'banana', 'bread', 'bottle'
]
IMG_TIMEOUT = 10.0


class GraspPoseDebugger:
    def __init__(self):
        self.rgb_image = None
        self.depth_image = None
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self._depth_cb)

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
            rospy.logerr("相机图像超时，退出")
            sys.exit(1)
        rospy.loginfo("相机图像就绪 ✓")

    def _detect_best_object(self):
        rospy.loginfo("等待 /detect_objects 服务...")
        try:
            rospy.wait_for_service('/detect_objects', timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("/detect_objects 不可用")
            sys.exit(1)

        srv = rospy.ServiceProxy('/detect_objects', DetectObjects)
        req = DetectObjectsRequest()
        req.target_classes = TARGET_CLASSES
        req.rgb_image = self.rgb_image
        req.depth_image = self.depth_image
        req.confidence_threshold = 0.3

        rospy.loginfo("调用 /detect_objects...")
        try:
            resp = srv(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"检测失败: {e}")
            sys.exit(1)

        if not resp.success or len(resp.detected_objects.objects) == 0:
            rospy.logwarn("未检测到任何物体，无法继续")
            sys.exit(0)

        # 取置信度最高的物体
        best = max(resp.detected_objects.objects, key=lambda o: o.confidence)
        rospy.loginfo(
            f"选择最高置信度物体: {best.class_name} "
            f"(confidence={best.confidence:.2f}, "
            f"base_link pos=({best.pose.position.x:.3f}, "
            f"{best.pose.position.y:.3f}, "
            f"{best.pose.position.z:.3f}))"
        )
        return best

    def run(self):
        self._wait_for_images()
        target_obj = self._detect_best_object()

        rospy.loginfo("等待 /compute_grasp_pose 服务...")
        try:
            rospy.wait_for_service('/compute_grasp_pose', timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("/compute_grasp_pose 不可用")
            sys.exit(1)

        srv = rospy.ServiceProxy('/compute_grasp_pose', ComputeGraspPose)
        req = ComputeGraspPoseRequest()
        req.target_object = target_obj
        req.grasp_type = "adaptive"

        rospy.loginfo("调用 /compute_grasp_pose (grasp_type=adaptive)...")
        try:
            resp = srv(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"/compute_grasp_pose 调用失败: {e}")
            sys.exit(1)

        rospy.loginfo("")
        rospy.loginfo("=" * 70)
        rospy.loginfo("  /compute_grasp_pose 结果")
        rospy.loginfo("=" * 70)

        if not resp.success:
            rospy.logwarn(f"服务失败: {resp.message}")
            return

        gp = resp.grasp_pose

        def fmt_pose(p):
            pos = p.position
            ori = p.orientation
            return (f"  pos=({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})  "
                    f"quat=({ori.x:.3f}, {ori.y:.3f}, {ori.z:.3f}, {ori.w:.3f})")

        rospy.loginfo(f"pre_grasp:\n{fmt_pose(gp.pre_grasp)}")
        rospy.loginfo(f"grasp:\n{fmt_pose(gp.grasp)}")
        rospy.loginfo(f"post_grasp:\n{fmt_pose(gp.post_grasp)}")
        rospy.loginfo(f"gripper_width: {gp.gripper_width:.4f} m")
        rospy.loginfo(f"approach_direction: {gp.approach_direction}")
        rospy.loginfo(f"服务消息: {resp.message}")
        rospy.loginfo("=" * 70)


def main():
    rospy.init_node('debug_03_grasp_pose', anonymous=True)
    rospy.loginfo("== debug_03_grasp_pose 启动 ==")
    debugger = GraspPoseDebugger()
    debugger.run()
    rospy.loginfo("== debug_03_grasp_pose 完成 ==")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

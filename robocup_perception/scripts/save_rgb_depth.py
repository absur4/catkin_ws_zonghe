#!/usr/bin/env python3

import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber

class RGBDepthSaver:

    def __init__(self):
        rospy.init_node("rgb_depth_saver")

        self.bridge = CvBridge()
        self.index = 0
        self.save_triggered = False  # 保存触发标志

        # 保存目录
        self.rgb_dir = "rgb"
        self.depth_dir = "depth"

        # 创建目录（如果不存在）
        os.makedirs(self.rgb_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)

        # 订阅RGB和Depth话题
        rgb_sub = Subscriber("/camera/color/image_raw", Image)
        depth_sub = Subscriber("/camera/aligned_depth_to_color/image_raw", Image)

        # 时间同步器：同步RGB和Depth图像
        ats = ApproximateTimeSynchronizer([rgb_sub, depth_sub], 10, 0.1)
        ats.registerCallback(self.callback)

        # 创建一个定时器，定期检查保存触发参数
        self.timer = rospy.Timer(rospy.Duration(0.1), self.check_save_trigger)

        rospy.loginfo("RGB + Depth saver started (单次保存模式)")
        rospy.loginfo("提示：设置参数 /rgb_depth_saver/save_one: true 来保存一张图像对")

        rospy.spin()

    def check_save_trigger(self, event):
        """检查是否触发了保存指令"""
        # 读取ROS参数，判断是否需要保存
        if rospy.has_param("/rgb_depth_saver/save_one"):
            self.save_triggered = rospy.get_param("/rgb_depth_saver/save_one")
            # 如果触发了保存，立即重置参数（避免重复保存）
            if self.save_triggered:
                rospy.set_param("/rgb_depth_saver/save_one", False)

    def callback(self, rgb_msg, depth_msg):
        """图像回调函数：仅当触发保存时才执行保存操作"""
        # 只有触发保存标志为True时，才保存图像
        if not self.save_triggered:
            return

        try:
            # 将ROS图像消息转换为OpenCV格式
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            # Depth图像保持原始格式（16位深度值）
            depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")

            # 生成带索引的文件名（6位数字，补零）
            rgb_name = f"{self.rgb_dir}/rgb_{self.index:06d}.png"
            depth_name = f"{self.depth_dir}/depth_{self.index:06d}.png"

            # 保存图像
            cv2.imwrite(rgb_name, rgb)
            cv2.imwrite(depth_name, depth)

            rospy.loginfo(f"成功保存第 {self.index} 张图像对")
            rospy.loginfo(f"RGB: {rgb_name} | Depth: {depth_name}")

            # 索引自增，保存标志重置
            self.index += 1
            self.save_triggered = False

        except Exception as e:
            rospy.logerr(f"保存图像失败: {e}")
            self.save_triggered = False  # 出错时也重置标志

if __name__ == "__main__":
    try:
        RGBDepthSaver()
    except rospy.ROSInterruptException:
        rospy.loginfo("节点被中断")
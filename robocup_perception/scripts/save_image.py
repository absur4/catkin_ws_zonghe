#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(msg):
    bridge = CvBridge()
    # 将ROS图像消息转换为OpenCV图像
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    # 保存图片
    cv2.imwrite("/home/songfei/pictures/capture.png", cv_image)
    rospy.loginfo("保存图像成功！")
    rospy.signal_shutdown("Saved one image and exit.")  # 保存一帧后退出

def listener():
    rospy.init_node('save_image_node', anonymous=True)
    rospy.Subscriber("/camera/color/image_raw", Image, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()

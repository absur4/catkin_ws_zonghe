#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

bridge = CvBridge()
count = 0

def callback(msg):
    global count

    depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    filename = "/home/songfei/depth_%d.png" % count

    cv2.imwrite(filename, depth)

    print("saved:", filename)

    count += 1

rospy.init_node("save_depth")

rospy.Subscriber(
    "/camera/depth/image_rect_raw",
    Image,
    callback
)

rospy.spin()
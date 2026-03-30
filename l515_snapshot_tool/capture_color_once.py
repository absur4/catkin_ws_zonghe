#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2


class ColorCaptureOnce:
    def __init__(self):
        rospy.init_node("capture_color_once", anonymous=True)

        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.output_dir = rospy.get_param("~output_dir", "./output")
        self.timeout = rospy.get_param("~timeout", 10.0)

        self.bridge = CvBridge()
        self.color_saved = False

        os.makedirs(self.output_dir, exist_ok=True)

        rospy.loginfo("Waiting for color image on topic: %s", self.image_topic)
        self.sub = rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)

    def image_callback(self, msg):
        if self.color_saved:
            return

        try:
            color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge conversion failed: %s", str(e))
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"{timestamp}_color.png")

        ok = cv2.imwrite(save_path, color_img)
        if not ok:
            rospy.logerr("Failed to save image to: %s", save_path)
            rospy.signal_shutdown("Save failed")
            return

        rospy.loginfo("Color image saved to: %s", save_path)
        self.color_saved = True
        rospy.signal_shutdown("Capture complete")

    def run(self):
        start_time = time.time()
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            if self.color_saved:
                break

            if time.time() - start_time > self.timeout:
                rospy.logerr("Timeout: no color image received within %.1f seconds", self.timeout)
                rospy.signal_shutdown("Timeout")
                break

            rate.sleep()


if __name__ == "__main__":
    try:
        node = ColorCaptureOnce()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        sys.exit(0)

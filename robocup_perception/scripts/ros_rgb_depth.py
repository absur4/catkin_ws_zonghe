  #!/usr/bin/env python3
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    rospy.init_node("save_one_frame_rgbd")
    bridge = CvBridge()

    rgb_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=5.0)
    depth_msg = rospy.wait_for_message("/camera/depth/image_rect_raw", Image, timeout=5.0)

    rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")

    cv2.imwrite("color.png", rgb)
    np.save("depth.npy", depth)

    rospy.loginfo("Saved color.png and depth.npy")

if __name__ == "__main__":
    main()


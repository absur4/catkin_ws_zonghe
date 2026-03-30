#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import CameraInfo

color_done = False
depth_done = False
aligned_done = False


def print_intrinsics(name, msg):
    K = msg.K
    D = msg.D

    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]

    print("\n==============================")
    print("Camera:", name)
    print("Resolution:", msg.width, "x", msg.height)

    print("\nIntrinsic Matrix K:")
    print("[{:.3f}, 0, {:.3f}]".format(fx, cx))
    print("[0, {:.3f}, {:.3f}]".format(fy, cy))
    print("[0, 0, 1]")

    print("\nfx =", fx)
    print("fy =", fy)
    print("cx =", cx)
    print("cy =", cy)

    print("\nDistortion Model:", msg.distortion_model)
    print("Distortion Coefficients:", D)

    print("==============================\n")


def color_callback(msg):
    global color_done
    if not color_done:
        print_intrinsics("COLOR CAMERA", msg)
        color_done = True
        check_done()


def depth_callback(msg):
    global depth_done
    if not depth_done:
        print_intrinsics("DEPTH CAMERA", msg)
        depth_done = True
        check_done()


def aligned_callback(msg):
    global aligned_done
    if not aligned_done:
        print_intrinsics("ALIGNED DEPTH TO COLOR", msg)
        aligned_done = True
        check_done()


def check_done():
    if color_done and depth_done and aligned_done:
        print("\n所有相机参数已经获取完成\n")
        rospy.signal_shutdown("done")


rospy.init_node("camera_intrinsics_reader")

rospy.Subscriber("/camera/color/camera_info", CameraInfo, color_callback)
rospy.Subscriber("/camera/depth/camera_info", CameraInfo, depth_callback)
rospy.Subscriber("/camera/aligned_depth_to_color/camera_info", CameraInfo, aligned_callback)

rospy.spin()

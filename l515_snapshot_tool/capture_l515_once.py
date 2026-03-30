#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
启动 L515 ROS 节点，抓取一对彩色图/深度图，并保存到输出目录。

默认行为：
1. 若 ROS master 未运行，则自动启动 roscore。
2. 自动执行: roslaunch realsense2_camera l515.launch
3. 订阅:
   - /camera/color/image_raw
   - /camera/aligned_depth_to_color/image_raw
4. 获取一对近似同步图像后保存并退出。

输出文件：
- color_<timestamp>.png
- depth_<timestamp>.png        (原始深度，优先保存为 16-bit PNG)
- depth_vis_<timestamp>.png    (便于查看的伪彩/归一化可视化图)
- depth_<timestamp>.npy        (原始深度 numpy 数据)

使用前提：
- 已在终端 source 过 ROS 与你的工作空间环境
- 你的工作空间中已经存在: realsense2_camera/l515.launch
- 已正确安装 rospy / cv_bridge / sensor_msgs / message_filters / OpenCV
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
import xmlrpc.client
from typing import Optional

import cv2
import numpy as np

import rospy
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class CaptureOnceNode:
    def __init__(
        self,
        output_dir: str,
        color_topic: str,
        depth_topic: str,
        launch_pkg: str,
        launch_file: str,
        timeout_sec: float,
        skip_launch: bool,
        startup_delay: float,
    ) -> None:
        self.output_dir = output_dir
        self.color_topic = color_topic
        self.depth_topic = depth_topic
        self.launch_pkg = launch_pkg
        self.launch_file = launch_file
        self.timeout_sec = timeout_sec
        self.skip_launch = skip_launch
        self.startup_delay = startup_delay

        self.bridge = CvBridge()
        self.capture_event = threading.Event()
        self.saved_prefix: Optional[str] = None
        self.error: Optional[str] = None

        self.roscore_proc: Optional[subprocess.Popen] = None
        self.roslaunch_proc: Optional[subprocess.Popen] = None
        self.started_roscore = False

        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _is_ros_master_running() -> bool:
        master_uri = os.environ.get("ROS_MASTER_URI", "http://127.0.0.1:11311")
        try:
            proxy = xmlrpc.client.ServerProxy(master_uri)
            proxy.getSystemState("/capture_l515_once")
            return True
        except Exception:
            return False

    @staticmethod
    def _terminate_process_tree(proc: Optional[subprocess.Popen], name: str) -> None:
        if proc is None or proc.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:
            print(f"[WARN] 终止 {name} 时出现异常: {exc}")
            return

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=3)
            except Exception as exc:
                print(f"[WARN] 强制终止 {name} 失败: {exc}")

    def start_roscore_if_needed(self) -> None:
        if self._is_ros_master_running():
            print("[INFO] 检测到 roscore 已在运行，复用当前 ROS master。")
            return

        print("[INFO] 未检测到 ROS master，正在启动 roscore ...")
        log_path = os.path.join(self.output_dir, "roscore.log")
        with open(log_path, "ab") as _:
            pass
        log_file = open(log_path, "ab")
        self.roscore_proc = subprocess.Popen(
            ["roscore"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        self.started_roscore = True

        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self._is_ros_master_running():
                print("[INFO] roscore 已启动。")
                return
            time.sleep(0.5)

        raise RuntimeError("roscore 启动超时，请检查 ROS 安装是否正常。")

    def start_camera_launch_if_needed(self) -> None:
        if self.skip_launch:
            print("[INFO] 已指定 --skip-launch，假定 L515 节点已在外部启动。")
            return

        print(f"[INFO] 正在启动相机节点: roslaunch {self.launch_pkg} {self.launch_file}")
        log_path = os.path.join(self.output_dir, "roslaunch_l515.log")
        with open(log_path, "ab") as _:
            pass
        log_file = open(log_path, "ab")
        self.roslaunch_proc = subprocess.Popen(
            ["roslaunch", self.launch_pkg, self.launch_file],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        # 给驱动和设备一些启动时间，避免一上来就订阅不到。
        time.sleep(self.startup_delay)

        if self.roslaunch_proc.poll() is not None:
            raise RuntimeError(
                "roslaunch 进程已提前退出。请检查输出目录中的 roslaunch_l515.log。"
            )

    @staticmethod
    def _make_depth_visualization(depth: np.ndarray) -> np.ndarray:
        valid_mask = np.isfinite(depth) & (depth > 0)
        if not np.any(valid_mask):
            return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

        valid_depth = depth[valid_mask].astype(np.float32)
        d_min = np.percentile(valid_depth, 2)
        d_max = np.percentile(valid_depth, 98)
        if d_max <= d_min:
            d_max = float(valid_depth.max())
            d_min = float(valid_depth.min())
            if d_max <= d_min:
                return np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

        depth_clipped = np.clip(depth.astype(np.float32), d_min, d_max)
        depth_norm = ((depth_clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
        depth_norm[~valid_mask] = 0
        return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    def _save_images(self, color_msg: Image, depth_msg: Image) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(self.output_dir, timestamp)

        try:
            color_img = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            raise RuntimeError(f"CvBridge 转换失败: {exc}")

        if color_img is None or depth_img is None:
            raise RuntimeError("收到的图像为空。")

        # 保存彩色图
        color_path = f"{prefix}_color.png"
        if not cv2.imwrite(color_path, color_img):
            raise RuntimeError(f"保存彩色图失败: {color_path}")

        # 保存原始深度数组，确保数据不丢失
        depth_npy_path = f"{prefix}_depth.npy"
        np.save(depth_npy_path, depth_img)

        # 尽量保存原始深度 PNG
        depth_png_path = f"{prefix}_depth.png"
        depth_to_save = depth_img
        if depth_to_save.dtype == np.float32 or depth_to_save.dtype == np.float64:
            # 若为 float 深度，为保留信息仍保存 npy；PNG 这里转成毫米级 uint16
            finite_mask = np.isfinite(depth_to_save)
            depth_mm = np.zeros_like(depth_to_save, dtype=np.float32)
            depth_mm[finite_mask] = np.clip(depth_to_save[finite_mask] * 1000.0, 0, 65535)
            depth_to_save = depth_mm.astype(np.uint16)
        elif depth_to_save.dtype not in (np.uint16, np.uint8):
            depth_to_save = depth_to_save.astype(np.uint16)

        if not cv2.imwrite(depth_png_path, depth_to_save):
            raise RuntimeError(f"保存深度图失败: {depth_png_path}")

        # 保存便于人眼查看的深度可视化图
        depth_vis = self._make_depth_visualization(depth_img)
        depth_vis_path = f"{prefix}_depth_vis.png"
        if not cv2.imwrite(depth_vis_path, depth_vis):
            raise RuntimeError(f"保存深度可视化图失败: {depth_vis_path}")

        self.saved_prefix = prefix
        print("[INFO] 图像保存成功:")
        print(f"       彩色图: {color_path}")
        print(f"       深度图: {depth_png_path}")
        print(f"       深度可视化: {depth_vis_path}")
        print(f"       深度原始数据: {depth_npy_path}")

    def _sync_callback(self, color_msg: Image, depth_msg: Image) -> None:
        if self.capture_event.is_set():
            return
        try:
            self._save_images(color_msg, depth_msg)
            self.capture_event.set()
        except Exception as exc:
            self.error = str(exc)
            self.capture_event.set()

    def run(self) -> int:
        self.start_roscore_if_needed()
        self.start_camera_launch_if_needed()

        rospy.init_node("capture_l515_once", anonymous=True)

        print(f"[INFO] 订阅彩色话题: {self.color_topic}")
        print(f"[INFO] 订阅深度话题: {self.depth_topic}")

        color_sub = message_filters.Subscriber(self.color_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)
        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=20, slop=0.2
        )
        ats.registerCallback(self._sync_callback)

        start = time.time()
        rate = rospy.Rate(20)
        print("[INFO] 等待同步图像并执行一次拍照 ...")
        while not rospy.is_shutdown() and not self.capture_event.is_set():
            if time.time() - start > self.timeout_sec:
                self.error = (
                    f"等待图像超时（>{self.timeout_sec:.1f}s）。请检查话题是否存在，"
                    f"以及 launch 中是否启用了 color/aligned depth。"
                )
                self.capture_event.set()
                break
            rate.sleep()

        if self.error is not None:
            print(f"[ERROR] {self.error}")
            return 1

        return 0

    def cleanup(self) -> None:
        self._terminate_process_tree(self.roslaunch_proc, "roslaunch")
        if self.started_roscore:
            self._terminate_process_tree(self.roscore_proc, "roscore")


def parse_args() -> argparse.Namespace:
    default_output = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

    parser = argparse.ArgumentParser(
        description="自动启动 L515 ROS 节点，抓取一帧彩色图和深度图并保存。"
    )
    parser.add_argument(
        "--output-dir",
        default=default_output,
        help=f"输出目录，默认: {default_output}",
    )
    parser.add_argument(
        "--color-topic",
        default="/camera/color/image_raw",
        help="彩色图话题，默认: /camera/color/image_raw",
    )
    parser.add_argument(
        "--depth-topic",
        default="/camera/aligned_depth_to_color/image_raw",
        help="深度图话题，默认: /camera/aligned_depth_to_color/image_raw",
    )
    parser.add_argument(
        "--launch-pkg",
        default="realsense2_camera",
        help="roslaunch 包名，默认: realsense2_camera",
    )
    parser.add_argument(
        "--launch-file",
        default="l515.launch",
        help="launch 文件名，默认: l515.launch",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=40.0,
        help="等待图像超时时间（秒），默认: 40",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=8.0,
        help="启动 roslaunch 后额外等待秒数，默认: 8",
    )
    parser.add_argument(
        "--skip-launch",
        action="store_true",
        help="若 L515 节点已经在外部启动，则不再重复 roslaunch。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    node = CaptureOnceNode(
        output_dir=os.path.abspath(args.output_dir),
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        launch_pkg=args.launch_pkg,
        launch_file=args.launch_file,
        timeout_sec=args.timeout,
        skip_launch=args.skip_launch,
        startup_delay=args.startup_delay,
    )

    try:
        return node.run()
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，准备退出。")
        return 130
    except Exception as exc:
        print(f"[ERROR] 程序执行失败: {exc}")
        return 1
    finally:
        node.cleanup()


if __name__ == "__main__":
    sys.exit(main())

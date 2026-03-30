#!/usr/bin/env python3
"""
视觉检测可视化节点
订阅 RGB/Depth，调用 Grounding DINO，发布带框与置信度的图像
"""
import os
import sys
import tempfile
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Header


# Open Vocabulary API 导入
WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OPEN_VOCAB_API = os.path.join(WS_ROOT, 'open_vocabulary', 'open_vocabulary', 'API')
DINO_CONFIG_PATH = os.path.join(
    WS_ROOT,
    'open_vocabulary',
    'open_vocabulary',
    'Grounded-SAM-2',
    'grounding_dino',
    'groundingdino',
    'config',
    'GroundingDINO_SwinT_OGC_3_15_new.py'
)
if OPEN_VOCAB_API not in sys.path:
    sys.path.append(OPEN_VOCAB_API)

try:
    from grounding_dino_api import GroundingDINOAPI
    DINO_AVAILABLE = True
except Exception as e:
    rospy.logwarn(f"[VISION VIEW] 无法导入 GroundingDINOAPI: {e}")
    DINO_AVAILABLE = False


class VisionBBoxViewer:
    def __init__(self):
        rospy.init_node('vision_bbox_viewer')

        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_rect_raw')
        self.annotated_topic = rospy.get_param('~annotated_topic', '/vision/annotated_image_3_15_new')
        self.detection_rate_hz = rospy.get_param('~detection_rate_hz', 1.0)
        self.depth_scale = rospy.get_param('~depth_scale', 0.001)

        self.target_classes = rospy.get_param('~detection/target_classes', [
            'cup', 'plate', 'spoon', 'fork', 'knife', 'bowl',
            'apple', 'banana', 'bread', 'bottle',
            'wrapper', 'tissue', 'napkin'
        ])
        self.confidence_threshold = rospy.get_param('~detection/confidence_threshold', 0.3)

        self.bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None

        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgb_cb, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_cb, queue_size=1)

        self.pub = rospy.Publisher(self.annotated_topic, Image, queue_size=1)

        self.dino_api = None
        if DINO_AVAILABLE:
            try:
                rospy.loginfo("[VISION VIEW] 初始化 GroundingDINOAPI...")
                if not os.path.exists(DINO_CONFIG_PATH):
                    rospy.logwarn(f"[VISION VIEW] GroundingDINO config 不存在: {DINO_CONFIG_PATH}")
                self.dino_api = GroundingDINOAPI(
                    device="cuda",
                    config_path=DINO_CONFIG_PATH if os.path.exists(DINO_CONFIG_PATH) else None
                )
                rospy.loginfo("[VISION VIEW] ✓ GroundingDINOAPI 已初始化")
            except Exception as e:
                rospy.logerr(f"[VISION VIEW] GroundingDINOAPI 初始化失败: {e}")
                self.dino_api = None

        self.timer = rospy.Timer(rospy.Duration(1.0 / max(self.detection_rate_hz, 0.1)), self._on_timer)
        rospy.loginfo(f"[VISION VIEW] 订阅 {self.rgb_topic} / {self.depth_topic}")
        rospy.loginfo(f"[VISION VIEW] 发布 {self.annotated_topic}")

    def _rgb_cb(self, msg):
        self.rgb_image = msg

    def _depth_cb(self, msg):
        self.depth_image = msg

    def _build_prompt(self, classes):
        prompt = ". ".join([c.strip().lower() for c in classes if c.strip()])
        if not prompt.endswith('.'):
            prompt += '.'
        return prompt

    def _depth_at(self, depth_cv, x, y):
        h, w = depth_cv.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        depth = depth_cv[y, x]
        if depth > 0:
            return float(depth) * self.depth_scale
        # 5x5 邻域补偿
        y0 = max(0, y - 2)
        y1 = min(h, y + 3)
        x0 = max(0, x - 2)
        x1 = min(w, x + 3)
        neighborhood = depth_cv[y0:y1, x0:x1]
        nonzero = neighborhood[neighborhood > 0]
        if nonzero.size == 0:
            return None
        return float(np.median(nonzero)) * self.depth_scale

    def _on_timer(self, _event):
        if self.rgb_image is None:
            return

        try:
            rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")
        except Exception:
            return

        depth_cv = None
        if self.depth_image is not None:
            try:
                depth_cv = self.bridge.imgmsg_to_cv2(self.depth_image, "16UC1")
            except Exception:
                depth_cv = None

        annotated = rgb_cv.copy()

        if self.dino_api is None:
            # 无法使用模型时，直接发布原图
            self._publish_image(annotated)
            return

        fd_rgb, temp_rgb = tempfile.mkstemp(prefix='vision_view_rgb_', suffix='.jpg')
        os.close(fd_rgb)
        cv2.imwrite(temp_rgb, rgb_cv)

        try:
            prompt = self._build_prompt(self.target_classes)
            results = self.dino_api.detect(
                image_path=temp_rgb,
                text_prompt=prompt,
                output_path=None,
                box_threshold=None,
                text_threshold=None
            )

            boxes = results.get("boxes", [])
            confidences = results.get("confidences", [])
            labels = results.get("labels", [])

            for box, conf, label in zip(boxes, confidences, labels):
                if conf < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                depth_m = None
                if depth_cv is not None:
                    depth_m = self._depth_at(depth_cv, cx, cy)

                label_clean = str(label).lower().replace('.', '').strip()
                if depth_m is not None:
                    text = f"{label_clean} {conf:.2f} {depth_m:.2f}m"
                else:
                    text = f"{label_clean} {conf:.2f}"

                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(annotated, text, (x1, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            self._publish_image(annotated)
        finally:
            if os.path.exists(temp_rgb):
                try:
                    os.remove(temp_rgb)
                except OSError:
                    pass

    def _publish_image(self, img):
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        self.pub.publish(msg)


if __name__ == '__main__':
    viewer = VisionBBoxViewer()
    rospy.spin()

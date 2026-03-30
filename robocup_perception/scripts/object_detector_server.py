#!/usr/bin/env python3
"""
物体检测服务 - 封装Grounded-SAM-2 API
集成open_vocabulary包的Grounding SAM API
"""
import sys
import os
WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OPEN_VOCAB_API = os.path.join(WS_ROOT, 'open_vocabulary', 'API')
if OPEN_VOCAB_API not in sys.path:
    sys.path.append(OPEN_VOCAB_API)

import rospy
import cv2
import numpy as np
import tempfile
from cv_bridge import CvBridge
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, Point, PointStamped

from robocup_msgs.srv import DetectObjects, DetectObjectsResponse
from robocup_msgs.msg import DetectedObject, DetectedObjectArray

# 导入Grounded-SAM-2 API
try:
    from grounding_sam_api import GroundingSAMAPI
    GSAM_AVAILABLE = True
except ImportError:
    rospy.logwarn("未能导入GroundingSAMAPI，将使用模拟模式")
    GSAM_AVAILABLE = False


class ObjectDetectorServer:
    def __init__(self):
        rospy.init_node('object_detector_server')

        # 加载相机内参
        self.fx = rospy.get_param('~camera_intrinsics/fx', 1230.0)
        self.fy = rospy.get_param('~camera_intrinsics/fy', 922.5)
        self.cx = rospy.get_param('~camera_intrinsics/cx', 640.0)
        self.cy = rospy.get_param('~camera_intrinsics/cy', 360.0)
        self.depth_scale = rospy.get_param('~depth_scale', 0.001)
        self.camera_frame_id = rospy.get_param('~camera_frame_id', 'camera_color_optical_frame')

        rospy.loginfo(f"相机内参: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        rospy.loginfo(f"深度缩放: {self.depth_scale}")

        # 初始化Grounded-SAM-2
        if GSAM_AVAILABLE:
            try:
                rospy.loginfo("初始化Grounded-SAM-2 API...")
                self.gsam_api = GroundingSAMAPI(device="cuda")
                rospy.loginfo("✓ GroundingSAM API已初始化")
            except Exception as e:
                rospy.logerr(f"GroundingSAM API初始化失败: {e}")
                self.gsam_api = None
        else:
            self.gsam_api = None

        # TF监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # OpenCV桥接
        self.bridge = CvBridge()

        # 创建服务
        self.service = rospy.Service('/detect_objects', DetectObjects, self.handle_detect)
        rospy.loginfo("✓ 物体检测服务已就绪")

    def depth_to_3d(self, u, v, depth):
        """2D像素坐标+深度 → 3D相机坐标"""
        X = (u - self.cx) * depth / self.fx
        Y = (v - self.cy) * depth / self.fy
        Z = depth
        return X, Y, Z

    def handle_detect(self, req):
        """处理检测请求"""
        rospy.loginfo(f"收到检测请求: {len(req.target_classes)} 个类别")

        if self.gsam_api is None:
            # 模拟模式：返回模拟数据
            rospy.logwarn("Grounded-SAM-2不可用，返回模拟检测结果")
            return self.mock_detection(req)

        try:
            # ROS Image → OpenCV
            rgb_cv = self.bridge.imgmsg_to_cv2(req.rgb_image, "bgr8")
            depth_cv = self.bridge.imgmsg_to_cv2(req.depth_image, "16UC1")

            rospy.loginfo(f"图像尺寸: RGB={rgb_cv.shape}, Depth={depth_cv.shape}")

            # 保存临时文件（避免并发请求覆盖）
            fd_rgb, temp_rgb = tempfile.mkstemp(prefix='detection_rgb_', suffix='.jpg')
            fd_depth, temp_depth = tempfile.mkstemp(prefix='detection_depth_', suffix='.png')
            os.close(fd_rgb)
            os.close(fd_depth)
            cv2.imwrite(temp_rgb, rgb_cv)
            cv2.imwrite(temp_depth, depth_cv)

            # 调用Grounded-SAM-2
            text_prompt = ", ".join(req.target_classes)
            rospy.loginfo(f"检测提示词: {text_prompt}")

            boxes, confidences, labels, masks = self.gsam_api.segment(
                image_path=temp_rgb,
                text_prompt=text_prompt,
                output_dir="/tmp/"
            )

            rospy.loginfo(f"检测到 {len(boxes)} 个候选框")

            # 构建响应
            detected_objects = []
            for i in range(len(boxes)):
                if confidences[i] < req.confidence_threshold:
                    continue

                obj = DetectedObject()
                obj.class_name = labels[i]
                obj.confidence = confidences[i]
                obj.bbox = [int(b) for b in boxes[i]]  # [x1, y1, x2, y2]

                # 计算质心3D坐标
                cx_2d = int((boxes[i][0] + boxes[i][2]) / 2)
                cy_2d = int((boxes[i][1] + boxes[i][3]) / 2)

                # 确保坐标在图像范围内
                cx_2d = max(0, min(cx_2d, depth_cv.shape[1] - 1))
                cy_2d = max(0, min(cy_2d, depth_cv.shape[0] - 1))

                depth_value = depth_cv[cy_2d, cx_2d] * self.depth_scale

                if depth_value > 0:
                    X, Y, Z = self.depth_to_3d(cx_2d, cy_2d, depth_value)
                    obj.centroid = Point(x=X, y=Y, z=Z)

                    # TF 变换：camera_color_optical_frame → base_link
                    try:
                        transform = self.tf_buffer.lookup_transform(
                            'base_link',
                            self.camera_frame_id,
                            rospy.Time(0),
                            rospy.Duration(1.0)
                        )
                        pt_stamped = PointStamped()
                        pt_stamped.header.frame_id = self.camera_frame_id
                        pt_stamped.point = obj.centroid
                        pt_in_base = tf2_geometry_msgs.do_transform_point(pt_stamped, transform)
                        obj.pose.position = pt_in_base.point
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                            tf2_ros.ExtrapolationException) as e:
                        rospy.logwarn(f"TF 变换失败，使用相机坐标: {e}")
                        obj.pose.position = obj.centroid
                    obj.pose.orientation.w = 1.0

                    detected_objects.append(obj)
                    rospy.loginfo(f"  ✓ {obj.class_name} (conf={obj.confidence:.2f}, "
                                  f"base_link=[{obj.pose.position.x:.2f}, "
                                  f"{obj.pose.position.y:.2f}, {obj.pose.position.z:.2f}])")

            # 返回结果
            resp = DetectObjectsResponse()
            resp.detected_objects.objects = detected_objects
            resp.success = len(detected_objects) > 0
            resp.message = f"检测到 {len(detected_objects)} 个物体"

            rospy.loginfo(f"检测完成: {resp.message}")
            return resp

        except Exception as e:
            rospy.logerr(f"检测过程出错: {e}")
            resp = DetectObjectsResponse()
            resp.success = False
            resp.message = f"检测失败: {str(e)}"
            return resp
        finally:
            for tmp_file in ('temp_rgb', 'temp_depth'):
                path = locals().get(tmp_file, None)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def mock_detection(self, req):
        """模拟检测结果（用于测试）"""
        resp = DetectObjectsResponse()

        # 创建几个模拟物体
        mock_objects = [
            ("cup", 0.85, [100, 100, 200, 200], [0.3, 0.0, 0.5]),
            ("plate", 0.90, [250, 150, 350, 250], [0.4, 0.1, 0.5]),
            ("spoon", 0.75, [400, 200, 450, 280], [0.2, -0.1, 0.5])
        ]

        detected_objects = []
        for name, conf, bbox, pos in mock_objects:
            if name in req.target_classes and conf >= req.confidence_threshold:
                obj = DetectedObject()
                obj.class_name = name
                obj.confidence = conf
                obj.bbox = bbox
                obj.centroid = Point(x=pos[0], y=pos[1], z=pos[2])
                obj.pose.position = obj.centroid
                obj.pose.orientation.w = 1.0
                detected_objects.append(obj)

        resp.detected_objects.objects = detected_objects
        resp.success = len(detected_objects) > 0
        resp.message = f"模拟检测到 {len(detected_objects)} 个物体"

        return resp


if __name__ == '__main__':
    try:
        server = ObjectDetectorServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("物体检测服务已停止")

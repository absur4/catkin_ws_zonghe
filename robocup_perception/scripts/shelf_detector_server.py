#!/usr/bin/env python3
"""
柜子层板检测服务
检测柜子的多层结构，返回每层的位置
"""
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from robocup_msgs.srv import DetectShelf, DetectShelfResponse
from robocup_msgs.msg import ShelfDetectionResult, ShelfLayer


class ShelfDetectorServer:
    def __init__(self):
        rospy.init_node('shelf_detector_server')

        # 加载参数
        self.num_shelves_hint = rospy.get_param('~num_shelves_hint', 4)
        self.min_layer_height = rospy.get_param('~min_layer_height', 0.15)
        self.max_layer_height = rospy.get_param('~max_layer_height', 0.50)

        # OpenCV桥接
        self.bridge = CvBridge()

        # 订阅深度图像（用于检测）
        self.depth_image = None
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)

        # 创建服务
        self.service = rospy.Service('/detect_shelf', DetectShelf, self.handle_detect)
        rospy.loginfo("✓ 柜子检测服务已启动")

    def depth_callback(self, msg):
        """深度图像回调"""
        self.depth_image = msg

    def detect_shelves_simple(self):
        """简单的柜子检测（均匀分层）"""
        result = ShelfDetectionResult()
        result.num_layers = self.num_shelves_hint
        result.cabinet_pose.position.x = 0.6
        result.cabinet_pose.position.y = 0.0
        result.cabinet_pose.position.z = 0.75
        result.cabinet_pose.orientation.w = 1.0
        result.cabinet_bbox = []

        # 假设柜子在机械臂前方，创建均匀分布的层
        base_x = 0.6  # 柜子距离机器人60cm
        base_y = 0.0
        base_z = 0.3  # 最底层高度30cm

        for i in range(self.num_shelves_hint):
            layer = ShelfLayer()
            layer.layer = i + 1
            layer.y_top = 0
            layer.y_bottom = 0
            layer.width = 0.5
            layer.depth = 0.35
            layer.height = 0.30
            layer.center_pose.position.x = base_x
            layer.center_pose.position.y = base_y
            layer.center_pose.position.z = base_z + i * layer.height
            layer.center_pose.orientation.w = 1.0
            result.layers.append(layer)

        return result

    def detect_shelves_from_depth(self, depth_image):
        """从深度图像检测柜子层板（简化版）"""
        # TODO: 实现基于深度图像的层板检测
        # 1. 将深度图转换为点云
        # 2. 检测水平平面
        # 3. 聚类为不同的层
        # 4. 计算每层的3D位置

        # 当前使用简单版本
        return self.detect_shelves_simple()

    def handle_detect(self, req):
        """处理柜子检测请求"""
        rospy.loginfo("检测柜子层板结构...")

        resp = DetectShelfResponse()

        try:
            depth_image = req.depth_image if req.depth_image.data else self.depth_image

            if depth_image is not None:
                # 使用深度图像检测
                resp.shelf_result = self.detect_shelves_from_depth(depth_image)
            else:
                # 使用简单检测
                rospy.logwarn("深度图像不可用，使用默认层板配置")
                resp.shelf_result = self.detect_shelves_simple()

            resp.success = True
            resp.message = f"检测到 {resp.shelf_result.num_layers} 层"

            rospy.loginfo(f"✓ 检测到 {resp.shelf_result.num_layers} 层柜子")
            for layer in resp.shelf_result.layers:
                rospy.loginfo(f"  第{layer.layer}层: z={layer.center_pose.position.z:.2f}m")

        except Exception as e:
            rospy.logerr(f"柜子检测失败: {e}")
            resp.success = False
            resp.message = str(e)

            # 返回默认配置
            resp.shelf_result = self.detect_shelves_simple()

        return resp


if __name__ == '__main__':
    try:
        server = ShelfDetectorServer()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("柜子检测服务已停止")

#!/usr/bin/env python3
"""
AssessScene State - 场景评估状态
订阅相机话题，调用物体检测服务
"""
import rospy
import smach
from sensor_msgs.msg import Image
from robocup_msgs.srv import DetectObjects, DetectObjectsRequest


class AssessScene(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['objects_detected', 'no_objects', 'perception_failed'],
            output_keys=['detected_objects', 'objects_to_pick']
        )

        # 从参数服务器读取检测配置
        self.target_classes = rospy.get_param('~detection/target_classes', [
            'cup', 'plate', 'spoon', 'fork', 'knife', 'bowl',
            'apple', 'banana', 'bread', 'bottle',
            'wrapper', 'tissue', 'napkin'
        ])
        self.confidence_threshold = rospy.get_param('~detection/confidence_threshold', 0.3)
        self.communicate_perception = rospy.get_param('~communicate_perception', True)

        # 订阅相机话题
        self.rgb_image = None
        self.depth_image = None
        self.rgb_sub = rospy.Subscriber(
            '/camera/color/image_raw',
            Image,
            self.rgb_callback
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/aligned_depth_to_color/image_raw',
            Image,
            self.depth_callback
        )

        # 创建检测服务客户端
        rospy.loginfo("AssessScene: 等待检测服务...")
        # Note: 实际运行时才等待服务

    def rgb_callback(self, msg):
        self.rgb_image = msg

    def depth_callback(self, msg):
        self.depth_image = msg

    def execute(self, userdata):
        rospy.loginfo("========== 评估场景 ==========")

        # 等待图像数据
        rospy.loginfo("等待相机图像...")
        rate = rospy.Rate(10)
        timeout = rospy.Time.now() + rospy.Duration(5.0)

        while (self.rgb_image is None or self.depth_image is None) and rospy.Time.now() < timeout:
            rate.sleep()

        if self.rgb_image is None or self.depth_image is None:
            rospy.logerr("未能获取相机图像")
            return 'perception_failed'

        rospy.loginfo("相机图像已就绪")

        # 等待检测服务
        try:
            rospy.wait_for_service('/detect_objects', timeout=10.0)
        except rospy.ROSException:
            rospy.logerr("检测服务不可用")
            return 'perception_failed'

        # 调用物体检测服务
        try:
            detect_srv = rospy.ServiceProxy('/detect_objects', DetectObjects)

            req = DetectObjectsRequest()
            req.target_classes = self.target_classes
            req.rgb_image = self.rgb_image
            req.depth_image = self.depth_image
            req.confidence_threshold = self.confidence_threshold

            rospy.loginfo(f"调用检测服务，目标类别: {len(req.target_classes)} 个")
            resp = detect_srv(req)

            if resp.success and len(resp.detected_objects.objects) > 0:
                rospy.loginfo(f"✓ 检测到 {len(resp.detected_objects.objects)} 个物体:")
                for obj in resp.detected_objects.objects:
                    rospy.loginfo(f"  - {obj.class_name} (置信度: {obj.confidence:.2f})")
                    # 规则书 Rule #11：感知公示（Communicating Perception）
                    if self.communicate_perception:
                        rospy.loginfo(f"[感知公示] 发现 {obj.class_name} 置信度 {obj.confidence:.0%}")

                userdata.detected_objects = resp.detected_objects.objects
                userdata.objects_to_pick = list(resp.detected_objects.objects)  # 复制列表
                return 'objects_detected'
            else:
                rospy.logwarn("未检测到物体")
                return 'no_objects'

        except rospy.ServiceException as e:
            rospy.logerr(f"检测服务调用失败: {e}")
            return 'perception_failed'

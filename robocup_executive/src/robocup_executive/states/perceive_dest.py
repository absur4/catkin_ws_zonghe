#!/usr/bin/env python3
"""
PerceiveDest State - 感知目的地状态
调用柜子检测服务，计算放置姿态
"""
import rospy
import smach
from sensor_msgs.msg import Image
from robocup_msgs.srv import DetectShelf, ComputePlacePose, DetectObjects
from robocup_msgs.srv import DetectShelfRequest, ComputePlacePoseRequest, DetectObjectsRequest


class PerceiveDest(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['perception_done', 'perception_failed', 'fatal_error'],
            input_keys=['destination', 'selected_object'],
            output_keys=['shelf_info', 'place_pose', 'target_layer']
        )

        # 订阅相机（复用assess_scene的订阅）
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

    def rgb_callback(self, msg):
        self.rgb_image = msg

    def depth_callback(self, msg):
        self.depth_image = msg

    def execute(self, userdata):
        rospy.loginfo("========== 感知目的地 ==========")

        destination = userdata.get('destination', '')
        rospy.loginfo(f"目的地: {destination}")

        if 'selected_object' not in userdata or userdata['selected_object'] is None:
            rospy.logerr("未提供 selected_object，无法计算放置姿态")
            return 'fatal_error'

        # 如果是柜子，需要检测层板
        if destination == 'cabinet':
            rospy.loginfo("检测柜子层板...")

            # 等待图像数据
            rate = rospy.Rate(10)
            timeout = rospy.Time.now() + rospy.Duration(3.0)

            while (self.rgb_image is None or self.depth_image is None) and rospy.Time.now() < timeout:
                rate.sleep()

            if self.rgb_image is None or self.depth_image is None:
                rospy.logwarn("未能获取相机图像，使用默认放置位置")
                return 'perception_failed'

            try:
                rospy.wait_for_service('/detect_shelf', timeout=5.0)
                detect_shelf_srv = rospy.ServiceProxy('/detect_shelf', DetectShelf)

                req = DetectShelfRequest()
                req.rgb_image = self.rgb_image
                req.depth_image = self.depth_image
                req.cabinet_prompt = "cabinet shelf"

                resp = detect_shelf_srv(req)

                if resp.success:
                    rospy.loginfo(f"✓ 检测到柜子，共 {len(resp.shelf_result.layers)} 层")
                    userdata['shelf_info'] = resp.shelf_result

                    # 选择放置层（默认第2层）
                    target_layer = 1 if len(resp.shelf_result.layers) > 1 else 0
                    userdata['target_layer'] = target_layer
                else:
                    rospy.logwarn("柜子检测失败")
                    return 'perception_failed'

            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr(f"柜子检测服务调用失败: {e}")
                return 'perception_failed'

        # 计算放置姿态
        try:
            rospy.wait_for_service('/compute_place_pose', timeout=5.0)
            compute_place_srv = rospy.ServiceProxy('/compute_place_pose', ComputePlacePose)

            req = ComputePlacePoseRequest()

            # 取出目标层 ShelfLayer 对象（规则书要求按相似性分组）
            if 'shelf_info' in userdata and userdata['shelf_info']:
                shelf = userdata['shelf_info']
                layer_idx = userdata.get('target_layer', 0)
                if layer_idx < len(shelf.layers):
                    req.target_layer = shelf.layers[layer_idx]

            # 检测柜子当前已有物品，用于相似性分组（规则书 §5.2 cabinet placement）
            try:
                rospy.wait_for_service('/detect_objects', timeout=5.0)
                detect_srv = rospy.ServiceProxy('/detect_objects', DetectObjects)
                det_req = DetectObjectsRequest()
                det_req.rgb_image = self.rgb_image
                det_req.depth_image = self.depth_image
                det_req.target_classes = ['cup', 'plate', 'bowl', 'spoon', 'fork',
                                          'knife', 'mug', 'apple', 'banana', 'bread']
                det_req.confidence_threshold = 0.3
                det_resp = detect_srv(det_req)
                if det_resp.success:
                    req.existing_objects = det_resp.detected_objects
                    rospy.loginfo(f"柜子现有物品: {len(det_resp.detected_objects.objects)} 件")
            except Exception as e:
                rospy.logwarn(f"柜子物品检测跳过: {e}")

            # 从 bbox 估算待放置物品尺寸
            obj = userdata['selected_object']
            if len(obj.bbox) >= 4:
                pixel_width = obj.bbox[2] - obj.bbox[0]
                pixel_height = obj.bbox[3] - obj.bbox[1]
                depth_m = obj.centroid.z if obj.centroid.z > 0 else 0.5
                # 使用 depth 和典型相机内参估算真实尺寸
                fx = rospy.get_param('/object_detector_server/camera_intrinsics/fx', 1230.0)
                req.object_width = float(pixel_width) * depth_m / fx
                req.object_depth = float(pixel_height) * depth_m / fx
            else:
                req.object_width = 0.08  # 默认 8cm
                req.object_depth = 0.08

            resp = compute_place_srv(req)

            if resp.success:
                rospy.loginfo("✓ 计算放置姿态成功")
                userdata['place_pose'] = resp.place_pose
                return 'perception_done'
            else:
                rospy.logwarn("放置姿态计算失败")
                return 'perception_failed'

        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr(f"放置姿态计算服务调用失败: {e}")
            return 'perception_failed'
        except Exception as e:
            rospy.logerr(f"感知目的地阶段出现异常: {e}")
            return 'perception_failed'

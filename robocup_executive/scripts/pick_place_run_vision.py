#!/usr/bin/env python3

import os  # 导入模块
import sys  # 导入模块
import tempfile  # 导入模块

import rospy  # 导入模块
import smach  # 导入模块
import smach_ros  # 导入模块
import actionlib  # 导入模块
import cv2  # 导入模块
import numpy as np  # 导入模块
import tf2_ros  # 导入模块
from cv_bridge import CvBridge  # 导入模块
from geometry_msgs.msg import Pose, Point, PointStamped  # 导入模块
from sensor_msgs.msg import Image  # 导入模块
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal  # 导入模块


# ============================================================
# 可修改区域：模拟检测到的桌面物品（名称, 置信度, x, y, z）
# x: 机器人正前方距离（m），y: 左正右负，z: 桌面高度（m）
# 坐标在机器人到达 kitchen 位置后的 base_link 帧下定义。
# ============================================================
MOCK_TABLE_OBJECTS = [  # 设置 MOCK_TABLE_OBJECTS
    ('cup',   0.85, 0.55,  0.10, 0.30),  # 执行语句
    ('plate', 0.72, 0.60, -0.05, 0.28),  # 执行语句
    ('apple', 0.65, 0.58,  0.20, 0.32),  # 执行语句
]  # 执行语句
FRUIT_KEYWORDS = ['apple', 'banana', 'orange']  # 设置 FRUIT_KEYWORDS

# 用于本地分类的关键词列表（与 object_classifier.py 保持一致）
CLEANABLE_KEYWORDS = ['cup', 'mug', 'plate', 'dish', 'bowl', 'spoon', 'fork', 'knife']  # 设置 CLEANABLE_KEYWORDS

# ============================================================
# 导航配置（取消中间点）
# ============================================================
CURRENT_LOCATION = None  # 设置 CURRENT_LOCATION


# 函数说明: _set_current_location 的用途说明
def _set_current_location(name):  # 定义函数_set_current_location
    global CURRENT_LOCATION  # 执行语句
    CURRENT_LOCATION = name  # 设置 CURRENT_LOCATION


# 函数说明: _plan_route 的用途说明
def _plan_route(current, target):  # 定义函数_plan_route
    return [target]  # 返回结果


# 函数说明: _send_nav_goal 的用途说明
def _send_nav_goal(nav_client, location_name, log_prefix):  # 定义函数_send_nav_goal
    if not nav_client.wait_for_server(timeout=rospy.Duration(10.0)):  # 条件判断
        rospy.logerr(f"{log_prefix} 导航动作服务器未响应")  # 输出错误日志
        return False  # 返回结果

    goal = NavigateToLocationGoal()  # 设置 goal
    goal.target_location = location_name  # 设置 goal.target_location

    rospy.loginfo(f"{log_prefix} 发送导航目标: {location_name}")  # 输出信息日志
    nav_client.send_goal(goal)  # 执行语句

    finished = nav_client.wait_for_result(timeout=rospy.Duration(120.0))  # 设置 finished
    if not finished:  # 条件判断
        nav_client.cancel_goal()  # 执行语句
        rospy.logerr(f"{log_prefix} 导航到 {location_name} 超时")  # 输出错误日志
        return False  # 返回结果

    result = nav_client.get_result()  # 设置 result
    if result and result.success:  # 条件判断
        _set_current_location(location_name)  # 执行语句
        rospy.loginfo(f"{log_prefix} ✓ 成功到达 {location_name}")  # 输出信息日志
        return True  # 返回结果

    rospy.logwarn(f"{log_prefix} ✗ 导航到 {location_name} 失败")  # 输出警告日志
    return False  # 返回结果


# 函数说明: _navigate_with_midpoints 的用途说明
def _navigate_with_midpoints(nav_client, target_name, log_prefix):  # 定义函数_navigate_with_midpoints
    route = _plan_route(CURRENT_LOCATION, target_name)  # 设置 route
    if len(route) > 1:  # 条件判断
        rospy.loginfo(f"{log_prefix} 规划中间点路径: {route}")  # 输出信息日志
    for waypoint in route:  # 循环遍历
        if not _send_nav_goal(nav_client, waypoint, log_prefix):  # 条件判断
            return False  # 返回结果
    return True  # 返回结果


# ============================================================
# Open Vocabulary API 导入
# ============================================================
WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # 设置 WS_ROOT
OPEN_VOCAB_API = os.path.join(WS_ROOT, 'open_vocabulary', 'open_vocabulary', 'API')  # 设置 OPEN_VOCAB_API
DINO_CONFIG_PATH = os.path.join(  # 设置 DINO_CONFIG_PATH
    WS_ROOT,  # 执行语句
    'open_vocabulary',  # 执行语句
    'open_vocabulary',  # 执行语句
    'Grounded-SAM-2',  # 执行语句
    'grounding_dino',  # 执行语句
    'groundingdino',  # 执行语句
    'config',  # 执行语句
    'GroundingDINO_SwinT_OGC_3_15_new.py'  # 执行语句
)  # 执行语句
if OPEN_VOCAB_API not in sys.path:  # 条件判断
    sys.path.append(OPEN_VOCAB_API)  # 执行语句

try:  # 开始异常处理
    from grounding_sam_api import GroundingSAMAPI  # 导入模块
    GSAM_AVAILABLE = True  # 设置 GSAM_AVAILABLE
except Exception as e:  # 异常分支
    rospy.logwarn(f"未能导入 GroundingSAMAPI，将使用模拟模式: {e}")  # 输出警告日志
    GSAM_AVAILABLE = False  # 设置 GSAM_AVAILABLE


# ============================================================
# 模拟 DetectedObject 消息（与 dry run 保持一致）
# ============================================================
# 类说明: MockDetectedObject 类的功能说明
class MockDetectedObject:  # 定义类MockDetectedObject的结构
    """模拟 robocup_msgs/DetectedObject，供 dry run 使用"""
    # 函数说明: __init__ 的用途说明
    def __init__(self, class_name, confidence=0.80, x=0.5, y=0.0, z=0.3):  # 定义函数__init__
        self.class_name = class_name  # 设置 self.class_name
        self.confidence = confidence  # 设置 self.confidence
        self.bbox = []  # 设置 self.bbox
        # centroid 即物品在 base_link 坐标系下的三维位置
        self.centroid = Point(x=x, y=y, z=z)  # 设置 self.centroid
        self.pose = Pose()  # 设置 self.pose
        self.pose.position.x = x  # 设置 self.pose.position.x
        self.pose.position.y = y  # 设置 self.pose.position.y
        self.pose.position.z = z  # 设置 self.pose.position.z
        self.pose.orientation.w = 1.0  # 设置 self.pose.orientation.w
        self.category = ''  # 设置 self.category


# ============================================================
# 视觉感知上下文：订阅相机 + 调 Grounding SAM
# ============================================================
# 类说明: VisionContext 类的功能说明
class VisionContext:  # 定义类VisionContext的结构
    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        # 相机内参
        self.fx = rospy.get_param('~camera_intrinsics/fx', 1230.0)  # 设置 self.fx
        self.fy = rospy.get_param('~camera_intrinsics/fy', 922.5)  # 设置 self.fy
        self.cx = rospy.get_param('~camera_intrinsics/cx', 640.0)  # 设置 self.cx
        self.cy = rospy.get_param('~camera_intrinsics/cy', 360.0)  # 设置 self.cy
        self.depth_scale = rospy.get_param('~depth_scale', 0.001)  # 设置 self.depth_scale
        self.camera_frame_id = rospy.get_param('~camera_frame_id', 'camera_color_optical_frame')  # 设置 self.camera_frame_id
        self.rgb_topic = rospy.get_param('~rgb_topic', '/camera/color/image_raw')  # 设置 self.rgb_topic
        self.depth_topic = rospy.get_param('~depth_topic', '/camera/depth/image_rect_raw')  # 设置 self.depth_topic
        self.require_depth = rospy.get_param('~require_depth', False)  #q1:如果要测量深度的话，这个是不是要为true
        self.default_object_z = rospy.get_param('~default_object_z', 0.30) #q2:这个是做什么的
        self.annotated_topic = rospy.get_param('~annotated_topic', '/vision/annotated_image_3_15_new')  # 设置 self.annotated_topic
        self.raw_preview_enabled = rospy.get_param('~raw_preview/enabled', True)  # 设置 self.raw_preview_enabled
        self.raw_preview_topic = rospy.get_param('~raw_preview/topic', '/vision/raw_image')  # 设置 self.raw_preview_topic
        self.raw_preview_rate_hz = rospy.get_param('~raw_preview/rate_hz', 10.0)  # 设置 self.raw_preview_rate_hz
        self.save_annotated = rospy.get_param('~save_annotated/enabled', True)  # 设置 self.save_annotated
        self.save_annotated_dir = rospy.get_param('~save_annotated/dir', '/tmp/vision_annotated') #q3:这个是不是可以更改保存识别后的图片的保存位置？

        self.rgb_image = None  # 设置 self.rgb_image
        self.depth_image = None #上面这两个为true会怎么样
        self.rgb_sub = rospy.Subscriber(self.rgb_topic, Image, self._rgb_callback)  # 设置 self.rgb_sub
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_callback)  # 设置 self.depth_sub

        self.bridge = CvBridge() #这是做什么的？
        self.annotated_pub = rospy.Publisher(self.annotated_topic, Image, queue_size=1)  # 设置 self.annotated_pub
        self.raw_preview_pub = None  # 设置 self.raw_preview_pub
        self._last_raw_pub = rospy.Time(0)  # 设置 self._last_raw_pub
        if self.raw_preview_enabled:  # 条件判断
            self.raw_preview_pub = rospy.Publisher(self.raw_preview_topic, Image, queue_size=1)  # 设置 self.raw_preview_pub
        # tf2_geometry_msgs 依赖 PyKDL，在 conda 环境中可能缺失
        try:  # 开始异常处理
            import tf2_geometry_msgs  # noqa: F401
            self.tf2_available = True  # 设置 self.tf2_available
        except Exception as e:  # 异常分支
            rospy.logwarn(f"tf2_geometry_msgs 不可用，将跳过 TF 变换: {e}")  # 输出警告日志
            self.tf2_available = False  # 设置 self.tf2_available

        self.tf_buffer = tf2_ros.Buffer()  # 设置 self.tf_buffer
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)  # 设置 self.tf_listener

        self.gsam_api = None  # 设置 self.gsam_api
        if GSAM_AVAILABLE:  # 条件判断
            try:  # 开始异常处理
                rospy.loginfo("初始化 GroundingSAM API...")  # 输出信息日志
                if not os.path.exists(DINO_CONFIG_PATH):  # 条件判断
                    rospy.logwarn(f"GroundingDINO config 不存在: {DINO_CONFIG_PATH}")  # 输出警告日志
                self.gsam_api = GroundingSAMAPI(  # 设置 self.gsam_api
                    device="cuda",  # 设置 device
                    grounding_dino_config=DINO_CONFIG_PATH if os.path.exists(DINO_CONFIG_PATH) else None  # 设置 grounding_dino_config
                )  # 执行语句
                rospy.loginfo("✓ GroundingSAM API 已初始化")  # 输出信息日志
            except Exception as e:  # 异常分支
                rospy.logerr(f"GroundingSAM API 初始化失败: {e}")  # 输出错误日志
                self.gsam_api = None  # 设置 self.gsam_api

    # 函数说明: _rgb_callback 的用途说明
    def _rgb_callback(self, msg):  # 定义函数_rgb_callback
        self.rgb_image = msg  # 设置 self.rgb_image
        if self.raw_preview_pub is None:  # 条件判断
            return  # 执行语句
        now = rospy.Time.now()  # 设置 now
        if self.raw_preview_rate_hz > 0:  # 条件判断
            min_dt = 1.0 / float(self.raw_preview_rate_hz)  # 设置 min_dt
            if (now - self._last_raw_pub).to_sec() < min_dt:  # 条件判断
                return  # 执行语句
        self.raw_preview_pub.publish(msg)  # 执行语句
        self._last_raw_pub = now  # 设置 self._last_raw_pub

    # 函数说明: _depth_callback 的用途说明
    def _depth_callback(self, msg):  # 定义函数_depth_callback
        self.depth_image = msg  # 设置 self.depth_image

    # 函数说明: wait_for_images 的用途说明
    def wait_for_images(self, timeout_sec=5.0):  # 定义函数wait_for_images
        rate = rospy.Rate(10)  # 设置 rate
        timeout = rospy.Time.now() + rospy.Duration(timeout_sec)  # 设置 timeout
        while (self.rgb_image is None or (self.require_depth and self.depth_image is None)) and rospy.Time.now() < timeout:  # 循环等待
            rate.sleep()  # 执行语句
        if self.rgb_image is None:  # 条件判断
            return False  # 返回结果
        if self.require_depth and self.depth_image is None:  # 条件判断
            return False  # 返回结果
        return True  # 返回结果

    # 函数说明: depth_to_3d 的用途说明
    def depth_to_3d(self, u, v, depth):  # 定义函数depth_to_3d
        """2D像素坐标+深度 → 3D相机坐标"""
        X = (u - self.cx) * depth / self.fx  # 设置 X
        Y = (v - self.cy) * depth / self.fy  # 设置 Y
        Z = depth  # 设置 Z
        return X, Y, Z  # 返回结果

    # 函数说明: _build_prompt 的用途说明
    def _build_prompt(self, classes):  # 定义函数_build_prompt
        prompt = ". ".join([c.strip().lower() for c in classes if c.strip()])  # 设置 prompt
        if not prompt.endswith('.'):  # 条件判断
            prompt += '.'  # 执行语句
        return prompt  # 返回结果

    # 函数说明: detect_objects 的用途说明
    def detect_objects(self, target_classes, confidence_threshold=0.3):  # 定义函数detect_objects
        if self.gsam_api is None:  # 条件判断
            rospy.logwarn("GroundingSAM 不可用，回退到模拟检测")  # 输出警告日志
            return [MockDetectedObject(*args) for args in MOCK_TABLE_OBJECTS]  # 返回结果

        if not self.wait_for_images():  # 条件判断
            rospy.logerr("未能获取相机图像")  # 输出错误日志
            return []  # 返回结果

        rgb_cv = self.bridge.imgmsg_to_cv2(self.rgb_image, "bgr8")  # 设置 rgb_cv
        depth_cv = self.bridge.imgmsg_to_cv2(self.depth_image, "16UC1")  # 设置 depth_cv

        # 保存临时文件
        fd_rgb, temp_rgb = tempfile.mkstemp(prefix='vision_rgb_', suffix='.jpg')  # 设置 fd_rgb, temp_rgb
        os.close(fd_rgb)  # 执行语句
        cv2.imwrite(temp_rgb, rgb_cv)  # 执行语句

        try:  # 开始异常处理
            text_prompt = self._build_prompt(target_classes)  # 设置 text_prompt
            rospy.loginfo(f"检测提示词: {text_prompt}")  # 输出信息日志

            results = self.gsam_api.segment(  # 设置 results
                image_path=temp_rgb,  # 设置 image_path
                text_prompt=text_prompt,  # 设置 text_prompt
                output_dir=None,  # 设置 output_dir
                save_mask=False,  # 设置 save_mask
                save_annotated=False,  # 设置 save_annotated
                save_json=False  # 设置 save_json
            )  # 执行语句

            boxes = results.get("boxes", [])  # 设置 boxes
            confidences = results.get("confidences", [])  # 设置 confidences
            labels = results.get("labels", [])  # 设置 labels

            detected_objects = []  # 设置 detected_objects
            annotated = rgb_cv.copy()  # 设置 annotated
            for i in range(len(boxes)):  # 循环遍历
                if confidences[i] < confidence_threshold:  # 条件判断
                    continue  # 继续下一轮循环

                x1, y1, x2, y2 = boxes[i]  # 设置 x1, y1, x2, y2
                cx_2d = int((x1 + x2) / 2)  # 设置 cx_2d
                cy_2d = int((y1 + y2) / 2)  # 设置 cy_2d

                cx_2d = max(0, min(cx_2d, depth_cv.shape[1] - 1))  # 设置 cx_2d
                cy_2d = max(0, min(cy_2d, depth_cv.shape[0] - 1))  # 设置 cy_2d

                depth_value = None  # 设置 depth_value
                if depth_cv is not None:  # 条件判断
                    depth_value = depth_cv[cy_2d, cx_2d] * self.depth_scale  # 设置 depth_value
                if depth_value is None or depth_value <= 0:  # 条件判断
                    y0 = max(0, cy_2d - 2)  # 设置 y0
                    y1 = min(depth_cv.shape[0], cy_2d + 3)  # 设置 y1
                    x0 = max(0, cx_2d - 2)  # 设置 x0
                    x1 = min(depth_cv.shape[1], cx_2d + 3)  # 设置 x1
                    neighborhood = depth_cv[y0:y1, x0:x1]  # 设置 neighborhood
                    nonzero = neighborhood[neighborhood > 0]  # 设置 nonzero
                    if nonzero.size == 0:  # 条件判断
                        depth_value = None  # 设置 depth_value
                    else:  # 条件分支
                        depth_value = float(np.median(nonzero)) * self.depth_scale  # 设置 depth_value

                if depth_value is None:  # 条件判断
                    X, Y, Z = 0.5, 0.0, self.default_object_z  # 设置 X, Y, Z
                else:  # 条件分支
                    X, Y, Z = self.depth_to_3d(cx_2d, cy_2d, depth_value)  # 设置 X, Y, Z
                label_raw = str(labels[i]).lower().strip()  # 设置 label_raw
                label = label_raw.replace('.', '').strip()  # 设置 label
                obj = MockDetectedObject(label, confidences[i], X, Y, Z)  # 设置 obj
                obj.bbox = [int(x1), int(y1), int(x2), int(y2)]  # 设置 obj.bbox

                # TF 变换：camera_color_optical_frame → base_link
                if self.tf2_available:  # 条件判断
                    try:  # 开始异常处理
                        transform = self.tf_buffer.lookup_transform(  # 设置 transform
                            'base_link',  # 执行语句
                            self.camera_frame_id,  # 执行语句
                            rospy.Time(0),  # 执行语句
                            rospy.Duration(1.0)  # 执行语句
                        )  # 执行语句
                        pt_stamped = PointStamped()  # 设置 pt_stamped
                        pt_stamped.header.frame_id = self.camera_frame_id  # 设置 pt_stamped.header.frame_id
                        pt_stamped.point = obj.centroid  # 设置 pt_stamped.point
                        import tf2_geometry_msgs  # 导入模块
                        pt_in_base = tf2_geometry_msgs.do_transform_point(pt_stamped, transform)  # 设置 pt_in_base
                        obj.pose.position = pt_in_base.point  # 设置 obj.pose.position
                    except (tf2_ros.LookupException, tf2_ros.ConnectivityException,  # 异常分支
                            tf2_ros.ExtrapolationException) as e:  # 进入代码块
                        rospy.logwarn(f"TF 变换失败，使用相机坐标: {e}")  # 输出警告日志
                        obj.pose.position = obj.centroid  # 设置 obj.pose.position
                else:  # 条件分支
                    obj.pose.position = obj.centroid  # 设置 obj.pose.position
                obj.pose.orientation.w = 1.0  # 设置 obj.pose.orientation.w

                detected_objects.append(obj)  # 执行语句

                # 画框与标签
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # 执行语句
                label_text = f"{label} {confidences[i]:.2f}"  # 设置 label_text
                cv2.putText(  # 执行语句
                    annotated,  # 执行语句
                    label_text,  # 执行语句
                    (int(x1), max(0, int(y1) - 5)),  # 执行语句
                    cv2.FONT_HERSHEY_SIMPLEX,  # 执行语句
                    0.5,  # 执行语句
                    (0, 255, 0),  # 执行语句
                    1,  # 执行语句
                    cv2.LINE_AA  # 执行语句
                )  # 执行语句

            # 发布带框图像
            try:  # 开始异常处理
                self.annotated_pub.publish(self.bridge.cv2_to_imgmsg(annotated, "bgr8"))  # 执行语句
            except Exception:  # 异常分支
                pass  # 占位

            # 保存带框图像
            if self.save_annotated:  # 条件判断
                try:  # 开始异常处理
                    os.makedirs(self.save_annotated_dir, exist_ok=True)  # 执行语句
                    stamp = rospy.Time.now().to_sec()  # 设置 stamp
                    out_path = os.path.join(self.save_annotated_dir, f"annotated_{stamp:.3f}.jpg")  # 设置 out_path
                    cv2.imwrite(out_path, annotated)  # 执行语句
                except Exception:  # 异常分支
                    pass  # 占位

            return detected_objects  # 返回结果
        finally:  # 异常收尾
            if os.path.exists(temp_rgb):  # 条件判断
                try:  # 开始异常处理
                    os.remove(temp_rgb)  # 执行语句
                except OSError:  # 异常分支
                    pass  # 占位

    # 函数说明: detect_single 的用途说明
    def detect_single(self, item_name, confidence_threshold=0.3):  # 定义函数detect_single
        results = self.detect_objects([item_name], confidence_threshold=confidence_threshold)  # 设置 results
        if not results:  # 条件判断
            return None  # 返回结果
        # 选择置信度最高的
        results.sort(key=lambda o: o.confidence, reverse=True)  # 执行语句
        return results[0]  # 返回结果


# ============================================================
# 可修改区域：早餐物品的模拟位置（按物品名索引）
# 坐标在机器人到达对应 source 位置后的 base_link 帧下定义。
# ============================================================
MOCK_BREAKFAST_OBJECTS = {  # 设置 MOCK_BREAKFAST_OBJECTS
    'bowl':   MockDetectedObject('bowl',   0.88, 0.50,  0.00, 0.30),  # 执行语句
    'spoon':  MockDetectedObject('spoon',  0.82, 0.52,  0.08, 0.30),  # 执行语句
    'cereal': MockDetectedObject('cereal', 0.76, 0.50, -0.05, 0.35),  # 执行语句
    'milk':   MockDetectedObject('milk',   0.79, 0.52,  0.05, 0.38),  # 执行语句
}  # 执行语句


# ============================================================
# 状态类定义
# ============================================================

# 类说明: InitSystem 类的功能说明
class InitSystem(smach.State):  # 定义类InitSystem的结构
    """[DRY RUN] 跳过所有服务检查，直接返回 initialized"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['initialized', 'init_failed']  # 设置 outcomes
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== [DRY RUN] 初始化系统 ==========")  # 输出信息日志
        services_to_check = [  # 设置 services_to_check
            '/detect_objects',  # 执行语句
            '/classify_object',  # 执行语句
            '/compute_grasp_pose',  # 执行语句
            '/compute_place_pose'  # 执行语句
        ]  # 执行语句
        for s in services_to_check:  # 循环遍历
            rospy.loginfo(f"[DRY RUN] 模拟检查服务 {s} ... OK")  # 输出信息日志
        for a in ['/pick_object', '/place_object', '/navigate_to_location']:  # 循环遍历
            rospy.loginfo(f"[DRY RUN] 模拟检查动作服务器 {a} ... OK")  # 输出信息日志
        rospy.loginfo("[DRY RUN] 系统初始化完成（模拟）")  # 输出信息日志
        return 'initialized'  # 返回结果


# 类说明: NavigateToKitchen 类的功能说明
class NavigateToKitchen(smach.State):  # 定义类NavigateToKitchen的结构
    """【真实导航】完全复制自 navigate_to_kitchen.py"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['arrived', 'navigation_failed']  # 设置 outcomes
        )  # 执行语句
        self.nav_client = actionlib.SimpleActionClient(  # 设置 self.nav_client
            '/navigate_to_location',  # 执行语句
            NavigateToLocationAction  # 执行语句
        )  # 执行语句
        rospy.loginfo("NavigateToKitchen: 等待导航动作服务器...")  # 输出信息日志

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== 导航到厨房 ==========")  # 输出信息日志
        if _navigate_with_midpoints(self.nav_client, "kitchen", "[NavigateToKitchen]"):  # 条件判断
            return 'arrived'  # 返回结果
        return 'navigation_failed'  # 返回结果


# 类说明: AssessScene 类的功能说明
class AssessScene(smach.State):  # 定义类AssessScene的结构
    """使用 GroundingSAM 进行真实检测（失败则回退模拟）"""

    # 函数说明: __init__ 的用途说明
    def __init__(self, vision_ctx):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['objects_detected', 'no_objects', 'perception_failed'],  # 设置 outcomes
            output_keys=['detected_objects', 'objects_to_pick']  # 设置 output_keys
        )  # 执行语句
        self.vision_ctx = vision_ctx  # 设置 self.vision_ctx
        self.target_classes = rospy.get_param('~detection/target_classes', [  # 设置 self.target_classes
            'cup', 'plate', 'spoon', 'fork', 'knife', 'bowl',  # 执行语句
            'apple', 'banana', 'bread', 'bottle',  # 执行语句
            'wrapper', 'tissue', 'napkin'  # 执行语句
        ])  # 执行语句
        self.confidence_threshold = rospy.get_param('~detection/confidence_threshold', 0.3)  # 设置 self.confidence_threshold
        self.communicate_perception = rospy.get_param('~communicate_perception', True)  # 设置 self.communicate_perception

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== 评估场景 ==========")  # 输出信息日志
        rospy.loginfo("等待相机图像并进行视觉检测...")  # 输出信息日志

        try:  # 开始异常处理
            detected = self.vision_ctx.detect_objects(  # 设置 detected
                self.target_classes, confidence_threshold=self.confidence_threshold  # 设置 self.target_classes, confidence_threshold
            )  # 执行语句
        except Exception as e:  # 异常分支
            rospy.logerr(f"视觉检测失败: {e}")  # 输出错误日志
            return 'perception_failed'  # 返回结果

        if detected:  # 条件判断
            rospy.loginfo(f"✓ 检测到 {len(detected)} 个物体:")  # 输出信息日志
            for obj in detected:  # 循环遍历
                rospy.loginfo(f"  - {obj.class_name} (置信度: {obj.confidence:.2f})")  # 输出信息日志
                if self.communicate_perception:  # 条件判断
                    rospy.loginfo(f"[感知公示] 发现 {obj.class_name} 置信度 {obj.confidence:.0%}")  # 输出信息日志

            userdata.detected_objects = detected  # 设置 userdata.detected_objects
            userdata.objects_to_pick = list(detected)  # 设置 userdata.objects_to_pick
            return 'objects_detected'  # 返回结果

        rospy.logwarn("未检测到物体")  # 输出警告日志
        return 'no_objects'  # 返回结果


# 类说明: SelectTarget 类的功能说明
class SelectTarget(smach.State):  # 定义类SelectTarget的结构
    """[DRY RUN] 本地分类规则，不调用 /classify_object 服务"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['target_selected', 'no_more_objects', 'failed'],  # 设置 outcomes
            input_keys=['objects_to_pick', 'current_object_index'],  # 设置 input_keys
            output_keys=['selected_object', 'object_category', 'destination', 'current_object_index']  # 设置 output_keys
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== [DRY RUN] 选择目标物品 ==========")  # 输出信息日志

        if 'objects_to_pick' not in userdata or len(userdata['objects_to_pick']) == 0:  # 条件判断
            rospy.loginfo("没有更多物品需要处理")  # 输出信息日志
            return 'no_more_objects'  # 返回结果

        if 'current_object_index' not in userdata:  # 条件判断
            userdata['current_object_index'] = 0  # 执行语句

        index = userdata['current_object_index']  # 设置 index

        if index >= len(userdata['objects_to_pick']):  # 条件判断
            rospy.loginfo("所有物品已处理完毕")  # 输出信息日志
            return 'no_more_objects'  # 返回结果

        selected_obj = userdata['objects_to_pick'][index]  # 设置 selected_obj
        rospy.loginfo(f"选择物品 [{index + 1}/{len(userdata['objects_to_pick'])}]: {selected_obj.class_name}")  # 输出信息日志

        # 本地分类（不调用 /classify_object 服务）
        name = selected_obj.class_name.lower()  # 设置 name
        try:  # 开始异常处理
            trash_kws = rospy.get_param('~task/trash_keywords', [])  # 设置 trash_kws
        except Exception:  # 异常分支
            trash_kws = []  # 设置 trash_kws

        if any(k in name for k in CLEANABLE_KEYWORDS):  # 条件判断
            category, destination = 'cleanable', 'dishwasher'  # 设置 category, destination
        elif any(k in name for k in trash_kws):  # 条件判断
            category, destination = 'trash', 'trash_bin'  # 设置 category, destination
        else:  # 条件分支
            category, destination = 'other', 'cabinet'  # 设置 category, destination

        rospy.loginfo(f"[DRY RUN] 本地分类: 类别={category}, 目的地={destination}")  # 输出信息日志

        userdata['selected_object'] = selected_obj  # 执行语句
        userdata['object_category'] = category  # 执行语句
        userdata['destination'] = destination  # 执行语句
        userdata['current_object_index'] = index + 1  # 执行语句

        return 'target_selected'  # 返回结果


# 类说明: ExecutePick 类的功能说明
class ExecutePick(smach.State):  # 定义类ExecutePick的结构
    """[DRY RUN] 打印六个抓取阶段，直接返回 pick_succeeded"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['pick_succeeded', 'pick_failed', 'fatal_error'],  # 设置 outcomes
            input_keys=['selected_object', 'object_category', 'objects_picked_count'],  # 设置 input_keys
            output_keys=['grasp_pose', 'objects_picked_count']  # 设置 output_keys
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== [DRY RUN] 执行抓取 ==========")  # 输出信息日志

        if 'selected_object' not in userdata:  # 条件判断
            rospy.logerr("未选择目标物品")  # 输出错误日志
            return 'fatal_error'  # 返回结果

        obj = userdata['selected_object']  # 设置 obj
        rospy.loginfo(f"目标物品: {obj.class_name}")  # 输出信息日志

        phases = ['接近目标', '计算抓取姿态', '移动到预抓取位', '张开夹爪', '执行抓取', '收回机械臂']  # 设置 phases
        for phase in phases:  # 循环遍历
            rospy.loginfo(f"  [DRY RUN] 抓取阶段: {phase}")  # 输出信息日志

        rospy.loginfo("[DRY RUN] ✓ 抓取成功（模拟）")  # 输出信息日志
        userdata['grasp_pose'] = None  # 执行语句
        userdata.objects_picked_count += 1  # 执行语句
        return 'pick_succeeded'  # 返回结果


# 类说明: NavigateBackToKitchen 类的功能说明
class NavigateBackToKitchen(smach.State):  # 定义类NavigateBackToKitchen的结构
    """【真实导航】放置完成后返回厨房桌边，准备抓取下一件物品"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['returned', 'navigation_failed']  # 设置 outcomes
        )  # 执行语句
        self.nav_client = actionlib.SimpleActionClient(  # 设置 self.nav_client
            '/navigate_to_location',  # 执行语句
            NavigateToLocationAction  # 执行语句
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== 返回厨房桌边 ==========")  # 输出信息日志
        if _navigate_with_midpoints(self.nav_client, "kitchen", "[NavigateBackToKitchen]"):  # 条件判断
            rospy.loginfo("✓ 已返回厨房桌边，准备抓取下一件")  # 输出信息日志
            return 'returned'  # 返回结果
        rospy.logwarn("返回厨房失败，尝试继续")  # 输出警告日志
        return 'navigation_failed'  # 返回结果


# 类说明: NavigateToDest 类的功能说明
class NavigateToDest(smach.State):  # 定义类NavigateToDest的结构
    """【真实导航】完全复制自 navigate_to_dest.py"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['arrived', 'navigation_failed', 'fatal_error'],  # 设置 outcomes
            input_keys=['destination']  # 设置 input_keys
        )  # 执行语句
        self.nav_client = actionlib.SimpleActionClient(  # 设置 self.nav_client
            '/navigate_to_location',  # 执行语句
            NavigateToLocationAction  # 执行语句
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== 导航到目的地 ==========")  # 输出信息日志

        if 'destination' not in userdata:  # 条件判断
            rospy.logerr("未指定目的地")  # 输出错误日志
            return 'fatal_error'  # 返回结果

        destination = userdata['destination']  # 设置 destination
        rospy.loginfo(f"目的地: {destination}")  # 输出信息日志

        if _navigate_with_midpoints(self.nav_client, destination, "[NavigateToDest]"):  # 条件判断
            return 'arrived'  # 返回结果
        return 'navigation_failed'  # 返回结果


# 类说明: PerceiveDest 类的功能说明
class PerceiveDest(smach.State):  # 定义类PerceiveDest的结构
    """[DRY RUN] 跳过柜子检测，返回默认放置姿态"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['perception_done', 'perception_failed', 'fatal_error'],  # 设置 outcomes
            input_keys=['destination', 'selected_object'],  # 设置 input_keys
            output_keys=['shelf_info', 'place_pose', 'target_layer']  # 设置 output_keys
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== [DRY RUN] 感知目的地 ==========")  # 输出信息日志

        destination = userdata.destination  # 设置 destination
        rospy.loginfo(f"[DRY RUN] 目的地: {destination}，跳过视觉感知（柜子检测/放置姿态计算）")  # 输出信息日志

        userdata['shelf_info'] = None  # 执行语句
        userdata['target_layer'] = 0  # 执行语句

        default_pose = Pose()  # 设置 default_pose
        default_pose.orientation.w = 1.0  # 设置 default_pose.orientation.w
        default_pose.position.x = 0.6  # 设置 default_pose.position.x
        default_pose.position.z = 0.8  # 设置 default_pose.position.z

        userdata['place_pose'] = default_pose  # 执行语句
        rospy.loginfo("[DRY RUN] ✓ 使用默认放置姿态 (x=0.6, z=0.8)")  # 输出信息日志
        return 'perception_done'  # 返回结果


# 类说明: ExecutePlace 类的功能说明
class ExecutePlace(smach.State):  # 定义类ExecutePlace的结构
    """[DRY RUN] 打印放置阶段（含洗碗机门提示），直接返回 place_succeeded"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['place_succeeded', 'place_failed', 'fatal_error'],  # 设置 outcomes
            input_keys=['place_pose', 'selected_object', 'objects_placed_count', 'destination'],  # 设置 input_keys
            output_keys=['objects_placed_count']  # 设置 output_keys
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== [DRY RUN] 执行放置 ==========")  # 输出信息日志

        destination = userdata.destination  # 设置 destination

        # 规则书 Rule #4：洗碗机默认关闭，需通知 referee 开门
        if destination == 'dishwasher':  # 条件判断
            rospy.logwarn("[DRY RUN][洗碗机门] 请 referee 打开洗碗机门（模拟 TTS 提示）")  # 输出警告日志
            rospy.loginfo("[DRY RUN] TTS: Please open the dishwasher door.")  # 输出信息日志

        phases = ['移动到放置预位', '打开夹爪', '降低机械臂', '释放物品']  # 设置 phases
        for phase in phases:  # 循环遍历
            rospy.loginfo(f"  [DRY RUN] 放置阶段: {phase}")  # 输出信息日志

        rospy.loginfo("[DRY RUN] ✓ 放置成功（模拟）")  # 输出信息日志

        userdata.objects_placed_count += 1  # 执行语句
        return 'place_succeeded'  # 返回结果


# 类说明: ServeBreakfast 类的功能说明
class ServeBreakfast(smach.State):  # 定义类ServeBreakfast的结构
    """
    准备早餐状态（规则书 §5.2 Main Goal）
    _navigate_to() 和 _compute_placement_pose() 保留真实实现；
    _detect_item() / _pick_item() / _place_item() 替换为 print。
    """

    # 函数说明: __init__ 的用途说明
    def __init__(self, vision_ctx):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['breakfast_served', 'breakfast_skipped', 'failed']  # 设置 outcomes
        )  # 执行语句
        self.vision_ctx = vision_ctx  # 设置 self.vision_ctx

        # 早餐物品定义（规则书 §5.2）
        self.breakfast_items = [  # 设置 self.breakfast_items
            {'name': 'bowl',   'source': 'kitchen_surface'},  # 执行语句
            {'name': 'spoon',  'source': 'kitchen_surface'},  # 执行语句
            {'name': 'cereal', 'source': 'cabinet'},  # 执行语句
            {'name': 'milk',   'source': 'cabinet'},  # 执行语句
        ]  # 执行语句

        # 桌面摆放偏移
        self.item_spacing = rospy.get_param('~breakfast/item_spacing', 0.15)  # 设置 self.item_spacing

        # 相邻物品配对偏移（规则书：spoon 紧靠 bowl，milk 在 cereal 左侧）
        self.pair_offsets = {  # 设置 self.pair_offsets
            'spoon':  [0.10, 0.0],  # 执行语句
            'milk':   [-0.10, 0.0],  # 执行语句
        }  # 执行语句

        # 导航客户端（真实）
        self.nav_client = actionlib.SimpleActionClient(  # 设置 self.nav_client
            '/navigate_to_location',  # 执行语句
            NavigateToLocationAction  # 执行语句
        )  # 执行语句

    # ------------------------------------------------------------------
    # 真实实现：导航
    # ------------------------------------------------------------------
    # 函数说明: _navigate_to 的用途说明
    def _navigate_to(self, location_name):  # 定义函数_navigate_to
        """导航到指定地点，返回 True/False（完全复制自原版）"""
        return _navigate_with_midpoints(self.nav_client, location_name, "[ServeBreakfast]")  # 返回结果

    # ------------------------------------------------------------------
    # 真实实现：放置姿态计算（纯数学，无 ROS 调用）
    # ------------------------------------------------------------------
    # 函数说明: _compute_placement_pose 的用途说明
    def _compute_placement_pose(self, item_name, index, prev_pose):  # 定义函数_compute_placement_pose
        """
        计算桌面放置位姿（完全复制自原版）：
        - 第一件物品放在桌子中央
        - 后续物品在 x 方向累加 item_spacing
        - spoon/milk 紧靠前一件物品（使用 pair_offsets）
        """
        pose = Pose()  # 设置 pose
        pose.orientation.w = 1.0  # 设置 pose.orientation.w

        base_x = 0.6  # 设置 base_x
        base_y = 0.0  # 设置 base_y
        base_z = 0.75  # 设置 base_z

        if prev_pose is None:  # 条件判断
            pose.position.x = base_x  # 设置 pose.position.x
            pose.position.y = base_y  # 设置 pose.position.y
            pose.position.z = base_z  # 设置 pose.position.z
        elif item_name in self.pair_offsets:  # 条件判断
            dx, dy = self.pair_offsets[item_name]  # 设置 dx, dy
            pose.position.x = prev_pose.position.x + dx  # 设置 pose.position.x
            pose.position.y = prev_pose.position.y + dy  # 设置 pose.position.y
            pose.position.z = base_z  # 设置 pose.position.z
        else:  # 条件分支
            pose.position.x = base_x  # 设置 pose.position.x
            pose.position.y = base_y + index * self.item_spacing  # 设置 pose.position.y
            pose.position.z = base_z  # 设置 pose.position.z

        return pose  # 返回结果

    # ------------------------------------------------------------------
    # [VISION] 检测、抓取、放置
    # ------------------------------------------------------------------
    # 函数说明: _detect_item 的用途说明
    def _detect_item(self, item_name):  # 定义函数_detect_item
        """使用视觉检测指定物品，失败回退模拟"""
        rospy.loginfo(f"  [VISION] 检测 {item_name}...")  # 输出信息日志
        obj = self.vision_ctx.detect_single(item_name, confidence_threshold=0.3)  # 设置 obj
        if obj is None:  # 条件判断
            rospy.logwarn(f"  [VISION] 未检测到 {item_name}，回退模拟")  # 输出警告日志
            obj = MOCK_BREAKFAST_OBJECTS.get(  # 设置 obj
                item_name,  # 执行语句
                MockDetectedObject(item_name)  # 执行语句
            )  # 执行语句
        rospy.loginfo(f"  [感知公示] 发现 {obj.class_name} 置信度 {obj.confidence:.0%}")  # 输出信息日志
        return obj  # 返回结果

    # 函数说明: _pick_item 的用途说明
    def _pick_item(self, detected_obj):  # 定义函数_pick_item
        """[DRY RUN] 打印六个抓取阶段，直接返回 True"""
        rospy.loginfo(f"  [DRY RUN] 模拟抓取 {detected_obj.class_name}...")  # 输出信息日志
        phases = ['接近目标', '计算抓取姿态', '移动到预抓取位', '张开夹爪', '执行抓取', '收回机械臂']  # 设置 phases
        for phase in phases:  # 循环遍历
            rospy.loginfo(f"    [DRY RUN] 抓取阶段: {phase}")  # 输出信息日志
        rospy.loginfo(f"  [DRY RUN] ✓ 抓取 {detected_obj.class_name} 成功（模拟）")  # 输出信息日志
        return True  # 返回结果

    # 函数说明: _place_item 的用途说明
    def _place_item(self, target_pose):  # 定义函数_place_item
        """[DRY RUN] 打印四个放置阶段，直接返回 True"""
        rospy.loginfo(f"  [DRY RUN] 模拟放置到 "  # 输出信息日志
                      f"({target_pose.position.x:.2f}, "  # 执行语句
                      f"{target_pose.position.y:.2f}, "  # 执行语句
                      f"{target_pose.position.z:.2f})...")  # 执行语句
        phases = ['移动到放置预位', '打开夹爪', '降低机械臂', '释放物品']  # 设置 phases
        for phase in phases:  # 循环遍历
            rospy.loginfo(f"    [DRY RUN] 放置阶段: {phase}")  # 输出信息日志
        rospy.loginfo(f"  [DRY RUN] ✓ 放置成功（模拟）")  # 输出信息日志
        return True  # 返回结果

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("========== 准备早餐 ==========")  # 输出信息日志

        enable_breakfast = rospy.get_param('~enable_breakfast_serving',  # 设置 enable_breakfast
                                           rospy.get_param('~enable_breakfast', False))  # 执行语句
        if not enable_breakfast:  # 条件判断
            rospy.loginfo("[ServeBreakfast] 早餐任务已跳过（未启用）")  # 输出信息日志
            return 'breakfast_skipped'  # 返回结果

        rospy.loginfo(f"[ServeBreakfast] 开始处理 {len(self.breakfast_items)} 件早餐物品")  # 输出信息日志

        prev_pose = None  # 设置 prev_pose
        failed_items = []  # 设置 failed_items

        for idx, item_info in enumerate(self.breakfast_items):  # 循环遍历
            item_name = item_info['name']  # 设置 item_name
            source = item_info['source']  # 设置 source

            rospy.loginfo(f"[ServeBreakfast] [{idx+1}/{len(self.breakfast_items)}] "  # 输出信息日志
                          f"处理 {item_name}（从 {source} 取）")  # 执行语句

            # 1. 导航到物品存放处（真实导航）
            rospy.loginfo(f"  Step 1: 导航到 {source}")  # 输出信息日志
            if not self._navigate_to(source):  # 条件判断
                rospy.logwarn(f"  导航失败，跳过 {item_name}")  # 输出警告日志
                failed_items.append(item_name)  # 执行语句
                continue  # 继续下一轮循环

            # 2. 检测物品（视觉）
            rospy.loginfo(f"  Step 2: 检测 {item_name}")  # 输出信息日志
            detected_obj = self._detect_item(item_name)  # 设置 detected_obj
            if detected_obj is None:  # 条件判断
                rospy.logwarn(f"  未检测到 {item_name}，跳过")  # 输出警告日志
                failed_items.append(item_name)  # 执行语句
                continue  # 继续下一轮循环

            # 3. 抓取（模拟）
            rospy.loginfo(f"  Step 3: 抓取 {item_name}")  # 输出信息日志
            if not self._pick_item(detected_obj):  # 条件判断
                rospy.logwarn(f"  抓取 {item_name} 失败，跳过")  # 输出警告日志
                failed_items.append(item_name)  # 执行语句
                continue  # 继续下一轮循环

            # 4. 导航到餐桌（真实导航）
            rospy.loginfo(f"  Step 4: 导航到 dining_table")  # 输出信息日志
            if not self._navigate_to('dining_table'):  # 条件判断
                rospy.logwarn(f"  导航到餐桌失败，跳过 {item_name}")  # 输出警告日志
                failed_items.append(item_name)  # 执行语句
                continue  # 继续下一轮循环

            # 5. 计算放置位姿并放置（姿态计算真实，放置模拟）
            target_pose = self._compute_placement_pose(item_name, idx, prev_pose)  # 设置 target_pose
            rospy.loginfo(f"  Step 5: 放置 {item_name} 到 "  # 输出信息日志
                          f"({target_pose.position.x:.2f}, {target_pose.position.y:.2f}, "  # 执行语句
                          f"{target_pose.position.z:.2f})")  # 执行语句
            if not self._place_item(target_pose):  # 条件判断
                rospy.logwarn(f"  放置 {item_name} 失败")  # 输出警告日志
                failed_items.append(item_name)  # 执行语句
            prev_pose = target_pose  # 设置 prev_pose

        if failed_items:  # 条件判断
            rospy.logwarn(f"[ServeBreakfast] 以下物品处理失败: {failed_items}")  # 输出警告日志
            if len(failed_items) == len(self.breakfast_items):  # 条件判断
                return 'failed'  # 返回结果

        rospy.loginfo("[ServeBreakfast] 早餐准备完成！")  # 输出信息日志
        return 'breakfast_served'  # 返回结果


# 类说明: TaskCompleted 类的功能说明
class TaskCompleted(smach.State):  # 定义类TaskCompleted的结构
    """完全复制自 task_completed.py"""

    # 函数说明: __init__ 的用途说明
    def __init__(self):  # 定义函数__init__
        smach.State.__init__(  # 执行语句
            self,  # 执行语句
            outcomes=['done'],  # 设置 outcomes
            input_keys=['objects_picked_count', 'objects_placed_count', 'failed_objects']  # 设置 input_keys
        )  # 执行语句

    # 函数说明: execute 的用途说明
    def execute(self, userdata):  # 定义函数execute
        rospy.loginfo("=" * 60)  # 输出信息日志
        rospy.loginfo(" " * 15 + "任务完成！")  # 输出信息日志
        rospy.loginfo("=" * 60)  # 输出信息日志

        # 正确代码
        picked = userdata.objects_picked_count  # 设置 picked
        placed = userdata.objects_placed_count  # 设置 placed
        failed = userdata.failed_objects  # 设置 failed

        rospy.loginfo(f"抓取物品数: {picked}")  # 输出信息日志
        rospy.loginfo(f"放置物品数: {placed}")  # 输出信息日志

        if len(failed) > 0:  # 条件判断
            rospy.loginfo(f"失败物品数: {len(failed)}")  # 输出信息日志
            rospy.loginfo(f"失败物品列表: {', '.join(failed)}")  # 输出信息日志

        rospy.loginfo("=" * 60)  # 输出信息日志

        return 'done'  # 返回结果


# ============================================================
# 状态机构建（完全复制自 pick_place_task.py）
# ============================================================

# 函数说明: create_cleanup_loop 的用途说明
def create_cleanup_loop():  # 定义函数create_cleanup_loop
    """创建内层清理循环状态机"""
    cleanup_sm = smach.StateMachine(  # 设置 cleanup_sm
        outcomes=['all_objects_processed', 'cleanup_failed'],  # 设置 outcomes
        input_keys=['detected_objects', 'objects_to_pick', 'objects_picked_count', 'objects_placed_count'],  # 设置 input_keys
        output_keys=['objects_picked_count', 'objects_placed_count', 'failed_objects']  # 设置 output_keys
    )  # 执行语句

    cleanup_sm.userdata.current_object_index = 0  # 设置 cleanup_sm.userdata.current_object_index
    cleanup_sm.userdata.selected_object = None  # 设置 cleanup_sm.userdata.selected_object
    cleanup_sm.userdata.object_category = ''  # 设置 cleanup_sm.userdata.object_category
    cleanup_sm.userdata.destination = ''  # 设置 cleanup_sm.userdata.destination
    cleanup_sm.userdata.grasp_pose = None  # 设置 cleanup_sm.userdata.grasp_pose
    cleanup_sm.userdata.place_pose = None  # 设置 cleanup_sm.userdata.place_pose
    cleanup_sm.userdata.shelf_info = None  # 设置 cleanup_sm.userdata.shelf_info
    cleanup_sm.userdata.target_layer = 0  # 设置 cleanup_sm.userdata.target_layer
    cleanup_sm.userdata.failed_objects = []  # 设置 cleanup_sm.userdata.failed_objects

    with cleanup_sm:  # 上下文管理
        smach.StateMachine.add(  # 执行语句
            'SELECT_TARGET',  # 执行语句
            SelectTarget(),  # 执行语句
            transitions={  # 设置 transitions
                'target_selected': 'EXECUTE_PICK',  # 执行语句
                'no_more_objects': 'all_objects_processed',  # 执行语句
                'failed': 'SELECT_TARGET'  # 分类失败跳过此物品，index 已递增，继续下一件
            },  # 执行语句
            remapping={  # 设置 remapping
                'objects_to_pick': 'objects_to_pick',  # 执行语句
                'current_object_index': 'current_object_index',  # 执行语句
                'selected_object': 'selected_object',  # 执行语句
                'object_category': 'object_category',  # 执行语句
                'destination': 'destination'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'EXECUTE_PICK',  # 执行语句
            ExecutePick(),  # 执行语句
            transitions={  # 设置 transitions
                'pick_succeeded': 'NAVIGATE_TO_DEST',  # 执行语句
                'pick_failed': 'SELECT_TARGET',  # 执行语句
                'fatal_error': 'cleanup_failed'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'selected_object': 'selected_object',  # 执行语句
                'object_category': 'object_category',  # 执行语句
                'grasp_pose': 'grasp_pose',  # 执行语句
                'objects_picked_count': 'objects_picked_count'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'NAVIGATE_TO_DEST',  # 执行语句
            NavigateToDest(),  # 执行语句
            transitions={  # 设置 transitions
                'arrived': 'PERCEIVE_DEST',  # 执行语句
                'navigation_failed': 'EXECUTE_PLACE',  # 执行语句
                'fatal_error': 'cleanup_failed'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'destination': 'destination'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'PERCEIVE_DEST',  # 执行语句
            PerceiveDest(),  # 执行语句
            transitions={  # 设置 transitions
                'perception_done': 'EXECUTE_PLACE',  # 执行语句
                'perception_failed': 'EXECUTE_PLACE',  # 执行语句
                'fatal_error': 'cleanup_failed'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'destination': 'destination',  # 执行语句
                'selected_object': 'selected_object',  # 执行语句
                'shelf_info': 'shelf_info',  # 执行语句
                'place_pose': 'place_pose',  # 执行语句
                'target_layer': 'target_layer'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'EXECUTE_PLACE',  # 执行语句
            ExecutePlace(),  # 执行语句
            transitions={  # 设置 transitions
                'place_succeeded': 'NAVIGATE_BACK_TO_KITCHEN',  # 放完再回桌边
                'place_failed':    'NAVIGATE_BACK_TO_KITCHEN',  # 失败也要回去取下一件
                'fatal_error':     'cleanup_failed'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'place_pose': 'place_pose',  # 执行语句
                'selected_object': 'selected_object',  # 执行语句
                'objects_placed_count': 'objects_placed_count',  # 执行语句
                'destination': 'destination'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        # 6. 放置完毕后返回厨房桌边，准备处理下一件物品
        smach.StateMachine.add(  # 执行语句
            'NAVIGATE_BACK_TO_KITCHEN',  # 执行语句
            NavigateBackToKitchen(),  # 执行语句
            transitions={  # 设置 transitions
                'returned':          'SELECT_TARGET',  # 正常返回，继续下一件
                'navigation_failed': 'SELECT_TARGET'   # 导航失败也尝试继续
            }  # 执行语句
        )  # 执行语句

    return cleanup_sm  # 返回结果


# 函数说明: create_main_state_machine 的用途说明
def create_main_state_machine(vision_ctx):  # 定义函数create_main_state_machine
    """创建外层主状态机"""
    main_sm = smach.StateMachine(outcomes=['task_succeeded', 'task_failed'])  # 设置 main_sm

    main_sm.userdata.detected_objects = []  # 设置 main_sm.userdata.detected_objects
    main_sm.userdata.objects_to_pick = []  # 设置 main_sm.userdata.objects_to_pick
    main_sm.userdata.objects_picked_count = 0  # 设置 main_sm.userdata.objects_picked_count
    main_sm.userdata.objects_placed_count = 0  # 设置 main_sm.userdata.objects_placed_count
    main_sm.userdata.failed_objects = []  # 设置 main_sm.userdata.failed_objects

    with main_sm:  # 上下文管理
        smach.StateMachine.add(  # 执行语句
            'INIT_SYSTEM',  # 执行语句
            InitSystem(),  # 执行语句
            transitions={  # 设置 transitions
                'initialized': 'NAVIGATE_TO_KITCHEN',  # 执行语句
                'init_failed': 'task_failed'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'NAVIGATE_TO_KITCHEN',  # 执行语句
            NavigateToKitchen(),  # 执行语句
            transitions={  # 设置 transitions
                'arrived': 'ASSESS_SCENE',  # 执行语句
                'navigation_failed': 'task_failed'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'ASSESS_SCENE',  # 执行语句
            AssessScene(vision_ctx),  # 执行语句
            transitions={  # 设置 transitions
                'objects_detected': 'TABLE_CLEANUP_LOOP',  # 执行语句
                'no_objects': 'SERVE_BREAKFAST',  # 执行语句
                'perception_failed': 'task_failed'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'detected_objects': 'detected_objects',  # 执行语句
                'objects_to_pick': 'objects_to_pick'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'TABLE_CLEANUP_LOOP',  # 执行语句
            create_cleanup_loop(),  # 执行语句
            transitions={  # 设置 transitions
                'all_objects_processed': 'SERVE_BREAKFAST',  # 执行语句
                'cleanup_failed': 'task_failed'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'detected_objects': 'detected_objects',  # 执行语句
                'objects_to_pick': 'objects_to_pick',  # 执行语句
                'objects_picked_count': 'objects_picked_count',  # 执行语句
                'objects_placed_count': 'objects_placed_count',  # 执行语句
                'failed_objects': 'failed_objects'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'SERVE_BREAKFAST',  # 执行语句
            ServeBreakfast(vision_ctx),  # 执行语句
            transitions={  # 设置 transitions
                'breakfast_served': 'TASK_COMPLETED',  # 执行语句
                'breakfast_skipped': 'TASK_COMPLETED',  # 执行语句
                'failed': 'task_failed'  # 执行语句
            }  # 执行语句
        )  # 执行语句

        smach.StateMachine.add(  # 执行语句
            'TASK_COMPLETED',  # 执行语句
            TaskCompleted(),  # 执行语句
            transitions={  # 设置 transitions
                'done': 'task_succeeded'  # 执行语句
            },  # 执行语句
            remapping={  # 设置 remapping
                'objects_picked_count': 'objects_picked_count',  # 执行语句
                'objects_placed_count': 'objects_placed_count',  # 执行语句
                'failed_objects': 'failed_objects'  # 执行语句
            }  # 执行语句
        )  # 执行语句

    return main_sm  # 返回结果


# 函数说明: main 的用途说明
def main():  # 定义函数main
    """主函数"""
    rospy.init_node('pick_place_run_vision')  # 执行语句

    rospy.loginfo("=" * 60)  # 输出信息日志
    rospy.loginfo(" " * 5 + "RoboCup@Home Pick and Place 任务 [VISION]")  # 输出信息日志
    rospy.loginfo(" " * 5 + "感知 = 视觉 | 抓取/放置 = 模拟 | 导航 = 真实")  # 输出信息日志
    rospy.loginfo("=" * 60)  # 输出信息日志

    vision_ctx = VisionContext()  # 设置 vision_ctx
    sm = create_main_state_machine(vision_ctx)  # 设置 sm

    sis = smach_ros.IntrospectionServer('pick_place_run_vision', sm, '/PICK_PLACE_TASK')  # 设置 sis
    sis.start()  # 执行语句

    rospy.loginfo("SMACH状态机已启动")  # 输出信息日志
    rospy.loginfo("可使用 rosrun smach_viewer smach_viewer.py 进行可视化")  # 输出信息日志
    rospy.loginfo("=" * 60)  # 输出信息日志

    rospy.loginfo("开始执行 Pick and Place 任务（Vision）...")  # 输出信息日志
    outcome = sm.execute()  # 设置 outcome

    rospy.loginfo("=" * 60)  # 输出信息日志
    if outcome == 'task_succeeded':  # 条件判断
        rospy.loginfo("✓ 任务成功完成！[VISION]")  # 输出信息日志
    else:  # 条件分支
        rospy.logerr("✗ 任务失败 [VISION]")  # 输出错误日志
    rospy.loginfo("=" * 60)  # 输出信息日志

    sis.stop()  # 执行语句


if __name__ == '__main__':  # 条件判断
    try:  # 开始异常处理
        main()  # 执行语句
    except rospy.ROSInterruptException:  # 异常分支
        rospy.loginfo("任务被中断")  # 输出信息日志

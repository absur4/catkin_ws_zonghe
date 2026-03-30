#!/usr/bin/env python3
"""
ServeBreakfast State - 准备早餐状态（规则书 §5.2 Main Goal）

按顺序取 bowl → spoon → cereal → milk，依次放到餐桌上：
  - bowl/spoon 来自 kitchen_surface
  - cereal/milk 来自 cabinet
  - spoon 紧靠 bowl 右侧 (+0.10m x)
  - cereal/milk 紧邻放置（milk 在 cereal 左侧 -0.10m x）
  - 每件物品与桌中心相距 item_spacing（累加），保留 ≥5cm 空白
"""
import rospy
import smach
import actionlib
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from robocup_msgs.msg import PickObjectAction, PickObjectGoal
from robocup_msgs.msg import PlaceObjectAction, PlaceObjectGoal
from robocup_msgs.srv import DetectObjects, DetectObjectsRequest
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal


class ServeBreakfast(smach.State):
    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['breakfast_served', 'breakfast_skipped', 'failed']
        )

        # 早餐物品定义（规则书 §5.2）
        self.breakfast_items = [
            {'name': 'bowl',   'source': 'kitchen_surface'},
            {'name': 'spoon',  'source': 'kitchen_surface'},
            {'name': 'cereal', 'source': 'cabinet'},
            {'name': 'milk',   'source': 'cabinet'},
        ]

        # 桌面摆放偏移：第 i 件物品相对桌中心的 x 偏移
        self.item_spacing = rospy.get_param('~breakfast/item_spacing', 0.15)

        # 相邻物品配对偏移（规则书要求 spoon 紧靠 bowl，cereal 紧靠 milk）
        # 键为物品名，值为相对前一件的偏移 [dx, dy]（米）
        self.pair_offsets = {
            'spoon':  [0.10, 0.0],   # spoon 在 bowl 右侧
            'milk':   [-0.10, 0.0],  # milk 在 cereal 左侧
        }

        # 动作客户端
        self.pick_client = actionlib.SimpleActionClient('/pick_object', PickObjectAction)
        self.place_client = actionlib.SimpleActionClient('/place_object', PlaceObjectAction)
        self.nav_client = actionlib.SimpleActionClient('/navigate_to_location', NavigateToLocationAction)

        # 相机订阅
        self.rgb_image = None
        self.depth_image = None
        rospy.Subscriber('/camera/color/image_raw', Image, self._rgb_cb)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self._depth_cb)

    def _rgb_cb(self, msg):
        self.rgb_image = msg

    def _depth_cb(self, msg):
        self.depth_image = msg

    def _wait_for_image(self, timeout=5.0):
        rate = rospy.Rate(10)
        deadline = rospy.Time.now() + rospy.Duration(timeout)
        while (self.rgb_image is None or self.depth_image is None) and rospy.Time.now() < deadline:
            rate.sleep()
        return self.rgb_image is not None and self.depth_image is not None

    def _navigate_to(self, location_name):
        """导航到指定地点，返回 True/False"""
        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr(f"[ServeBreakfast] 导航服务器未响应")
            return False
        goal = NavigateToLocationGoal()
        goal.target_location = location_name
        self.nav_client.send_goal(goal)
        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))
        if not finished:
            self.nav_client.cancel_goal()
            rospy.logerr(f"[ServeBreakfast] 导航到 {location_name} 超时")
            return False
        result = self.nav_client.get_result()
        return result is not None and result.success

    def _detect_item(self, item_name):
        """检测指定物品，返回 DetectedObject 或 None"""
        if not self._wait_for_image(5.0):
            rospy.logerr(f"[ServeBreakfast] 无法获取相机图像")
            return None
        try:
            rospy.wait_for_service('/detect_objects', timeout=10.0)
            detect_srv = rospy.ServiceProxy('/detect_objects', DetectObjects)
            req = DetectObjectsRequest()
            req.target_classes = [item_name]
            req.rgb_image = self.rgb_image
            req.depth_image = self.depth_image
            req.confidence_threshold = 0.3
            resp = detect_srv(req)
            if resp.success and len(resp.detected_objects.objects) > 0:
                obj = resp.detected_objects.objects[0]
                rospy.loginfo(f"[感知公示] 发现 {obj.class_name} 置信度 {obj.confidence:.0%}")
                return obj
            else:
                rospy.logwarn(f"[ServeBreakfast] 未找到 {item_name}")
                return None
        except Exception as e:
            rospy.logerr(f"[ServeBreakfast] 检测失败: {e}")
            return None

    def _pick_item(self, detected_obj):
        """执行抓取，返回 True/False"""
        if not self.pick_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("[ServeBreakfast] 抓取服务器未响应")
            return False
        goal = PickObjectGoal()
        goal.target_object = detected_obj
        goal.grasp_strategy = "adaptive"
        self.pick_client.send_goal(goal)
        finished = self.pick_client.wait_for_result(timeout=rospy.Duration(60.0))
        if not finished:
            self.pick_client.cancel_goal()
            return False
        result = self.pick_client.get_result()
        return result is not None and result.success

    def _place_item(self, target_pose):
        """执行放置，返回 True/False"""
        if not self.place_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("[ServeBreakfast] 放置服务器未响应")
            return False
        goal = PlaceObjectGoal()
        goal.target_pose = target_pose
        goal.place_strategy = "gentle"
        self.place_client.send_goal(goal)
        finished = self.place_client.wait_for_result(timeout=rospy.Duration(60.0))
        if not finished:
            self.place_client.cancel_goal()
            return False
        result = self.place_client.get_result()
        return result is not None and result.success

    def _compute_placement_pose(self, item_name, index, prev_pose):
        """
        计算桌面放置位姿：
        - 第一件物品放在桌子中央（由导航目标点近似）
        - 后续物品在 x 方向累加 item_spacing
        - spoon/milk 紧靠前一件物品（使用 pair_offsets）
        """
        pose = Pose()
        pose.orientation.w = 1.0

        # 桌中心默认坐标（由 dining_table 导航点决定，这里使用 base_link 相对坐标）
        base_x = 0.6   # 机器人前方 0.6m（到桌面）
        base_y = 0.0
        base_z = 0.75  # 桌面高度（米）

        if prev_pose is None:
            # 第一件物品放在桌中央
            pose.position.x = base_x
            pose.position.y = base_y
            pose.position.z = base_z
        elif item_name in self.pair_offsets:
            # 紧靠前一件（spoon 靠 bowl，milk 靠 cereal）
            dx, dy = self.pair_offsets[item_name]
            pose.position.x = prev_pose.position.x + dx
            pose.position.y = prev_pose.position.y + dy
            pose.position.z = base_z
        else:
            # 独立放置，y 方向偏移 item_spacing
            pose.position.x = base_x
            pose.position.y = base_y + index * self.item_spacing
            pose.position.z = base_z

        return pose

    def execute(self, userdata):
        rospy.loginfo("========== 准备早餐 ==========")

        enable_breakfast = rospy.get_param('~enable_breakfast_serving',
                                           rospy.get_param('~enable_breakfast', False))
        if not enable_breakfast:
            rospy.loginfo("[ServeBreakfast] 早餐任务已跳过（未启用）")
            return 'breakfast_skipped'

        rospy.loginfo(f"[ServeBreakfast] 开始处理 {len(self.breakfast_items)} 件早餐物品")

        prev_pose = None
        failed_items = []

        for idx, item_info in enumerate(self.breakfast_items):
            item_name = item_info['name']
            source = item_info['source']

            rospy.loginfo(f"[ServeBreakfast] [{idx+1}/{len(self.breakfast_items)}] "
                          f"处理 {item_name}（从 {source} 取）")

            # 1. 导航到物品存放处
            rospy.loginfo(f"  Step 1: 导航到 {source}")
            if not self._navigate_to(source):
                rospy.logwarn(f"  导航失败，跳过 {item_name}")
                failed_items.append(item_name)
                continue

            # 2. 检测物品
            rospy.loginfo(f"  Step 2: 检测 {item_name}")
            detected_obj = self._detect_item(item_name)
            if detected_obj is None:
                rospy.logwarn(f"  未检测到 {item_name}，跳过")
                failed_items.append(item_name)
                continue

            # 3. 抓取
            rospy.loginfo(f"  Step 3: 抓取 {item_name}")
            if not self._pick_item(detected_obj):
                rospy.logwarn(f"  抓取 {item_name} 失败，跳过")
                failed_items.append(item_name)
                continue

            # 4. 导航到餐桌
            rospy.loginfo(f"  Step 4: 导航到 dining_table")
            if not self._navigate_to('dining_table'):
                rospy.logwarn(f"  导航到餐桌失败，跳过 {item_name}")
                failed_items.append(item_name)
                continue

            # 5. 计算放置位姿并放置
            target_pose = self._compute_placement_pose(item_name, idx, prev_pose)
            rospy.loginfo(f"  Step 5: 放置 {item_name} 到 "
                          f"({target_pose.position.x:.2f}, {target_pose.position.y:.2f}, "
                          f"{target_pose.position.z:.2f})")
            if not self._place_item(target_pose):
                rospy.logwarn(f"  放置 {item_name} 失败")
                failed_items.append(item_name)
                # 不 continue：位姿仍更新，以免后续物品叠放
            prev_pose = target_pose

        if failed_items:
            rospy.logwarn(f"[ServeBreakfast] 以下物品处理失败: {failed_items}")
            if len(failed_items) == len(self.breakfast_items):
                return 'failed'

        rospy.loginfo("[ServeBreakfast] 早餐准备完成！")
        return 'breakfast_served'

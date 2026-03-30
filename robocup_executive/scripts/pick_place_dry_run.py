#!/usr/bin/env python3

import rospy
import smach
import smach_ros
import actionlib
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal
from geometry_msgs.msg import Pose, Point  # 增加 Point


# ============================================================
# 可修改区域：模拟检测到的桌面物品（名称, 置信度, x, y, z）
# x: 机器人正前方距离（m），y: 左正右负，z: 桌面高度（m）
# 坐标在机器人到达 kitchen 位置后的 base_link 帧下定义。
# ============================================================
MOCK_TABLE_OBJECTS = [
    ('cup',   0.85, 0.55,  0.10, 0.30),
    ('plate', 0.72, 0.60, -0.05, 0.28),
    ('apple', 0.65, 0.58,  0.20, 0.32),
]
FRUIT_KEYWORDS = ['apple', 'banana', 'orange']

# 用于本地分类的关键词列表（与 object_classifier.py 保持一致）
CLEANABLE_KEYWORDS = ['cup', 'mug', 'plate' 'dish', 'bowl', 'spoon', 'fork', 'knife'] #原来还有个plate


# ============================================================
# 模拟 DetectedObject 消息
# ============================================================
class MockDetectedObject:
    """模拟 robocup_msgs/DetectedObject，供 dry run 使用"""
    def __init__(self, class_name, confidence=0.80, x=0.5, y=0.0, z=0.3):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = []
        # centroid 即物品在 base_link 坐标系下的三维位置
        self.centroid = Point(x=x, y=y, z=z)
        self.pose = Pose()
        self.pose.position.x = x
        self.pose.position.y = y
        self.pose.position.z = z
        self.pose.orientation.w = 1.0
        self.category = ''


# ============================================================
# 可修改区域：早餐物品的模拟位置（按物品名索引）
# 坐标在机器人到达对应 source 位置后的 base_link 帧下定义。
# ============================================================
MOCK_BREAKFAST_OBJECTS = {
    'bowl':   MockDetectedObject('bowl',   0.88, 0.50,  0.00, 0.30),
    'spoon':  MockDetectedObject('spoon',  0.82, 0.52,  0.08, 0.30),
    'cereal': MockDetectedObject('cereal', 0.76, 0.50, -0.05, 0.35),
    'milk':   MockDetectedObject('milk',   0.79, 0.52,  0.05, 0.38),
}


# ============================================================
# 状态类定义
# ============================================================

class InitSystem(smach.State):
    """[DRY RUN] 跳过所有服务检查，直接返回 initialized"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['initialized', 'init_failed']
        )

    def execute(self, userdata):
        rospy.loginfo("========== [DRY RUN] 初始化系统 ==========")
        services_to_check = [
            '/detect_objects',
            '/classify_object',
            '/compute_grasp_pose',
            '/compute_place_pose'
        ]
        for s in services_to_check:
            rospy.loginfo(f"[DRY RUN] 模拟检查服务 {s} ... OK")
        for a in ['/pick_object', '/place_object', '/navigate_to_location']:
            rospy.loginfo(f"[DRY RUN] 模拟检查动作服务器 {a} ... OK")
        rospy.loginfo("[DRY RUN] 系统初始化完成（模拟）")
        return 'initialized'


class NavigateToKitchen(smach.State):
    """【真实导航】完全复制自 navigate_to_kitchen.py"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['arrived', 'navigation_failed']
        )
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )
        rospy.loginfo("NavigateToKitchen: 等待导航动作服务器...")

    def execute(self, userdata):
        rospy.loginfo("========== 导航到厨房 ==========")

        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("导航动作服务器未响应")
            return 'navigation_failed'

        goal = NavigateToLocationGoal()
        goal.target_location = "kitchen"

        rospy.loginfo("发送导航目标: kitchen")
        self.nav_client.send_goal(goal)

        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))

        if not finished:
            rospy.logerr("导航超时")
            self.nav_client.cancel_goal()
            return 'navigation_failed'

        result = self.nav_client.get_result()

        if result and result.success:
            rospy.loginfo("成功到达厨房！")
            return 'arrived'
        else:
            rospy.logerr("导航失败")
            return 'navigation_failed'


class AssessScene(smach.State):
    """[DRY RUN] 跳过相机和检测服务，返回 MOCK_TABLE_OBJECTS"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['objects_detected', 'no_objects', 'perception_failed'],
            output_keys=['detected_objects', 'objects_to_pick']
        )

    def execute(self, userdata):
        rospy.loginfo("========== [DRY RUN] 评估场景 ==========")
        rospy.loginfo("[DRY RUN] 跳过相机等待和 /detect_objects 服务调用")

        mock_objects = [MockDetectedObject(*args) for args in MOCK_TABLE_OBJECTS]

        rospy.loginfo(f"[DRY RUN] 模拟检测到 {len(mock_objects)} 个物体:")
        for obj in mock_objects:
            rospy.loginfo(f"  - {obj.class_name} (置信度: {obj.confidence:.2f})")
            rospy.loginfo(f"[感知公示] 发现 {obj.class_name} 置信度 {obj.confidence:.0%}")

        userdata.detected_objects = mock_objects
        userdata.objects_to_pick = list(mock_objects)

        if len(mock_objects) > 0:
            return 'objects_detected'
        else:
            return 'no_objects'


class SelectTarget(smach.State):
    """[DRY RUN] 本地分类规则，不调用 /classify_object 服务"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['target_selected', 'no_more_objects', 'failed'],
            input_keys=['objects_to_pick', 'current_object_index'],
            output_keys=['selected_object', 'object_category', 'destination', 'current_object_index']
        )

    def execute(self, userdata):
        rospy.loginfo("========== [DRY RUN] 选择目标物品 ==========")

        if 'objects_to_pick' not in userdata or len(userdata['objects_to_pick']) == 0:
            rospy.loginfo("没有更多物品需要处理")
            return 'no_more_objects'

        if 'current_object_index' not in userdata:
            userdata['current_object_index'] = 0

        index = userdata['current_object_index']

        if index >= len(userdata['objects_to_pick']):
            rospy.loginfo("所有物品已处理完毕")
            return 'no_more_objects'

        selected_obj = userdata['objects_to_pick'][index]
        rospy.loginfo(f"选择物品 [{index + 1}/{len(userdata['objects_to_pick'])}]: {selected_obj.class_name}")

        # 本地分类（不调用 /classify_object 服务）
        name = selected_obj.class_name.lower()
        try:
            trash_kws = rospy.get_param('~task/trash_keywords', [])
        except Exception:
            trash_kws = []

        if any(k in name for k in CLEANABLE_KEYWORDS):
            category, destination = 'cleanable', 'dishwasher'
        # elif any(k in name for k in FRUIT_KEYWORDS):
        #     category, destination = 'fruit', 'dining_table'  # 新增的水果分类
        elif any(k in name for k in trash_kws):
            category, destination = 'trash', 'trash_bin'

        else:
            category, destination = 'other', 'cabinet'

        rospy.loginfo(f"[DRY RUN] 本地分类: 类别={category}, 目的地={destination}")

        userdata['selected_object'] = selected_obj
        userdata['object_category'] = category
        userdata['destination'] = destination
        userdata['current_object_index'] = index + 1

        return 'target_selected'


class ExecutePick(smach.State):
    """[DRY RUN] 打印六个抓取阶段，直接返回 pick_succeeded"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['pick_succeeded', 'pick_failed', 'fatal_error'],
            input_keys=['selected_object', 'object_category', 'objects_picked_count'],
            output_keys=['grasp_pose', 'objects_picked_count']
        )

    def execute(self, userdata):
        rospy.loginfo("========== [DRY RUN] 执行抓取 ==========")

        if 'selected_object' not in userdata:
            rospy.logerr("未选择目标物品")
            return 'fatal_error'

        obj = userdata['selected_object']
        rospy.loginfo(f"目标物品: {obj.class_name}")

        phases = ['接近目标', '计算抓取姿态', '移动到预抓取位', '张开夹爪', '执行抓取', '收回机械臂']
        for phase in phases:
            rospy.loginfo(f"  [DRY RUN] 抓取阶段: {phase}")

        rospy.loginfo("[DRY RUN] ✓ 抓取成功（模拟）")
        userdata['grasp_pose'] = None
        userdata.objects_picked_count += 1
        return 'pick_succeeded'


class NavigateBackToKitchen(smach.State):
    """【真实导航】放置完成后返回厨房桌边，准备抓取下一件物品"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['returned', 'navigation_failed']
        )
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )

    def execute(self, userdata):
        rospy.loginfo("========== 返回厨房桌边 ==========")

        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("导航动作服务器未响应")
            return 'navigation_failed'

        goal = NavigateToLocationGoal()
        goal.target_location = "kitchen"

        rospy.loginfo("发送导航目标: kitchen（返回抓取位）")
        self.nav_client.send_goal(goal)

        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))

        if not finished:
            rospy.logerr("返回厨房导航超时")
            self.nav_client.cancel_goal()
            return 'navigation_failed'

        result = self.nav_client.get_result()

        if result and result.success:
            rospy.loginfo("✓ 已返回厨房桌边，准备抓取下一件")
            return 'returned'
        else:
            rospy.logwarn("返回厨房失败，尝试继续")
            return 'navigation_failed'


class NavigateToDest(smach.State):
    """【真实导航】完全复制自 navigate_to_dest.py"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['arrived', 'navigation_failed', 'fatal_error'],
            input_keys=['destination']
        )
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )

    def execute(self, userdata):
        rospy.loginfo("========== 导航到目的地 ==========")

        if 'destination' not in userdata:
            rospy.logerr("未指定目的地")
            return 'fatal_error'

        destination = userdata['destination']
        rospy.loginfo(f"目的地: {destination}")

        if not self.nav_client.wait_for_server(timeout=rospy.Duration(10.0)):
            rospy.logerr("导航动作服务器未响应")
            return 'navigation_failed'

        goal = NavigateToLocationGoal()
        goal.target_location = destination

        rospy.loginfo(f"发送导航目标: {destination}")
        self.nav_client.send_goal(goal)

        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(120.0))

        if not finished:
            rospy.logerr("导航超时")
            self.nav_client.cancel_goal()
            return 'navigation_failed'

        result = self.nav_client.get_result()

        if result and result.success:
            rospy.loginfo(f"✓ 成功到达 {destination}")
            return 'arrived'
        else:
            rospy.logwarn(f"✗ 导航到 {destination} 失败")
            return 'navigation_failed'


class PerceiveDest(smach.State):
    """[DRY RUN] 跳过柜子检测，返回默认放置姿态"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['perception_done', 'perception_failed', 'fatal_error'],
            input_keys=['destination', 'selected_object'],
            output_keys=['shelf_info', 'place_pose', 'target_layer']
        )

    def execute(self, userdata):
        rospy.loginfo("========== [DRY RUN] 感知目的地 ==========")

        destination = userdata.destination
        rospy.loginfo(f"[DRY RUN] 目的地: {destination}，跳过视觉感知（柜子检测/放置姿态计算）")

        userdata['shelf_info'] = None
        userdata['target_layer'] = 0

        default_pose = Pose()
        default_pose.orientation.w = 1.0
        default_pose.position.x = 0.6
        default_pose.position.z = 0.8

        userdata['place_pose'] = default_pose
        rospy.loginfo("[DRY RUN] ✓ 使用默认放置姿态 (x=0.6, z=0.8)")
        return 'perception_done'


class ExecutePlace(smach.State):
    """[DRY RUN] 打印放置阶段（含洗碗机门提示），直接返回 place_succeeded"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['place_succeeded', 'place_failed', 'fatal_error'],
            input_keys=['place_pose', 'selected_object', 'objects_placed_count', 'destination'],
            output_keys=['objects_placed_count']
        )

    def execute(self, userdata):
        rospy.loginfo("========== [DRY RUN] 执行放置 ==========")

        destination = userdata.destination

        # 规则书 Rule #4：洗碗机默认关闭，需通知 referee 开门
        if destination == 'dishwasher':
            rospy.logwarn("[DRY RUN][洗碗机门] 请 referee 打开洗碗机门（模拟 TTS 提示）")
            rospy.loginfo("[DRY RUN] TTS: Please open the dishwasher door.")

        phases = ['移动到放置预位', '打开夹爪', '降低机械臂', '释放物品']
        for phase in phases:
            rospy.loginfo(f"  [DRY RUN] 放置阶段: {phase}")

        rospy.loginfo("[DRY RUN] ✓ 放置成功（模拟）")

        userdata.objects_placed_count += 1
        return 'place_succeeded'


class ServeBreakfast(smach.State):
    """
    准备早餐状态（规则书 §5.2 Main Goal）
    _navigate_to() 和 _compute_placement_pose() 保留真实实现；
    _detect_item() / _pick_item() / _place_item() 替换为 print。
    """

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

        # 桌面摆放偏移
        self.item_spacing = rospy.get_param('~breakfast/item_spacing', 0.15)

        # 相邻物品配对偏移（规则书：spoon 紧靠 bowl，milk 在 cereal 左侧）
        self.pair_offsets = {
            'spoon':  [0.10, 0.0],
            'milk':   [-0.10, 0.0],
        }

        # 导航客户端（真实）
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location',
            NavigateToLocationAction
        )

    # ------------------------------------------------------------------
    # 真实实现：导航
    # ------------------------------------------------------------------
    def _navigate_to(self, location_name):
        """导航到指定地点，返回 True/False（完全复制自原版）"""
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

    # ------------------------------------------------------------------
    # 真实实现：放置姿态计算（纯数学，无 ROS 调用）
    # ------------------------------------------------------------------
    def _compute_placement_pose(self, item_name, index, prev_pose):
        """
        计算桌面放置位姿（完全复制自原版）：
        - 第一件物品放在桌子中央
        - 后续物品在 x 方向累加 item_spacing
        - spoon/milk 紧靠前一件物品（使用 pair_offsets）
        """
        pose = Pose()
        pose.orientation.w = 1.0

        base_x = 0.6
        base_y = 0.0
        base_z = 0.75

        if prev_pose is None:
            pose.position.x = base_x
            pose.position.y = base_y
            pose.position.z = base_z
        elif item_name in self.pair_offsets:
            dx, dy = self.pair_offsets[item_name]
            pose.position.x = prev_pose.position.x + dx
            pose.position.y = prev_pose.position.y + dy
            pose.position.z = base_z
        else:
            pose.position.x = base_x
            pose.position.y = base_y + index * self.item_spacing
            pose.position.z = base_z

        return pose

    # ------------------------------------------------------------------
    # [DRY RUN] 模拟：检测、抓取、放置
    # ------------------------------------------------------------------
    def _detect_item(self, item_name):
        """[DRY RUN] 跳过相机和检测服务，返回 MOCK_BREAKFAST_OBJECTS 中的模拟物品"""
        rospy.loginfo(f"  [DRY RUN] 模拟检测 {item_name}...")
        obj = MOCK_BREAKFAST_OBJECTS.get(
            item_name,
            MockDetectedObject(item_name)
        )
        rospy.loginfo(f"  [DRY RUN][感知公示] 发现 {obj.class_name} 置信度 {obj.confidence:.0%}")
        return obj

    def _pick_item(self, detected_obj):
        """[DRY RUN] 打印六个抓取阶段，直接返回 True"""
        rospy.loginfo(f"  [DRY RUN] 模拟抓取 {detected_obj.class_name}...")
        phases = ['接近目标', '计算抓取姿态', '移动到预抓取位', '张开夹爪', '执行抓取', '收回机械臂']
        for phase in phases:
            rospy.loginfo(f"    [DRY RUN] 抓取阶段: {phase}")
        rospy.loginfo(f"  [DRY RUN] ✓ 抓取 {detected_obj.class_name} 成功（模拟）")
        return True

    def _place_item(self, target_pose):
        """[DRY RUN] 打印四个放置阶段，直接返回 True"""
        rospy.loginfo(f"  [DRY RUN] 模拟放置到 "
                      f"({target_pose.position.x:.2f}, "
                      f"{target_pose.position.y:.2f}, "
                      f"{target_pose.position.z:.2f})...")
        phases = ['移动到放置预位', '打开夹爪', '降低机械臂', '释放物品']
        for phase in phases:
            rospy.loginfo(f"    [DRY RUN] 放置阶段: {phase}")
        rospy.loginfo(f"  [DRY RUN] ✓ 放置成功（模拟）")
        return True

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

            # 1. 导航到物品存放处（真实导航）
            rospy.loginfo(f"  Step 1: 导航到 {source}")
            if not self._navigate_to(source):
                rospy.logwarn(f"  导航失败，跳过 {item_name}")
                failed_items.append(item_name)
                continue

            # 2. 检测物品（模拟）
            rospy.loginfo(f"  Step 2: 检测 {item_name}")
            detected_obj = self._detect_item(item_name)
            if detected_obj is None:
                rospy.logwarn(f"  未检测到 {item_name}，跳过")
                failed_items.append(item_name)
                continue

            # 3. 抓取（模拟）
            rospy.loginfo(f"  Step 3: 抓取 {item_name}")
            if not self._pick_item(detected_obj):
                rospy.logwarn(f"  抓取 {item_name} 失败，跳过")
                failed_items.append(item_name)
                continue

            # 4. 导航到餐桌（真实导航）
            rospy.loginfo(f"  Step 4: 导航到 dining_table")
            if not self._navigate_to('dining_table'):
                rospy.logwarn(f"  导航到餐桌失败，跳过 {item_name}")
                failed_items.append(item_name)
                continue

            # 5. 计算放置位姿并放置（姿态计算真实，放置模拟）
            target_pose = self._compute_placement_pose(item_name, idx, prev_pose)
            rospy.loginfo(f"  Step 5: 放置 {item_name} 到 "
                          f"({target_pose.position.x:.2f}, {target_pose.position.y:.2f}, "
                          f"{target_pose.position.z:.2f})")
            if not self._place_item(target_pose):
                rospy.logwarn(f"  放置 {item_name} 失败")
                failed_items.append(item_name)
            prev_pose = target_pose

        if failed_items:
            rospy.logwarn(f"[ServeBreakfast] 以下物品处理失败: {failed_items}")
            if len(failed_items) == len(self.breakfast_items):
                return 'failed'

        rospy.loginfo("[ServeBreakfast] 早餐准备完成！")
        return 'breakfast_served'


class TaskCompleted(smach.State):
    """完全复制自 task_completed.py"""

    def __init__(self):
        smach.State.__init__(
            self,
            outcomes=['done'],
            input_keys=['objects_picked_count', 'objects_placed_count', 'failed_objects']
        )

    def execute(self, userdata):
        rospy.loginfo("=" * 60)
        rospy.loginfo(" " * 15 + "任务完成！")
        rospy.loginfo("=" * 60)

        # 正确代码
        picked = userdata.objects_picked_count
        placed = userdata.objects_placed_count
        failed = userdata.failed_objects

        rospy.loginfo(f"抓取物品数: {picked}")
        rospy.loginfo(f"放置物品数: {placed}")

        if len(failed) > 0:
            rospy.loginfo(f"失败物品数: {len(failed)}")
            rospy.loginfo(f"失败物品列表: {', '.join(failed)}")

        rospy.loginfo("=" * 60)

        return 'done'


# ============================================================
# 状态机构建（完全复制自 pick_place_task.py）
# ============================================================

def create_cleanup_loop():
    """创建内层清理循环状态机"""
    cleanup_sm = smach.StateMachine(
        outcomes=['all_objects_processed', 'cleanup_failed'],
        input_keys=['detected_objects', 'objects_to_pick', 'objects_picked_count', 'objects_placed_count'],
        output_keys=['objects_picked_count', 'objects_placed_count', 'failed_objects']
    )

    cleanup_sm.userdata.current_object_index = 0
    cleanup_sm.userdata.selected_object = None
    cleanup_sm.userdata.object_category = ''
    cleanup_sm.userdata.destination = ''
    cleanup_sm.userdata.grasp_pose = None
    cleanup_sm.userdata.place_pose = None
    cleanup_sm.userdata.shelf_info = None
    cleanup_sm.userdata.target_layer = 0
    cleanup_sm.userdata.failed_objects = []

    with cleanup_sm:
        smach.StateMachine.add(
            'SELECT_TARGET',
            SelectTarget(),
            transitions={
                'target_selected': 'EXECUTE_PICK',
                'no_more_objects': 'all_objects_processed',
                'failed': 'SELECT_TARGET'  # 分类失败跳过此物品，index 已递增，继续下一件
            },
            remapping={
                'objects_to_pick': 'objects_to_pick',
                'current_object_index': 'current_object_index',
                'selected_object': 'selected_object',
                'object_category': 'object_category',
                'destination': 'destination'
            }
        )

        smach.StateMachine.add(
            'EXECUTE_PICK',
            ExecutePick(),
            transitions={
                'pick_succeeded': 'NAVIGATE_TO_DEST',
                'pick_failed': 'SELECT_TARGET',
                'fatal_error': 'cleanup_failed'
            },
            remapping={
                'selected_object': 'selected_object',
                'object_category': 'object_category',
                'grasp_pose': 'grasp_pose',
                'objects_picked_count': 'objects_picked_count'
            }
        )

        smach.StateMachine.add(
            'NAVIGATE_TO_DEST',
            NavigateToDest(),
            transitions={
                'arrived': 'PERCEIVE_DEST',
                'navigation_failed': 'EXECUTE_PLACE',
                'fatal_error': 'cleanup_failed'
            },
            remapping={
                'destination': 'destination'
            }
        )

        smach.StateMachine.add(
            'PERCEIVE_DEST',
            PerceiveDest(),
            transitions={
                'perception_done': 'EXECUTE_PLACE',
                'perception_failed': 'EXECUTE_PLACE',
                'fatal_error': 'cleanup_failed'
            },
            remapping={
                'destination': 'destination',
                'selected_object': 'selected_object',
                'shelf_info': 'shelf_info',
                'place_pose': 'place_pose',
                'target_layer': 'target_layer'
            }
        )

        smach.StateMachine.add(
            'EXECUTE_PLACE',
            ExecutePlace(),
            transitions={
                'place_succeeded': 'NAVIGATE_BACK_TO_KITCHEN',  # 放完再回桌边
                'place_failed':    'NAVIGATE_BACK_TO_KITCHEN',  # 失败也要回去取下一件
                'fatal_error':     'cleanup_failed'
            },
            remapping={
                'place_pose': 'place_pose',
                'selected_object': 'selected_object',
                'objects_placed_count': 'objects_placed_count',
                'destination': 'destination'
            }
        )

        # 6. 放置完毕后返回厨房桌边，准备处理下一件物品
        smach.StateMachine.add(
            'NAVIGATE_BACK_TO_KITCHEN',
            NavigateBackToKitchen(),
            transitions={
                'returned':          'SELECT_TARGET',  # 正常返回，继续下一件
                'navigation_failed': 'SELECT_TARGET'   # 导航失败也尝试继续
            }
        )

    return cleanup_sm


def create_main_state_machine():
    """创建外层主状态机"""
    main_sm = smach.StateMachine(outcomes=['task_succeeded', 'task_failed'])

    main_sm.userdata.detected_objects = []
    main_sm.userdata.objects_to_pick = []
    main_sm.userdata.objects_picked_count = 0
    main_sm.userdata.objects_placed_count = 0
    main_sm.userdata.failed_objects = []

    with main_sm:
        smach.StateMachine.add(
            'INIT_SYSTEM',
            InitSystem(),
            transitions={
                'initialized': 'NAVIGATE_TO_KITCHEN',
                'init_failed': 'task_failed'
            }
        )

        smach.StateMachine.add(
            'NAVIGATE_TO_KITCHEN',
            NavigateToKitchen(),
            transitions={
                'arrived': 'ASSESS_SCENE',
                'navigation_failed': 'task_failed'
            }
        )

        smach.StateMachine.add(
            'ASSESS_SCENE',
            AssessScene(),
            transitions={
                'objects_detected': 'TABLE_CLEANUP_LOOP',
                'no_objects': 'SERVE_BREAKFAST',
                'perception_failed': 'task_failed'
            },
            remapping={
                'detected_objects': 'detected_objects',
                'objects_to_pick': 'objects_to_pick'
            }
        )

        smach.StateMachine.add(
            'TABLE_CLEANUP_LOOP',
            create_cleanup_loop(),
            transitions={
                'all_objects_processed': 'SERVE_BREAKFAST',
                'cleanup_failed': 'task_failed'
            },
            remapping={
                'detected_objects': 'detected_objects',
                'objects_to_pick': 'objects_to_pick',
                'objects_picked_count': 'objects_picked_count',
                'objects_placed_count': 'objects_placed_count',
                'failed_objects': 'failed_objects'
            }
        )

        smach.StateMachine.add(
            'SERVE_BREAKFAST',
            ServeBreakfast(),
            transitions={
                'breakfast_served': 'TASK_COMPLETED',
                'breakfast_skipped': 'TASK_COMPLETED',
                'failed': 'task_failed'
            }
        )

        smach.StateMachine.add(
            'TASK_COMPLETED',
            TaskCompleted(),
            transitions={
                'done': 'task_succeeded'
            },
            remapping={
                'objects_picked_count': 'objects_picked_count',
                'objects_placed_count': 'objects_placed_count',
                'failed_objects': 'failed_objects'
            }
        )

    return main_sm


def main():
    """主函数"""
    rospy.init_node('pick_place_dry_run')

    rospy.loginfo("=" * 60)
    rospy.loginfo(" " * 5 + "RoboCup@Home Pick and Place 任务 [DRY RUN]")
    rospy.loginfo(" " * 5 + "感知/抓取/放置 = 模拟 | 导航 = 真实")
    rospy.loginfo("=" * 60)

    sm = create_main_state_machine()

    sis = smach_ros.IntrospectionServer('pick_place_dry_run', sm, '/PICK_PLACE_TASK')
    sis.start()

    rospy.loginfo("SMACH状态机已启动")
    rospy.loginfo("可使用 rosrun smach_viewer smach_viewer.py 进行可视化")
    rospy.loginfo("=" * 60)

    rospy.loginfo("开始执行 Pick and Place 任务（Dry Run）...")
    outcome = sm.execute()

    rospy.loginfo("=" * 60)
    if outcome == 'task_succeeded':
        rospy.loginfo("✓ 任务成功完成！[DRY RUN]")
    else:
        rospy.logerr("✗ 任务失败 [DRY RUN]")
    rospy.loginfo("=" * 60)

    sis.stop()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("任务被中断")

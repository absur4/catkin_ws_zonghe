#!/usr/bin/env python3
import rospy
import math
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PoseStamped
from sensor_msgs.msg import LaserScan
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseActionResult
import actionlib
import numpy as np
from move_base_msgs.msg import MoveBaseActionGoal

class NavigationMonitor:

    def __init__(self):
        rospy.init_node('navigation_monitor', anonymous=True)
        
        # 参数（新增检测距离区间参数，方便后续调参）
        self.still_timeout = rospy.get_param('~still_timeout', 5.0)
        self.detect_min_dist = rospy.get_param('~detect_min_dist', 0.4)  # 40cm
        self.detect_max_dist = rospy.get_param('~detect_max_dist', 0.6)  # 60cm
        self.detect_width = rospy.get_param('~detect_width', 0.6)        # 60cm
        self.move_distance = rospy.get_param('~move_distance', 0.1)      # 10cm
        self.corridor_speed = rospy.get_param('~corridor_speed', 0.1)    # 0.1m/s
        self.angular_speed = rospy.get_param('~angular_speed', 0.4)      # rad/s
        self.laser_x_offset = rospy.get_param('~laser_x_offset', 0.30)   # 雷达到 base_link 的 x 偏移
        self.corner_min_range = rospy.get_param('~corner_min_range', 0.2)
        self.corner_max_range = rospy.get_param('~corner_max_range', 2.5)
        self.corner_angle_deg = rospy.get_param('~corner_angle_deg', 15.0)
        self.corner_step = rospy.get_param('~corner_step', 5)
        self.mid_start_threshold = rospy.get_param('~mid_start_threshold', 0.6)
        self.mid_goal_threshold = rospy.get_param('~mid_goal_threshold', 0.6)
        self.base_frame = rospy.get_param('~base_frame', 'base_footprint')
        
        # 角度变化阈值（弧度）：超过该值判定为角度有变化
        self.angle_threshold = rospy.get_param('~angle_threshold', 0.02)  # 约1.15度
        
        # 状态变量初始化
        self.last_position = None
        self.last_orientation = None  # 新增：保存上一时刻角度
        self.last_movement_time = rospy.Time.now()
        self.is_still = False
        self.still_start_time = rospy.Time.now()
        self.recovery_active = False
        self.back_count = 0
        self.corridor_adjusted = False
        
        # 保存当前的导航目标 & 新增：导航完成状态标记
        self.current_goal = None
        self.goal_sent = False
        self.nav_complete = True  # 初始无目标，标记为导航完成
        
        # TF监听器
        self.tf_listener = tf.TransformListener()
        
        # 订阅话题
        self.odom_sub = rospy.Subscriber('/odometry/filtered', Odometry, self.odom_callback)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # 发布速度命令
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # MoveBase action client
        self.move_base_client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        rospy.loginfo("等待move_base action服务器...")
        self.move_base_client.wait_for_server()
        
        # 订阅move_base的目标 & 新增：订阅move_base导航结果
        self.goal_sub = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        # 新增监听 move_base Action 客户端下发的目标

        self.move_base_result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.move_base_result_callback)
        
        # 激光雷达数据
        self.latest_scan = None
        self.midpoints = self._load_midpoints()
        
        # 定时器
        self.check_timer = rospy.Timer(rospy.Duration(0.5), self.check_still_status)
        
        rospy.loginfo("导航监控节点已启动")

        # 在 __init__ 最后加入这几行
        self.action_goal_sub = rospy.Subscriber(
            '/move_base/goal',          # move_base action server 接收的目标
            MoveBaseActionGoal,
            self.action_goal_callback
)

    def _load_midpoints(self):
        locations = rospy.get_param('/navigation/locations', {})
        midpoints = {}
        for name in ['mid_1_1', 'mid_1_2', 'mid_2_1', 'mid_2_2']:
            if name in locations:
                midpoints[name] = locations[name]
        if len(midpoints) < 4:
            rospy.logwarn("未完整加载 mid 点坐标，走廊居中逻辑可能不会触发")
        return midpoints

    def _reset_back_counter(self, reason):
        self.back_count = 0
        self.corridor_adjusted = False
        rospy.logdebug(f"重置后退计数: {reason}")

    def _get_robot_pose_map(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform('map', self.base_frame, rospy.Time(0))
            return trans  # (x, y, z)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            return None

    def _is_mid_segment(self):
        if len(self.midpoints) < 4 or self.current_goal is None:
            return False

        pose = self._get_robot_pose_map()
        if pose is None:
            return False

        gx = self.current_goal.pose.position.x
        gy = self.current_goal.pose.position.y

        m11 = self.midpoints.get('mid_1_1')
        m12 = self.midpoints.get('mid_1_2')
        m21 = self.midpoints.get('mid_2_1')
        m22 = self.midpoints.get('mid_2_2')
        if not all([m11, m12, m21, m22]):
            return False

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        robot_xy = (pose[0], pose[1])

        goal_is_mid_12 = dist((gx, gy), (m12['x'], m12['y'])) <= self.mid_goal_threshold
        goal_is_mid_22 = dist((gx, gy), (m22['x'], m22['y'])) <= self.mid_goal_threshold
        near_mid_11 = dist(robot_xy, (m11['x'], m11['y'])) <= self.mid_start_threshold
        near_mid_21 = dist(robot_xy, (m21['x'], m21['y'])) <= self.mid_start_threshold

        return (goal_is_mid_12 and near_mid_11) or (goal_is_mid_22 and near_mid_21)
    
    def goal_callback(self, msg):
        """保存最新的导航目标（仅在正常导航时生效）"""
        self.current_goal = msg
        self.goal_sent = True
        self.is_still = False  # 新目标下达，重置静止状态
        self.nav_complete = False  # 新目标发布，标记为导航未完成
        self._reset_back_counter("新目标")
        rospy.loginfo(f"保存新的导航目标: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
    
    def move_base_result_callback(self, msg):
        """监听move_base导航结果，更新导航完成状态"""
        # MoveBaseResult状态码：3=成功，其他=失败/取消
        if msg.status.status == 3:
            rospy.loginfo("导航目标已完成，停止监测静止状态")
            self.nav_complete = True
            self.goal_sent = False  # 导航完成，重置目标发送状态
            self.is_still = False   # 重置静止状态
            self.recovery_active = False  # 重置自救状态
            self._reset_back_counter("导航完成")
    
    def odom_callback(self, msg):
            """里程计回调，使用实际速度精准检测机器人是否在移动"""
            current_position = msg.pose.pose.position
            
            # 直接读取里程计自带的线速度和角速度（绝对值）
            linear_v = abs(msg.twist.twist.linear.x)
            linear_vy = abs(msg.twist.twist.linear.y)  # 考虑全向轮底盘可能有y轴速度
            angular_v = abs(msg.twist.twist.angular.z)
            
            if (self.goal_sent and not self.recovery_active and not self.nav_complete):
                
                # 判断是否完全静止：线速度 < 0.02m/s 且 角速度 < 0.05rad/s
                is_pos_still = (linear_v < 0.02) and (linear_vy < 0.02)
                is_angle_still = angular_v < 0.05
                is_fully_still = is_pos_still and is_angle_still
                
                if is_fully_still:
                    if not self.is_still:
                        # 开始静止计时
                        self.is_still = True
                        self.still_start_time = rospy.Time.now()
                        rospy.logdebug(f"机器人速度趋近于零，开始计时 (v:{linear_v:.3f}, w:{angular_v:.3f})")
                else:
                    # 速度恢复，判定为移动，重置所有静止状态
                    self.is_still = False
                    self.recovery_active = False
            
            # 仍然更新历史位置，以防你其他地方还需要用
            self.last_position = current_position

    def scan_callback(self, msg):
        """激光雷达数据回调，保存最新扫描数据"""
        self.latest_scan = msg
    
    def check_front_obstacle(self):
        """
        检测 base_link 坐标系下：
        x ∈ [0.35 , 0.50]
        y ∈ [-0.30 , +0.30]
        """

        if self.latest_scan is None:
            rospy.logwarn("没有激光雷达数据")
            return True

        # ===== 你的矩形参数 =====
        x_min = 0.35
        x_max = 0.50
        y_min = -0.30
        y_max = 0.30

        angle = self.latest_scan.angle_min

        for r in self.latest_scan.ranges:

            if np.isinf(r) or np.isnan(r):
                angle += self.latest_scan.angle_increment
                continue

            # 1️⃣ 转换为雷达坐标系下的点
            x_laser = r * math.cos(angle)
            y_laser = r * math.sin(angle)

            # 2️⃣ 转换到 base_link 坐标系
            # 雷达在 base_link 的 x = 0.30
            x_base = x_laser + self.laser_x_offset
            y_base = y_laser

            # 3️⃣ 判断是否在矩形内
            if (x_min <= x_base <= x_max) and (y_min <= y_base <= y_max):
                rospy.loginfo(
                    f"检测到障碍物: x={x_base:.2f}, y={y_base:.2f}"
                )
                return True

            angle += self.latest_scan.angle_increment

        return False
    def move_robot(self, distance):
        """精准控制机器人前进/后退指定距离（单位：米）"""
        rospy.loginfo(f"执行自救移动: {'前进' if distance>0 else '后退'} {abs(distance):.2f}米")
        
        cmd = Twist()
        # 控制移动速度（前进慢一点，后退稍快，避免打滑）
        cmd.linear.x = 0.05 if distance > 0 else -0.08
        cmd.angular.z = 0.0  # 纯直线移动
        
        # 计算需要移动的时间（距离/速度）
        move_duration = abs(distance / cmd.linear.x)
        start_time = rospy.Time.now()
        rate = rospy.Rate(20)  # 20Hz发布速度指令
        
        # 持续发布速度直到达到指定时间
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < move_duration:
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
        
        # 停止机器人
        cmd.linear.x = 0.0
        self.cmd_vel_pub.publish(cmd)
        rospy.loginfo("自救移动完成")

    def _publish_cmd_for_duration(self, linear_x, angular_z, duration):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        rate = rospy.Rate(20)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < duration:
            self.cmd_vel_pub.publish(cmd)
            rate.sleep()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_vel_pub.publish(cmd)

    def _rotate_in_place(self, angle):
        if abs(angle) < 1e-3:
            return
        angular = self.angular_speed if angle > 0 else -self.angular_speed
        duration = abs(angle / angular)
        rospy.loginfo(f"原地旋转 {angle:.2f} rad")
        self._publish_cmd_for_duration(0.0, angular, duration)

    def _drive_forward(self, distance, speed):
        if distance <= 0:
            return
        duration = distance / speed
        rospy.loginfo(f"前进 {distance:.2f} m, 速度 {speed:.2f} m/s")
        self._publish_cmd_for_duration(speed, 0.0, duration)

    def _normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _detect_right_angle_corners(self):
        if self.latest_scan is None:
            rospy.logwarn("没有激光雷达数据")
            return []

        points = []
        angle = self.latest_scan.angle_min
        for i, r in enumerate(self.latest_scan.ranges):
            if np.isinf(r) or np.isnan(r):
                angle += self.latest_scan.angle_increment
                continue
            if r < self.corner_min_range or r > self.corner_max_range:
                angle += self.latest_scan.angle_increment
                continue
            x_laser = r * math.cos(angle)
            y_laser = r * math.sin(angle)
            x_base = x_laser + self.laser_x_offset
            y_base = y_laser
            points.append((i, x_base, y_base))
            angle += self.latest_scan.angle_increment

        corners = []
        step = max(1, int(self.corner_step))
        max_angle = math.radians(self.corner_angle_deg)

        for idx in range(step, len(points) - step):
            i0, x0, y0 = points[idx]
            i1, x1, y1 = points[idx - step]
            i2, x2, y2 = points[idx + step]
            v1 = (x1 - x0, y1 - y0)
            v2 = (x2 - x0, y2 - y0)
            norm1 = math.hypot(v1[0], v1[1])
            norm2 = math.hypot(v2[0], v2[1])
            if norm1 < 1e-3 or norm2 < 1e-3:
                continue
            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            if abs(angle - math.pi / 2) <= max_angle:
                corners.append((i0, x0, y0, v1, v2))

        return corners

    def _select_symmetric_corners(self, corners):
        left = None
        right = None
        for i0, x0, y0, v1, v2 in corners:
            if y0 >= 0:
                dist = math.hypot(x0, y0)
                if left is None or dist < left['dist']:
                    left = {'x': x0, 'y': y0, 'v1': v1, 'v2': v2, 'dist': dist}
            else:
                dist = math.hypot(x0, y0)
                if right is None or dist < right['dist']:
                    right = {'x': x0, 'y': y0, 'v1': v1, 'v2': v2, 'dist': dist}
        if left and right:
            return left, right
        return None, None

    def _edge_angle_parallel_x(self, v1, v2):
        def score(vec):
            dx, dy = vec
            if abs(dx) < 1e-3:
                return float('inf')
            return abs(dy / dx)

        edge = v1 if score(v1) <= score(v2) else v2
        angle = math.atan2(edge[1], edge[0])
        if angle > math.pi / 2:
            angle -= math.pi
        if angle < -math.pi / 2:
            angle += math.pi
        return angle

    def _compute_corridor_angle(self, left, right):
        angle_l = self._edge_angle_parallel_x(left['v1'], left['v2'])
        angle_r = self._edge_angle_parallel_x(right['v1'], right['v2'])
        angle = self._normalize_angle((angle_l + angle_r) / 2.0)
        return angle

    def _perform_corridor_centering(self):
        corners = self._detect_right_angle_corners()
        left, right = self._select_symmetric_corners(corners)
        if left is None or right is None:
            rospy.logwarn("未检测到对称直角点，跳过走廊居中")
            return False

        corridor_angle = self._compute_corridor_angle(left, right)
        x_center = (left['x'] + right['x']) / 2.0
        y_center = (left['y'] + right['y']) / 2.0

        # 将中心点投影到走廊对齐坐标系
        y_center_aligned = -math.sin(corridor_angle) * x_center + math.cos(corridor_angle) * y_center

        rospy.loginfo(f"走廊角度: {corridor_angle:.2f} rad, 中心偏移 y: {y_center_aligned:.2f} m")

        # 1) 先对齐走廊方向
        self._rotate_in_place(-corridor_angle)

        # 2) 侧移到走廊中线（通过旋转+前进实现）
        if y_center_aligned > 0:
            self._rotate_in_place(math.pi / 2)
            self._drive_forward(abs(y_center_aligned), self.corridor_speed)
            self._rotate_in_place(-math.pi / 2)
        elif y_center_aligned < 0:
            self._rotate_in_place(-math.pi / 2)
            self._drive_forward(abs(y_center_aligned), self.corridor_speed)
            self._rotate_in_place(math.pi / 2)
        else:
            rospy.loginfo("已在走廊中线附近，无需侧移")

        return True
    
    def check_still_status(self, event):
        """定时检查静止状态，仅在正常导航时触发自救"""
        # 核心修改：增加导航未完成的判断（self.nav_complete is False）
        if not self.goal_sent or self.recovery_active or not self.is_still or self.nav_complete:
            return
        
        # 计算静止时长
        still_duration = (rospy.Time.now() - self.still_start_time).to_sec()
        
        # 静止超5秒，触发自救
        if still_duration >= self.still_timeout:
            rospy.logwarn(f"机器人完全静止超过{self.still_timeout}秒，触发自救行为")
            self.perform_recovery()
    
    def perform_recovery(self):
        """执行完整的自救流程：取消当前目标→检测障碍物→移动→恢复导航"""
        self.recovery_active = True  # 标记为自救中，避免重复触发
        
        # 校验是否有保存的导航目标
        if self.current_goal is None:
            rospy.logwarn("无保存的导航目标，无法恢复导航")
            self.recovery_active = False
            self.is_still = False
            return
        
        # 1. 取消当前的move_base导航目标
        rospy.loginfo(f"取消当前导航目标: ({self.current_goal.pose.position.x:.2f}, {self.current_goal.pose.position.y:.2f})")
        self.move_base_client.cancel_all_goals()
        rospy.sleep(0.5)  # 等待取消生效
        
        # 2. 检测前方障碍物（40~60cm区域）
        obstacle_detected = self.check_front_obstacle()
        
        # 3. 根据障碍物情况执行移动
        if not obstacle_detected:
            self.move_robot(self.move_distance)  # 无障碍物：前进10cm
            self._reset_back_counter("前进自救")
        else:
            self.move_robot(-self.move_distance) # 有障碍物：后退10cm
            self.back_count += 1
            rospy.loginfo(f"后退次数: {self.back_count}")

            if self.back_count >= 2 and self._is_mid_segment() and not self.corridor_adjusted:
                rospy.logwarn("检测到第二次后退，尝试走廊居中处理")
                if self._perform_corridor_centering():
                    self.corridor_adjusted = True
        
        # 4. 重新发送原导航目标，恢复正常导航
        rospy.sleep(0.5)  # 等待机器人稳定
        rospy.loginfo(f"重新发送导航目标: ({self.current_goal.pose.position.x:.2f}, {self.current_goal.pose.position.y:.2f})")
        
        # 构建MoveBaseGoal并发送
        goal = MoveBaseGoal()
        goal.target_pose.header = self.current_goal.header
        goal.target_pose.header.stamp = rospy.Time.now()  # 更新时间戳，避免过期
        goal.target_pose.pose = self.current_goal.pose
        
        self.move_base_client.send_goal(goal)
        
        # 5. 重置状态，等待下一次检测
        self.recovery_active = False
        self.is_still = False
        rospy.loginfo("自救完成，已恢复正常导航")

    def action_goal_callback(self, msg):
        extracted_goal = PoseStamped()
        extracted_goal.header = msg.goal.target_pose.header
        extracted_goal.pose = msg.goal.target_pose.pose
        self.current_goal = extracted_goal
        self.goal_sent = True
        self.is_still = False
        self.nav_complete = False
        self._reset_back_counter("Action 新目标")
        rospy.loginfo(f"[Monitor] 通过 Action 收到新目标: ({extracted_goal.pose.position.x:.2f}, {extracted_goal.pose.position.y:.2f})")
if __name__ == '__main__':
    try:
        monitor = NavigationMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("导航监控节点已退出")

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
        
        # 角度变化阈值（弧度）：超过该值判定为角度有变化
        self.angle_threshold = rospy.get_param('~angle_threshold', 0.02)  # 约1.15度
        
        # 状态变量初始化
        self.last_position = None
        self.last_orientation = None  # 新增：保存上一时刻角度
        self.last_movement_time = rospy.Time.now()
        self.is_still = False
        self.still_start_time = rospy.Time.now()
        self.recovery_active = False
        
        # 保存当前的导航目标 & 新增：导航完成状态标记
        self.current_goal = None
        self.goal_sent = False
        self.nav_complete = True  # 初始无目标，标记为导航完成
        
        # TF监听器
        self.tf_listener = tf.TransformListener()
        
        # 订阅话题
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
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
        
        # 定时器
        self.check_timer = rospy.Timer(rospy.Duration(0.5), self.check_still_status)
        
        rospy.loginfo("导航监控节点已启动")

        # 在 __init__ 最后加入这几行
        self.action_goal_sub = rospy.Subscriber(
            '/move_base/goal',          # move_base action server 接收的目标
            MoveBaseActionGoal,
            self.action_goal_callback
)
    
    def goal_callback(self, msg):
        """保存最新的导航目标（仅在正常导航时生效）"""
        self.current_goal = msg
        self.goal_sent = True
        self.is_still = False  # 新目标下达，重置静止状态
        self.nav_complete = False  # 新目标发布，标记为导航未完成
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
            x_base = x_laser + 0.30
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
        else:
            self.move_robot(-self.move_distance) # 有障碍物：后退10cm
        
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
        rospy.loginfo(f"[Monitor] 通过 Action 收到新目标: ({extracted_goal.pose.position.x:.2f}, {extracted_goal.pose.position.y:.2f})")
if __name__ == '__main__':
    try:
        monitor = NavigationMonitor()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("导航监控节点已退出")
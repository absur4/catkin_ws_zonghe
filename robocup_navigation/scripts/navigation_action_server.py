#!/usr/bin/env python3
"""
导航动作服务器 (RViz兼容等待模式)
提供位置名称到坐标的转换，完全兼容底层的自救监控节点
"""
import rospy
import actionlib
import yaml
import os
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationResult, NavigateToLocationFeedback
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
import tf.transformations as tf_trans

class NavigationActionServer:
    def __init__(self):
        rospy.init_node('navigation_action_server')

        # 加载预定义位置
        locations_file = rospy.get_param('~locations_file', '')
        if not locations_file:
            locations_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'config', 'locations.yaml'
            )

        self.locations = {}
        self.default_timeout = rospy.get_param('~timeout', 180.0) # 稍微延长超时时间，给自救留出时间
        try:
            if os.path.exists(locations_file):
                with open(locations_file, 'r') as f:
                    config = yaml.safe_load(f)
                    self.locations = config.get('locations', {})
                    rospy.loginfo(f"已加载 {len(self.locations)} 个预定义位置")
        except Exception as e:
            rospy.logerr(f"加载位置配置失败: {e}")

        # 核心修改 1：像 RViz 一样发布简单目标，从而完美触发你的 monitor 节点
        self.goal_pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
        
        # 核心修改 2：监听底层最终结果
        self.result_sub = rospy.Subscriber('/move_base/result', MoveBaseActionResult, self.result_cb)
        self.current_nav_status = None

        # 创建导航动作服务器 (对接 SMACH)
        self.server = actionlib.SimpleActionServer(
            '/navigate_to_location',
            NavigateToLocationAction,
            execute_cb=self.execute_navigation,
            auto_start=False
        )
        self.server.start()
        rospy.loginfo("✓ 导航动作服务器已启动 (RViz自救兼容模式)")

    def result_cb(self, msg):
        """记录底层 move_base 的状态：3=成功，4=彻底失败，2=被取消(自救触发时)"""
        self.current_nav_status = msg.status.status

    def execute_navigation(self, goal):
        """执行导航任务"""
        location_name = goal.target_location
        rospy.loginfo("=" * 50)
        rospy.loginfo(f"开始导航到: {location_name}")
        
        result = NavigateToLocationResult()
        feedback = NavigateToLocationFeedback()
        self.current_nav_status = None # 每次新任务重置状态

        # 构建目标位姿
        target_pose = PoseStamped()
        target_pose.header.frame_id = "map"
        target_pose.header.stamp = rospy.Time.now()
        
        if location_name in self.locations:
            loc = self.locations[location_name]
            target_pose.pose.position.x = loc['x']
            target_pose.pose.position.y = loc['y']
            target_pose.pose.position.z = 0.0

            quat = tf_trans.quaternion_from_euler(0, 0, loc['theta'])
            target_pose.pose.orientation.x = quat[0]
            target_pose.pose.orientation.y = quat[1]
            target_pose.pose.orientation.z = quat[2]
            target_pose.pose.orientation.w = quat[3]
        else:
            rospy.logerr(f"未知位置: {location_name}")
            result.success = False
            self.server.set_aborted(result)
            return

        # 像 RViz 一样发布目标！此时你的 monitor 会接管监控
        self.goal_pub.publish(target_pose)
        
        feedback.current_status = "导航中..."
        self.server.publish_feedback(feedback)

        # 监控进度
        rate = rospy.Rate(5)  # 5Hz
        start_time = rospy.Time.now()
        timeout = rospy.Duration(self.default_timeout)

        while not rospy.is_shutdown():
            # 1. 检查超时
            if rospy.Time.now() - start_time > timeout:
                rospy.logwarn(f"导航到 {location_name} 超时")
                result.success = False
                self.server.set_aborted(result)
                return

            # 2. 检查上层是否取消
            if self.server.is_preempt_requested():
                rospy.loginfo("SMACH 任务请求取消导航")
                result.success = False
                self.server.set_preempted(result)
                return

            # 3. 检查底层结果
            # 注意：如果状态是 2 (PREEMPTED)，说明你的 monitor 正在自救（它取消了目标），我们在这里选择【忽略并耐心等待】！
            if self.current_nav_status == 3: # SUCCEEDED
                rospy.loginfo(f"✓ 成功到达 {location_name}")
                result.success = True
                result.final_pose = target_pose.pose
                self.server.set_succeeded(result)
                return
            elif self.current_nav_status == 4: # ABORTED
                # 只有 move_base 彻底放弃（比如自救完还是去不了），才向上层报错
                rospy.logwarn(f"✗ 导航到 {location_name} 彻底失败 (已尝试自救)")
                result.success = False
                self.server.set_aborted(result)
                return

            rate.sleep()

if __name__ == '__main__':
    server = NavigationActionServer()
    rospy.spin()
#!/usr/bin/env python3
"""
debug_06_navigation.py — navigate_to_location 动作调试脚本

先决条件：
  /navigate_to_location 动作服务器 + 底盘 + AMCL

用法：
  rosrun robocup_debug debug_06_navigation.py                   # 列出所有位置
  rosrun robocup_debug debug_06_navigation.py --to kitchen
  rosrun robocup_debug debug_06_navigation.py --to dishwasher

功能：
  - 无参数运行时：从 locations.yaml 读取并打印所有已知位置名
  - --to <name>：导航到指定地点，实时打印 distance_remaining + current_status
  - 到达后打印 final_pose
"""
import sys
import os
import argparse
import rospy
import actionlib
import yaml
from robocup_msgs.msg import NavigateToLocationAction, NavigateToLocationGoal

LOCATIONS_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'robocup_navigation', 'config', 'locations.yaml'
)


def load_locations():
    """从 locations.yaml 加载所有位置"""
    yaml_path = os.path.normpath(LOCATIONS_YAML)
    if not os.path.exists(yaml_path):
        # 尝试从 ROS 包路径查找
        try:
            import rospkg
            rp = rospkg.RosPack()
            nav_path = rp.get_path('robocup_navigation')
            yaml_path = os.path.join(nav_path, 'config', 'locations.yaml')
        except Exception:
            rospy.logwarn(f"locations.yaml 未找到，位置列表不可用")
            return {}
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('locations', {})
    except Exception as e:
        rospy.logwarn(f"读取 locations.yaml 失败: {e}")
        return {}


def print_locations(locations):
    rospy.loginfo("")
    rospy.loginfo("=" * 55)
    rospy.loginfo("  已知位置列表（来自 locations.yaml）")
    rospy.loginfo("=" * 55)
    if not locations:
        rospy.logwarn("（未找到位置定义）")
    else:
        rospy.loginfo(f"{'位置名':<20} {'x':>6} {'y':>6} {'θ':>6}  描述")
        rospy.loginfo('-' * 55)
        for name, info in locations.items():
            x = info.get('x', 0.0)
            y = info.get('y', 0.0)
            theta = info.get('theta', 0.0)
            desc = info.get('description', '')
            rospy.loginfo(f"{name:<20} {x:>6.2f} {y:>6.2f} {theta:>6.2f}  {desc}")
    rospy.loginfo("=" * 55)
    rospy.loginfo("")
    rospy.loginfo("使用示例：")
    rospy.loginfo("  rosrun robocup_debug debug_06_navigation.py --to kitchen")


class NavigationDebugger:
    def __init__(self, target):
        self.target = target
        self.nav_client = actionlib.SimpleActionClient(
            '/navigate_to_location', NavigateToLocationAction
        )

    def _feedback_cb(self, feedback):
        rospy.loginfo(
            f"  [FEEDBACK] dist_remaining={feedback.distance_remaining:.3f}m  "
            f"status={feedback.current_status}"
        )

    def run(self):
        rospy.loginfo(f"导航目标: {self.target}")
        rospy.loginfo("等待 /navigate_to_location 动作服务器（超时 15s）...")
        if not self.nav_client.wait_for_server(timeout=rospy.Duration(15.0)):
            rospy.logerr("/navigate_to_location 动作服务器未响应")
            rospy.logerr("请先运行：rosrun robocup_navigation navigation_action_server.py")
            sys.exit(1)
        rospy.loginfo("/navigate_to_location 服务器已就绪 ✓")

        goal = NavigateToLocationGoal()
        goal.target_location = self.target

        rospy.loginfo(f"\n发送导航目标: {self.target}")
        rospy.loginfo("--- feedback 流 ---")
        self.nav_client.send_goal(goal, feedback_cb=self._feedback_cb)

        finished = self.nav_client.wait_for_result(timeout=rospy.Duration(180.0))

        rospy.loginfo("")
        rospy.loginfo("=" * 60)
        rospy.loginfo("  NavigateToLocation 最终结果")
        rospy.loginfo("=" * 60)

        if not finished:
            rospy.logerr("导航超时（180s）")
            self.nav_client.cancel_goal()
            return

        result = self.nav_client.get_result()
        if result is None:
            rospy.logerr("未收到结果")
            return

        rospy.loginfo(f"success : {result.success}")
        rospy.loginfo(f"message : {result.message}")
        if result.success:
            p = result.final_pose.position
            q = result.final_pose.orientation
            rospy.loginfo(
                f"final_pose: pos=({p.x:.4f}, {p.y:.4f}, {p.z:.4f})  "
                f"quat=({q.x:.3f}, {q.y:.3f}, {q.z:.3f}, {q.w:.3f})"
            )
        rospy.loginfo("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='debug_06_navigation: 导航动作调试')
    parser.add_argument('--to', dest='target', default=None,
                        help='目标位置名称（不指定则列出所有位置）')
    args = parser.parse_args(rospy.myargv(argv=sys.argv)[1:])

    rospy.init_node('debug_06_navigation', anonymous=True)
    rospy.loginfo("== debug_06_navigation 启动 ==")

    locations = load_locations()

    if args.target is None:
        print_locations(locations)
        rospy.loginfo("== debug_06_navigation 完成（列表模式）==")
        return

    if locations and args.target not in locations:
        rospy.logwarn(f"位置 '{args.target}' 不在 locations.yaml 中，仍会尝试导航")

    debugger = NavigationDebugger(args.target)
    debugger.run()
    rospy.loginfo("== debug_06_navigation 完成 ==")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

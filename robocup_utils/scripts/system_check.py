#!/usr/bin/env python3
"""
系统检查工具
检查所有ROS服务和动作服务器是否正常运行
"""
import rospy
import sys
from colorama import Fore, Style, init

# 初始化colorama（彩色输出）
init(autoreset=True)


class SystemChecker:
    def __init__(self):
        rospy.init_node('system_checker', anonymous=True)

        # 需要检查的服务列表
        self.required_services = [
            '/detect_objects',
            '/detect_shelf',
            '/compute_grasp_pose',
            '/compute_place_pose',
            '/classify_object',
        ]

        # 需要检查的动作服务器
        self.required_actions = [
            '/pick_object',
            '/place_object',
            '/navigate_to_location',
        ]

        # 需要检查的话题
        self.required_topics = [
            '/camera/color/image_raw',
            '/camera/aligned_depth_to_color/image_raw',
            '/joint_states',
            '/tf',
        ]

    def check_services(self):
        """检查ROS服务"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}检查 ROS 服务...")
        print(f"{Fore.CYAN}{'='*60}")

        all_services = rospy.get_published_topics()
        available_services = rospy.get_service_list()

        success_count = 0
        for service in self.required_services:
            try:
                rospy.wait_for_service(service, timeout=2.0)
                print(f"{Fore.GREEN}✓ {service}")
                success_count += 1
            except rospy.ROSException:
                print(f"{Fore.RED}✗ {service} - 不可用")

        return success_count, len(self.required_services)

    def check_actions(self):
        """检查动作服务器"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}检查动作服务器...")
        print(f"{Fore.CYAN}{'='*60}")

        all_topics = rospy.get_published_topics()
        topic_names = [topic[0] for topic in all_topics]

        success_count = 0
        for action in self.required_actions:
            # 检查action的feedback话题是否存在
            feedback_topic = f"{action}/feedback"
            if feedback_topic in topic_names:
                print(f"{Fore.GREEN}✓ {action}")
                success_count += 1
            else:
                print(f"{Fore.RED}✗ {action} - 不可用")

        return success_count, len(self.required_actions)

    def check_topics(self):
        """检查话题"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}检查话题...")
        print(f"{Fore.CYAN}{'='*60}")

        all_topics = rospy.get_published_topics()
        topic_names = [topic[0] for topic in all_topics]

        success_count = 0
        for topic in self.required_topics:
            if topic in topic_names:
                print(f"{Fore.GREEN}✓ {topic}")
                success_count += 1
            else:
                print(f"{Fore.YELLOW}⚠ {topic} - 未发布")

        return success_count, len(self.required_topics)

    def check_nodes(self):
        """检查关键节点"""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}检查关键节点...")
        print(f"{Fore.CYAN}{'='*60}")

        nodes = rospy.get_node_names()

        key_nodes = [
            'object_detector_server',
            'object_classifier',
            'grasp_pose_computer',
            'place_pose_computer',
            'pick_action_server',
            'place_action_server',
            'navigation_action_server',
        ]

        success_count = 0
        for node in key_nodes:
            found = any(node in n for n in nodes)
            if found:
                print(f"{Fore.GREEN}✓ {node}")
                success_count += 1
            else:
                print(f"{Fore.RED}✗ {node} - 未运行")

        return success_count, len(key_nodes)

    def run_check(self):
        """运行完整检查"""
        print(f"\n{Fore.YELLOW}{'*'*60}")
        print(f"{Fore.YELLOW}RoboCup@Home 系统检查工具")
        print(f"{Fore.YELLOW}{'*'*60}")

        # 检查各个组件
        service_ok, service_total = self.check_services()
        action_ok, action_total = self.check_actions()
        topic_ok, topic_total = self.check_topics()
        node_ok, node_total = self.check_nodes()

        # 总结
        total_ok = service_ok + action_ok + topic_ok + node_ok
        total_items = service_total + action_total + topic_total + node_total

        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}检查总结")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"服务:     {service_ok}/{service_total}")
        print(f"动作服务器: {action_ok}/{action_total}")
        print(f"话题:     {topic_ok}/{topic_total}")
        print(f"节点:     {node_ok}/{node_total}")
        print(f"{Fore.CYAN}{'='*60}")

        if total_ok == total_items:
            print(f"{Fore.GREEN}✓ 系统检查通过！所有组件正常运行。")
            return 0
        elif total_ok >= total_items * 0.7:
            print(f"{Fore.YELLOW}⚠ 系统部分就绪（{total_ok}/{total_items}），部分组件未运行。")
            return 1
        else:
            print(f"{Fore.RED}✗ 系统未就绪（{total_ok}/{total_items}），多个组件未运行。")
            print(f"\n{Fore.YELLOW}请先启动系统：")
            print(f"  roslaunch robocup_executive system_bringup.launch")
            return 2


if __name__ == '__main__':
    try:
        checker = SystemChecker()
        exit_code = checker.run_check()
        sys.exit(exit_code)
    except rospy.ROSInterruptException:
        print(f"{Fore.RED}系统检查被中断")
        sys.exit(3)
    except Exception as e:
        print(f"{Fore.RED}检查失败: {e}")
        sys.exit(4)

#!/usr/bin/env python3
"""
Pick and Place任务主状态机
完整实现端到端任务流程

双层SMACH结构：
- 外层：INIT → NAV_KITCHEN → ASSESS → CLEANUP_LOOP → BREAKFAST → COMPLETED
- 内层：SELECT → PICK → NAV_DEST → PERCEIVE_DEST → PLACE → NAV_BACK_TO_KITCHEN → (循环)
"""
import rospy
import smach
import smach_ros

# 导入所有状态类
from robocup_executive.states import (
    InitSystem,
    NavigateToKitchen,
    AssessScene,
    SelectTarget,
    ExecutePick,
    NavigateToDest,
    NavigateBackToKitchen,
    PerceiveDest,
    ExecutePlace,
    ServeBreakfast,
    TaskCompleted
)


def create_cleanup_loop():
    """创建内层清理循环状态机"""
    cleanup_sm = smach.StateMachine(
        outcomes=['all_objects_processed', 'cleanup_failed'],
        input_keys=['detected_objects', 'objects_to_pick', 'objects_picked_count', 'objects_placed_count'],
        output_keys=['objects_picked_count', 'objects_placed_count', 'failed_objects']
    )

    # 初始化内层状态机的用户数据
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
        # 1. 选择下一个目标物品
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

        # 2. 执行抓取
        smach.StateMachine.add(
            'EXECUTE_PICK',
            ExecutePick(),
            transitions={
                'pick_succeeded': 'NAVIGATE_TO_DEST',
                'pick_failed': 'SELECT_TARGET',  # 重试下一个
                'fatal_error': 'cleanup_failed'
            },
            remapping={
                'selected_object': 'selected_object',
                'object_category': 'object_category',
                'grasp_pose': 'grasp_pose',
                'objects_picked_count': 'objects_picked_count',
                'failed_objects': 'failed_objects'
            }
        )

        # 3. 导航到目的地
        smach.StateMachine.add(
            'NAVIGATE_TO_DEST',
            NavigateToDest(),
            transitions={
                'arrived': 'PERCEIVE_DEST',
                'navigation_failed': 'EXECUTE_PLACE',  # 尝试原地放置
                'fatal_error': 'cleanup_failed'
            },
            remapping={
                'destination': 'destination'
            }
        )

        # 4. 感知目的地（柜子检测等）
        smach.StateMachine.add(
            'PERCEIVE_DEST',
            PerceiveDest(),
            transitions={
                'perception_done': 'EXECUTE_PLACE',
                'perception_failed': 'EXECUTE_PLACE',  # 使用默认位置
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

        # 5. 执行放置
        smach.StateMachine.add(
            'EXECUTE_PLACE',
            ExecutePlace(),
            transitions={
                'place_succeeded': 'NAVIGATE_BACK_TO_KITCHEN',  # 放完返回桌边
                'place_failed':    'NAVIGATE_BACK_TO_KITCHEN',  # 失败也要回去取下一件
                'fatal_error':     'cleanup_failed'
            },
            remapping={
                'place_pose': 'place_pose',
                'selected_object': 'selected_object',
                'objects_placed_count': 'objects_placed_count',  # input+output：先读后写
                'destination': 'destination',
                'failed_objects': 'failed_objects'
            }
        )

        # 6. 返回厨房桌边，准备抓取下一件物品
        smach.StateMachine.add(
            'NAVIGATE_BACK_TO_KITCHEN',
            NavigateBackToKitchen(),
            transitions={
                'returned':          'SELECT_TARGET',
                'navigation_failed': 'SELECT_TARGET'  # 导航失败也尝试继续
            }
        )

    return cleanup_sm


def create_main_state_machine():
    """创建外层主状态机"""
    main_sm = smach.StateMachine(outcomes=['task_succeeded', 'task_failed'])

    # 初始化用户数据
    main_sm.userdata.detected_objects = []
    main_sm.userdata.objects_to_pick = []
    main_sm.userdata.objects_picked_count = 0
    main_sm.userdata.objects_placed_count = 0
    main_sm.userdata.failed_objects = []

    with main_sm:
        # 1. 初始化系统
        smach.StateMachine.add(
            'INIT_SYSTEM',
            InitSystem(),
            transitions={
                'initialized': 'NAVIGATE_TO_KITCHEN',
                'init_failed': 'task_failed'
            }
        )

        # 2. 导航到厨房
        smach.StateMachine.add(
            'NAVIGATE_TO_KITCHEN',
            NavigateToKitchen(),
            transitions={
                'arrived': 'ASSESS_SCENE',
                'navigation_failed': 'task_failed'
            }
        )

        # 3. 评估场景（检测物体）
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

        # 4. 嵌入内层清理循环
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

        # 5. 准备早餐（可选）
        smach.StateMachine.add(
            'SERVE_BREAKFAST',
            ServeBreakfast(),
            transitions={
                'breakfast_served': 'TASK_COMPLETED',
                'breakfast_skipped': 'TASK_COMPLETED',
                'failed': 'task_failed'
            }
        )

        # 6. 任务完成
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
    rospy.init_node('pick_place_task')

    rospy.loginfo("=" * 60)
    rospy.loginfo(" " * 10 + "RoboCup@Home Pick and Place 任务")
    rospy.loginfo("=" * 60)

    # 创建状态机
    sm = create_main_state_machine()

    # 启动introspection服务器（用于可视化）
    sis = smach_ros.IntrospectionServer('pick_place_task', sm, '/PICK_PLACE_TASK')
    sis.start()

    rospy.loginfo("SMACH状态机已启动")
    rospy.loginfo("可使用 rosrun smach_viewer smach_viewer.py 进行可视化")
    rospy.loginfo("=" * 60)

    # 执行状态机
    rospy.loginfo("开始执行Pick and Place任务...")
    outcome = sm.execute()

    # 输出结果
    rospy.loginfo("=" * 60)
    if outcome == 'task_succeeded':
        rospy.loginfo("✓ 任务成功完成！")
    else:
        rospy.logerr("✗ 任务失败")
    rospy.loginfo("=" * 60)

    # 停止introspection服务器
    sis.stop()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("任务被中断")

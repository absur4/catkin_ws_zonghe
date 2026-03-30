#!/usr/bin/env python3
"""
任务数据结构定义
用于SMACH状态之间传递数据
"""
from dataclasses import dataclass, field
from typing import List

# Note: These will be imported from robocup_msgs once the package is compiled
# For now, we use placeholders
# from robocup_msgs.msg import DetectedObject, GraspPose, ShelfDetectionResult
# from geometry_msgs.msg import Pose


@dataclass
class TaskUserData:
    """任务执行过程中的所有数据"""

    # 感知数据
    detected_objects: List = field(default_factory=list)  # List[DetectedObject]
    objects_to_pick: List = field(default_factory=list)   # List[DetectedObject]
    shelf_info: object = None  # ShelfDetectionResult

    # 当前任务状态
    current_object_index: int = 0
    selected_object: object = None  # DetectedObject
    object_category: str = ""
    destination: str = ""

    # 姿态数据
    grasp_pose: object = None  # GraspPose
    place_pose: object = None  # Pose
    target_layer: int = 0

    # 统计信息
    objects_picked_count: int = 0
    objects_placed_count: int = 0
    failed_objects: List[str] = field(default_factory=list)

    # 任务配置
    max_retries: int = 3
    retry_count: int = 0

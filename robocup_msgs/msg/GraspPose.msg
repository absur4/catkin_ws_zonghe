# 抓取姿态 (Approach-Grasp-Retreat)
geometry_msgs/Pose pre_grasp    # 预抓取姿态
geometry_msgs/Pose grasp        # 抓取姿态
geometry_msgs/Pose post_grasp   # 抓取后姿态
float32 gripper_width           # 夹爪开度 (米, 0.0-0.07)
string approach_direction       # 接近方向: "top", "side", "front"

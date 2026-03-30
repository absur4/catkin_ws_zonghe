# 柜子单层信息
int32 layer                    # 层号 (从上到下: 1, 2, 3, ...)
int32 y_top                    # 顶部y坐标 (像素)
int32 y_bottom                 # 底部y坐标 (像素)
geometry_msgs/Pose center_pose # 层中心3D位姿 (base_link坐标系)
float32 width                  # 层宽度 (米)
float32 depth                  # 层深度 (米)
float32 height                 # 层高度 (米)

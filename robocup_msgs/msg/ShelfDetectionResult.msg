# 柜子检测结果
Header header
geometry_msgs/Pose cabinet_pose  # 柜子整体位姿
int32[] cabinet_bbox             # 柜子2D边界框 [x1, y1, x2, y2]
ShelfLayer[] layers              # 各层信息
int32 num_layers                 # 总层数

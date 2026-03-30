# 单个检测到的物体
string class_name              # 物品类别名称
geometry_msgs/Pose pose        # 物体在世界坐标系中的位姿
geometry_msgs/Point centroid   # 质心坐标
float32 confidence             # 检测置信度 (0.0-1.0)
int32[] bbox                   # 2D边界框 [x1, y1, x2, y2]
string category                # 分类结果: "utensil", "food", "trash", "unknown"

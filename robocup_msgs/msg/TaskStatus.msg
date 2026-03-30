# 任务状态
string current_state            # 当前状态名称
string status                   # 状态: "running", "success", "failed", "paused"
int32 objects_picked            # 已抓取物品数量
int32 objects_placed            # 已放置物品数量
string[] remaining_objects      # 剩余待处理物品
string error_message            # 错误信息

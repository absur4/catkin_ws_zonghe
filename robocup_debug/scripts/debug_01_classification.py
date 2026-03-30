#!/usr/bin/env python3
"""
debug_01_classification.py — 分类服务调试脚本

先决条件：
  rosrun robocup_perception object_classifier.py

功能：对一组物品名批量调用真实 /classify_object 服务，打印对照表
"""
import sys
import rospy
from robocup_msgs.srv import ClassifyObject, ClassifyObjectRequest

# 测试物品列表（含边界情况）
TEST_ITEMS = [
    # cleanable → dishwasher
    'cup', 'spoon', 'fork', 'knife', 'plate', 'mug', 'bowl',
    # other → cabinet（默认）
    'apple', 'banana', 'bread',
    # 未知物品 → other → cabinet
    'unknown_xyz',
]

COL_W = 16


def classify_all(items):
    rospy.loginfo("等待 /classify_object 服务...")
    try:
        rospy.wait_for_service('/classify_object', timeout=10.0)
    except rospy.ROSException:
        rospy.logerr("/classify_object 服务不可用（超时 10s）")
        rospy.logerr("请先运行：rosrun robocup_perception object_classifier.py")
        sys.exit(1)

    srv = rospy.ServiceProxy('/classify_object', ClassifyObject)

    header = (
        f"{'物品':<{COL_W}} {'类别':<{COL_W}} {'目的地':<{COL_W}} {'状态'}"
    )
    separator = '-' * (COL_W * 3 + 10)

    rospy.loginfo("")
    rospy.loginfo("=" * (COL_W * 3 + 10))
    rospy.loginfo("  /classify_object 服务测试结果")
    rospy.loginfo("=" * (COL_W * 3 + 10))
    rospy.loginfo(header)
    rospy.loginfo(separator)

    ok_count = 0
    fail_count = 0

    for item in items:
        req = ClassifyObjectRequest()
        req.object_name = item
        try:
            resp = srv(req)
            if resp.success:
                status = "OK"
                ok_count += 1
            else:
                status = "FAIL(服务返回 success=False)"
                fail_count += 1
            rospy.loginfo(
                f"{item:<{COL_W}} {resp.category:<{COL_W}} {resp.destination:<{COL_W}} {status}"
            )
        except rospy.ServiceException as e:
            fail_count += 1
            rospy.logerr(f"{item:<{COL_W}} 调用失败: {e}")

    rospy.loginfo(separator)
    rospy.loginfo(f"共 {len(items)} 项：成功 {ok_count}，失败 {fail_count}")

    # 验证预期分类
    rospy.loginfo("")
    rospy.loginfo("--- 预期验证 ---")
    expected = {
        'cup': 'dishwasher', 'spoon': 'dishwasher', 'fork': 'dishwasher',
        'knife': 'dishwasher', 'plate': 'dishwasher', 'mug': 'dishwasher',
        'bowl': 'dishwasher',
        'apple': 'cabinet', 'banana': 'cabinet', 'bread': 'cabinet',
        'unknown_xyz': 'cabinet',
    }
    srv2 = rospy.ServiceProxy('/classify_object', ClassifyObject)
    mismatch = 0
    for item, expected_dest in expected.items():
        req = ClassifyObjectRequest()
        req.object_name = item
        try:
            resp = srv2(req)
            if resp.destination != expected_dest:
                rospy.logwarn(
                    f"[MISMATCH] {item}: 期望 {expected_dest}，实际 {resp.destination}"
                )
                mismatch += 1
        except rospy.ServiceException:
            pass

    if mismatch == 0:
        rospy.loginfo("所有预期分类验证通过 ✓")
    else:
        rospy.logwarn(f"{mismatch} 项分类与预期不符，请检查 classification_rules.yaml")

    # 提示 trash 关键字设置
    rospy.loginfo("")
    rospy.loginfo("提示：若要测试 trash → trash_bin，请先运行：")
    rospy.loginfo('  rosparam set /object_classifier/trash_keywords "[bottle, wrapper, tissue]"')
    rospy.loginfo('  然后测试：bottle, wrapper, tissue')


def main():
    rospy.init_node('debug_01_classification', anonymous=True)
    rospy.loginfo("== debug_01_classification 启动 ==")
    classify_all(TEST_ITEMS)
    rospy.loginfo("== debug_01_classification 完成 ==")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

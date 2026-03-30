#!/usr/bin/env python3
"""
服务测试脚本
测试各个ROS服务的基本功能
"""
import rospy
from robocup_msgs.srv import DetectObjects, ClassifyObject, ComputeGraspPose
from robocup_msgs.msg import DetectedObject
from geometry_msgs.msg import Pose


def test_classify_service():
    """测试物品分类服务"""
    print("\n" + "="*60)
    print("测试物品分类服务 /classify_object")
    print("="*60)

    try:
        rospy.wait_for_service('/classify_object', timeout=5.0)
        classify = rospy.ServiceProxy('/classify_object', ClassifyObject)

        # 测试几种物品
        test_items = ['cup', 'plate', 'spoon', 'apple', 'wrapper']

        for item in test_items:
            resp = classify(item)
            if resp.success:
                print(f"✓ {item:15s} -> 类别: {resp.category:10s}, 目的地: {resp.destination}")
            else:
                print(f"✗ {item} 分类失败")

        return True

    except Exception as e:
        print(f"✗ 分类服务测试失败: {e}")
        return False


def test_grasp_pose_service():
    """测试抓取姿态计算服务"""
    print("\n" + "="*60)
    print("测试抓取姿态计算服务 /compute_grasp_pose")
    print("="*60)

    try:
        rospy.wait_for_service('/compute_grasp_pose', timeout=5.0)
        compute_grasp = rospy.ServiceProxy('/compute_grasp_pose', ComputeGraspPose)

        # 创建测试物体
        obj = DetectedObject()
        obj.class_name = "cup"
        obj.pose.position.x = 0.5
        obj.pose.position.y = 0.0
        obj.pose.position.z = 0.3
        obj.pose.orientation.w = 1.0

        resp = compute_grasp(target_object=obj, grasp_type="auto")

        if resp.success:
            print(f"✓ 抓取姿态计算成功")
            print(f"  抓取位置: ({resp.grasp_pose.grasp.position.x:.2f}, "
                  f"{resp.grasp_pose.grasp.position.y:.2f}, "
                  f"{resp.grasp_pose.grasp.position.z:.2f})")
            print(f"  预抓取高度: {resp.grasp_pose.pre_grasp.position.z:.2f}m")
            print(f"  夹爪宽度: {resp.grasp_pose.gripper_width:.3f}m")
            print(f"  接近方向: {resp.grasp_pose.approach_direction}")
        else:
            print(f"✗ 抓取姿态计算失败: {resp.message}")

        return resp.success

    except Exception as e:
        print(f"✗ 抓取姿态服务测试失败: {e}")
        return False


def print_service_list():
    """打印所有可用服务"""
    print("\n" + "="*60)
    print("当前可用的RoboCup服务")
    print("="*60)

    services = rospy.get_service_list()
    robocup_services = [s for s in services if 'detect' in s or 'compute' in s or 'classify' in s]

    if robocup_services:
        for service in sorted(robocup_services):
            print(f"  {service}")
    else:
        print("  未找到RoboCup相关服务")


def main():
    """主函数"""
    rospy.init_node('test_services', anonymous=True)

    print("\n" + "*"*60)
    print("RoboCup@Home 服务测试工具")
    print("*"*60)

    # 打印服务列表
    print_service_list()

    # 等待一下确保服务就绪
    rospy.sleep(1.0)

    # 测试分类服务
    classify_ok = test_classify_service()

    # 测试抓取姿态服务
    grasp_ok = test_grasp_pose_service()

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"物品分类服务: {'✓ 通过' if classify_ok else '✗ 失败'}")
    print(f"抓取姿态服务: {'✓ 通过' if grasp_ok else '✗ 失败'}")
    print("="*60)

    if classify_ok and grasp_ok:
        print("✓ 所有测试通过！")
        return 0
    else:
        print("✗ 部分测试失败，请检查服务是否正常运行")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        import sys
        sys.exit(exit_code)
    except rospy.ROSInterruptException:
        print("测试被中断")
        import sys
        sys.exit(2)

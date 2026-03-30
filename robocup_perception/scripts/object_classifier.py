#!/usr/bin/env python3
"""
物品分类服务
根据物品名称返回类别和目的地
"""
import rospy
import yaml
import os
from robocup_msgs.srv import ClassifyObject, ClassifyObjectResponse


class ObjectClassifier:
    def __init__(self):
        rospy.init_node('object_classifier')

        # 加载分类规则
        default_rules = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'config', 'classification_rules.yaml')
        )
        rules_path = rospy.get_param(
            '~classification_rules',
            default_rules
        )

        rospy.loginfo(f"加载分类规则: {rules_path}")

        try:
            with open(rules_path, 'r') as f:
                self.rules = yaml.safe_load(f)
            rospy.loginfo(f"成功加载 {len(self.rules.get('object_categories', {}))} 个类别")
        except Exception as e:
            rospy.logerr(f"加载分类规则失败: {e}")
            # 使用默认规则
            self.rules = self.get_default_rules()

        # 从参数服务器动态覆盖垃圾类关键词（Setup Days 赛前注入）
        trash_keywords = rospy.get_param('~trash_keywords', [])
        if trash_keywords:
            if 'object_categories' in self.rules and 'trash' in self.rules['object_categories']:
                self.rules['object_categories']['trash']['keywords'] = trash_keywords
                rospy.loginfo(f"垃圾类关键词已从 rosparam 覆盖: {trash_keywords}")

        # 创建服务
        self.service = rospy.Service('/classify_object', ClassifyObject, self.handle_classify)
        rospy.loginfo("✓ 物品分类服务已就绪")

    def get_default_rules(self):
        """返回默认分类规则（对齐 rulebook2026 §5.2）"""
        return {
            'object_categories': {
                'cleanable': {
                    'keywords': ['cup', 'mug', 'plate', 'dish', 'bowl', 'spoon', 'fork', 'knife'],
                    'destination': 'dishwasher'
                },
                'trash': {
                    'keywords': [],  # 运行时由 rosparam 覆盖
                    'destination': 'trash_bin'
                }
            },
            'default_category': 'other',
            'default_destination': 'cabinet'
        }

    def handle_classify(self, req):
        """根据物品名称返回类别和目的地"""
        item_name = req.object_name.lower()
        rospy.loginfo(f"分类请求: {item_name}")

        # 遍历规则
        for category, info in self.rules.get('object_categories', {}).items():
            keywords = info.get('keywords', [])
            if any(keyword in item_name for keyword in keywords):
                resp = ClassifyObjectResponse()
                resp.category = category
                resp.destination = info.get('destination', 'trash_bin')
                resp.success = True
                rospy.loginfo(f"  ✓ 分类: {category}, 目的地: {resp.destination}")
                return resp

        # 默认分类
        rospy.logwarn(f"  未找到匹配规则，使用默认分类")
        resp = ClassifyObjectResponse()
        resp.category = self.rules.get('default_category', 'trash')
        resp.destination = self.rules.get('default_destination', 'trash_bin')
        resp.success = True
        return resp


if __name__ == '__main__':
    try:
        classifier = ObjectClassifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("物品分类服务已停止")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""代码内调用 detect_grasp API 的示例（无命令行参数）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .pipeline import DetectGraspConfig, DetectGraspRunner
except ImportError:  # 兼容脚本直接运行
    from pipeline import DetectGraspConfig, DetectGraspRunner


def load_profile_bundle(
    profile_bundle_path: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """从单个 JSON 文件读取 default_profile 与 category_profiles。"""
    p = Path(profile_bundle_path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {profile_bundle_path}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    default_profile = data.get("default_profile", {})
    category_profiles = data.get("category_profiles", {})

    if not isinstance(default_profile, dict):
        raise ValueError("default_profile 必须是 dict")
    if not isinstance(category_profiles, dict):
        raise ValueError("category_profiles 必须是 dict")

    return default_profile, category_profiles


def print_result(tag: str, result: dict[str, Any]) -> None:
    """打印关键输出字段，便于快速查看效果。"""
    best_conf = result.get("best_grasp_conf")
    best_pose = result.get("best_grasp_pose")
    payload = result.get("result_payload", {})
    src = payload.get("source", {})

    print(f"\n[{tag}] class: {src.get('class_name')}, conf: {src.get('confidence')}")
    print(f"[{tag}] best_grasp_conf: {best_conf}")
    print(f"[{tag}] result_json_path: {result.get('result_json_path')}")

    if best_pose is None:
        print(f"[{tag}] best_grasp_pose: None")
    else:
        pose = np.asarray(best_pose, dtype=np.float64)
        print(f"[{tag}] best_grasp_pose shape: {pose.shape}")
        print(pose)


def run_multi_category_grasp(
    *,
    profile_bundle_path: str,
    rgb_path: str,
    depth_path: str,
    categories: list[str],
    conda_exe: str = "conda",
    detect_env: str = "dsam2",
    grasp_env: str = "GraspGen",
) -> dict[str, dict[str, Any]]:
    """对多个类别串行推理，复用同一批模型，返回每个类别结果。"""
    if not categories:
        raise ValueError("categories 不能为空")

    if not Path(rgb_path).exists() or not Path(depth_path).exists():
        raise FileNotFoundError(f"RGBD 文件不存在: rgb={rgb_path}, depth={depth_path}")

    default_profile, category_profiles = load_profile_bundle(profile_bundle_path)

    cfg = DetectGraspConfig(
        conda_exe=conda_exe,
        detect_env_name=detect_env,
        grasp_env_name=grasp_env,
        input_source="rgbd_files",
        rgb_path=rgb_path,
        depth_path=depth_path,
        default_profile=default_profile,
        category_profiles=category_profiles,
    )

    results: dict[str, dict[str, Any]] = {}

    # 常驻 runner：模型只加载一次，后续切类别仅切运行参数。
    with DetectGraspRunner(cfg) as runner:
        for category in categories:
            result = runner.run(category=category)
            results[category] = result
            print_result(category, result)

        # 演示：单次调用临时覆盖参数（不修改 JSON 配置文件）。
        override_result = runner.run(
            category=categories[0],
            profile_override={
                "estimator_config": {
                    "grasp_threshold": 0.72,
                    "topk_num_grasps": 25,
                }
            },
        )
        results[f"{categories[0]}+override"] = override_result
        print_result(f"{categories[0]}+override", override_result)

    return results


def main() -> int:
    """最小可运行示例：改下面三个路径后可直接运行。"""
    profile_bundle_path = "/home/h/PPC/src/sg_pkg/scripts/detect_grasp/profile_bundle_template.json"
    rgb_path = "/home/h/PPC/src/GraspGen/rs_data/4object/color.png"
    depth_path = "/home/h/PPC/src/GraspGen/rs_data/4object/depth.png"

    categories = ["cola", "bottle"]

    _ = run_multi_category_grasp(
        profile_bundle_path=profile_bundle_path,
        rgb_path=rgb_path,
        depth_path=depth_path,
        categories=categories,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

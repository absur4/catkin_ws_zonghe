#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry for detect_grasp pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .pipeline import DetectGraspConfig, run_pipeline
    from .protocol import DEFAULT_INTRINSICS
except ImportError:  # script mode
    from pipeline import DetectGraspConfig, run_pipeline
    from protocol import DEFAULT_INTRINSICS


def _load_json_if_exists(path: str) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_profile_bundle(path: str) -> tuple[dict, dict]:
    if not path:
        return {}, {}
    data = _load_json_if_exists(path)
    default_profile = data.get("default_profile", {})
    category_profiles = data.get("category_profiles", {})
    if not isinstance(default_profile, dict):
        raise ValueError("profile bundle field 'default_profile' must be a dict")
    if not isinstance(category_profiles, dict):
        raise ValueError("profile bundle field 'category_profiles' must be a dict")
    return default_profile, category_profiles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", default="default")
    parser.add_argument("--input_source", default="rgbd_files", choices=["camera", "rgbd_files"])
    parser.add_argument("--rgb_path", default="")
    parser.add_argument("--depth_path", default="")

    parser.add_argument("--conda_exe", default="conda")
    parser.add_argument("--detect_env", default="dsam2")
    parser.add_argument("--grasp_env", default="GraspGen")
    parser.add_argument(
        "--grasp_mode",
        default="graspgen",
        choices=["graspgen", "tableware_pca"],
    )

    parser.add_argument("--profile_bundle_json", default="")
    parser.add_argument("--default_profile_json", default="")
    parser.add_argument("--category_profiles_json", default="")

    parser.add_argument("--text_prompt", default="")
    parser.add_argument("--segmentation_method", default="", choices=["", "bbox", "sam2_mask"])

    args = parser.parse_args()

    bundle_default_profile, bundle_category_profiles = _load_profile_bundle(
        args.profile_bundle_json
    )
    default_profile = bundle_default_profile
    category_profiles = dict(bundle_category_profiles)

    if args.default_profile_json:
        default_profile = _load_json_if_exists(args.default_profile_json)
    if args.category_profiles_json:
        category_profiles.update(_load_json_if_exists(args.category_profiles_json))

    config = DetectGraspConfig(
        category=args.category,
        input_source=args.input_source,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        intrinsics=dict(DEFAULT_INTRINSICS),
        default_profile=default_profile if default_profile else DetectGraspConfig().default_profile,
        category_profiles=category_profiles,
        conda_exe=args.conda_exe,
        detect_env_name=args.detect_env,
        grasp_env_name=args.grasp_env,
        grasp_mode=args.grasp_mode,
    )

    profile_override = {}
    if args.text_prompt:
        profile_override["text_prompt"] = args.text_prompt
    if args.segmentation_method:
        profile_override.setdefault("segmenter_kwargs", {})["segmentation_method"] = args.segmentation_method

    result = run_pipeline(
        config,
        profile_override=profile_override,
        grasp_mode=args.grasp_mode,
    )

    print("[OK] Pipeline done")
    print(f"grasp_mode: {result.get('grasp_mode', args.grasp_mode)}")
    print(f"status: {result.get('status', 'ok')}")
    if result.get("status", "ok") != "ok":
        print(f"message: {result.get('message', '')}")
    print(f"best_grasp_conf: {result['best_grasp_conf']}")
    if result["best_grasp_pose"] is not None:
        print("best_grasp_pose:")
        print(np.asarray(result["best_grasp_pose"]))
    else:
        print("best_grasp_pose: None")
    print("result_json_path:", result["result_json_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

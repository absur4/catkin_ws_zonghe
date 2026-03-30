#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Load scene/object clouds from NPZ and run GraspGenPointCloudEstimator.

When called via the pipeline, reads config from GRASP_PIPELINE_CONFIG env var.
Can also be called programmatically via run_grasp_estimation(cfg).
"""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from graspgen_pc_grasp_estimator import (
    GraspGenPointCloudConfig,
    GraspGenPointCloudEstimator,
)
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    visualize_grasp,
    visualize_pointcloud,
)


# -------- Default values (used when no config is provided) --------
DEFAULT_WORK_DIR = Path(__file__).resolve().parents[1] / "pc_npz"


def _as_jsonable(result):
    grasp_info = result.get("grasp_info", {})
    grasp_conf = grasp_info.get("grasp_conf", result.get("grasp_conf", []))
    grasp_poses = grasp_info.get("grasp_poses", result.get("grasp_poses", []))
    return {
        "gripper_name": result.get("gripper_name"),
        "best_grasp_pose": None
        if result.get("best_grasp_pose") is None
        else np.asarray(result["best_grasp_pose"]).tolist(),
        "best_grasp_conf": result.get("best_grasp_conf"),
        "num_grasps": int(np.asarray(grasp_conf).shape[0]),
        "grasp_conf": np.asarray(grasp_conf).tolist(),
        "grasp_poses": np.asarray(grasp_poses).tolist(),
    }


def _visualize_like_demo_scene(result, num_visualize_grasps=10,
                               block_after_visualization=True):
    vis = create_visualizer()
    vis.delete()

    scene_info = result.get("scene_info", {})
    object_info = result.get("object_info", {})
    grasp_info = result.get("grasp_info", {})

    scene_pc = np.asarray(scene_info.get("collision_scene_pc", np.empty((0, 3))))
    scene_color = np.asarray(
        scene_info.get("collision_scene_color", np.empty((0, 3)))
    )
    obj_pc = np.asarray(object_info.get("pc", np.empty((0, 3))))
    obj_color = np.asarray(object_info.get("pc_color", np.empty((0, 3))))
    grasp_poses = np.asarray(grasp_info.get("grasp_poses", np.empty((0, 4, 4))))
    grasp_conf = np.asarray(grasp_info.get("grasp_conf", np.empty((0,))))

    if scene_pc.shape[0] > 0 and scene_color.shape[0] == scene_pc.shape[0]:
        visualize_pointcloud(vis, "pc_scene", scene_pc, scene_color, size=0.0025)
    if obj_pc.shape[0] > 0 and obj_color.shape[0] == obj_pc.shape[0]:
        visualize_pointcloud(vis, "pc_obj", obj_pc, obj_color, size=0.005)

    if grasp_poses.shape[0] == 0:
        print("[INFO] 无抓姿可视化")
        return

    gripper_name = result.get("gripper_name", "robotiq_2f_140")
    order = np.argsort(-grasp_conf)
    top_n = max(1, int(num_visualize_grasps))
    top_idx = order[: min(top_n, len(order))]
    best_idx = int(top_idx[0])

    for rank, idx in enumerate(top_idx):
        pose = grasp_poses[idx]
        conf = float(grasp_conf[idx])
        if idx == best_idx:
            print(f"[VIS] best grasp conf={conf:.4f}")
            visualize_grasp(
                vis,
                "best_grasp",
                pose,
                color=[0, 255, 0],
                gripper_name=gripper_name,
                linewidth=10.0,
            )
        else:
            print(f"[VIS] top grasp #{rank + 1} conf={conf:.4f}")
            visualize_grasp(
                vis,
                f"grasps/{rank:03d}/grasp",
                pose,
                color=[255, 165, 0],
                gripper_name=gripper_name,
                linewidth=6.0,
            )

    if block_after_visualization:
        if sys.stdin is not None and sys.stdin.isatty():
            input("[INFO] Press Enter to close visualization and continue...")
        else:
            print("[INFO] Non-interactive session detected, skip blocking input()")


def run_grasp_estimation(cfg=None):
    """Run grasp estimation from NPZ and save result JSON.

    Parameters
    ----------
    cfg : dict or None
        Configuration dict (same structure as PipelineConfig.to_json() output).
        If None, loads from GRASP_PIPELINE_CONFIG env var or uses defaults.

    Returns
    -------
    str
        Path to the output JSON file.
    """
    if cfg is None:
        config_path = os.environ.get("GRASP_PIPELINE_CONFIG", "")
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}

    estimator_config, estimator = _create_estimator(cfg)
    return _run_grasp_estimation_core(estimator, estimator_config, cfg)


def _create_estimator(cfg):
    """Create GraspGenPointCloudEstimator from config."""
    estimator_overrides = cfg.get("estimator_config", {
        "grasp_threshold": 0.80,
        "num_grasps": 200,
        "return_topk": True,
        "topk_num_grasps": 20,
        "filter_collisions": True,
        "collision_threshold": 0.001,
    })
    estimator_config = GraspGenPointCloudConfig(**estimator_overrides)
    estimator = GraspGenPointCloudEstimator(estimator_config)
    return estimator_config, estimator


def _run_grasp_estimation_core(estimator, estimator_config, cfg):
    """Run grasp estimation with a pre-loaded estimator.

    Parameters
    ----------
    estimator : GraspGenPointCloudEstimator
        Already-initialized estimator (model loaded).
    estimator_config : GraspGenPointCloudConfig
        Estimator config (for JSON output).
    cfg : dict
        Configuration dict.

    Returns
    -------
    str
        Path to the output JSON file.
    """
    work_dir = cfg.get("work_dir", str(DEFAULT_WORK_DIR))
    npz_filename = cfg.get("npz_filename", "scene_object_clouds.npz")
    result_json_filename = cfg.get("result_json_filename", "grasp_result.json")
    input_npz_path = str(Path(work_dir) / npz_filename)
    output_json_path = str(Path(work_dir) / result_json_filename)

    if not os.path.exists(input_npz_path):
        raise FileNotFoundError(f"Input NPZ not found: {input_npz_path}")
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    enable_visualization = cfg.get("enable_visualization", True)
    num_visualize_grasps = cfg.get("num_visualize_grasps", 10)
    block_after_visualization = cfg.get("block_after_visualization", True)

    # ── Load NPZ ──
    with np.load(input_npz_path, allow_pickle=False) as npz:
        scene_pc = npz["scene_pc"]
        scene_color = npz["scene_color"]
        object_pc = npz["object_pc"]
        object_color = npz["object_color"]
        meta_json = str(npz["meta_json"])
    source_meta = json.loads(meta_json)

    # ── Run estimation ──
    result = estimator.estimate_grasps(
        scene_pc=scene_pc,
        object_pc=object_pc,
        scene_color=scene_color,
        object_color=object_color,
    )

    # ── Save JSON ──
    out = {
        "source": source_meta,
        "estimator_config": asdict(estimator_config),
        "result": _as_jsonable(result),
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[OK] Saved grasp result JSON: {output_json_path}")
    best_conf = out["result"]["best_grasp_conf"]
    if best_conf is None:
        print("[INFO] No valid grasp generated")
    else:
        print(f"[INFO] Best grasp confidence: {best_conf:.4f}")
        print("[INFO] Best grasp pose (4x4):")
        print(np.asarray(out["result"]["best_grasp_pose"]))

    if enable_visualization:
        _visualize_like_demo_scene(
            result,
            num_visualize_grasps=num_visualize_grasps,
            block_after_visualization=block_after_visualization,
        )

    return output_json_path


def worker_main():
    """Worker mode: load model once, process multiple configs from stdin.

    Protocol:
      - Reads config JSON file paths from stdin (one per line).
      - Prints "__DONE__" to stdout after each run.
      - Exits on "__QUIT__" or EOF.
    """
    estimator = None
    estimator_config = None

    for line in sys.stdin:
        config_path = line.strip()
        if not config_path:
            continue
        if config_path == "__QUIT__":
            break

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Lazy init: create estimator on first request
        if estimator is None:
            estimator_config, estimator = _create_estimator(cfg)

        try:
            _run_grasp_estimation_core(estimator, estimator_config, cfg)
            sys.stdout.write("__DONE__\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] Grasp estimation failed: {e}", file=sys.stderr)
            sys.stdout.write("__ERROR__\n")
            sys.stdout.flush()


def main():
    run_grasp_estimation()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker_main()
    else:
        main()

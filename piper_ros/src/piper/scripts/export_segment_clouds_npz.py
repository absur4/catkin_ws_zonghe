#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run ObjectPointCloudSegmenter and export selected scene/object clouds to NPZ.

When called via the pipeline, reads config from GRASP_PIPELINE_CONFIG env var.
Can also be called programmatically via run_segmentation(cfg).
"""

import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from object_pc_segment import ObjectPointCloudSegmenter


# -------- Default values (used when no config is provided) --------
DEFAULT_WORK_DIR = Path(__file__).resolve().parents[1] / "pc_npz"

DEFAULT_INTRINSICS = {
    "fx": 901.2042236328125,
    "fy": 901.2002563476562,
    "cx": 656.2393188476562,
    "cy": 366.75244140625,
    "width": 1280,
    "height": 720,
}

DEFAULT_SEGMENTER_KWARGS = {
    "text_prompt": "bottle.",
    "output_dir": "/tmp/object_clouds",
    "voxel_size": 0.01,
    "table_remove_thresh": 0.02,
    "depth_range": [0.1, 1.0],
    "bbox_horizontal_shrink_ratio": 0.12,
    "save_clouds_to_disk": False,
}


def _select_result(results, policy="top_conf", index=0):
    if not results:
        raise RuntimeError("No objects detected by ObjectPointCloudSegmenter")

    if policy == "index":
        idx = index
        if idx < 0 or idx >= len(results):
            raise IndexError(
                f"target_object_index {idx} out of range for {len(results)} results"
            )
        selected = results[idx]
    elif policy == "top_conf":
        selected = max(results, key=lambda r: float(r.get("confidence", -1.0)))
        idx = results.index(selected)
    else:
        raise ValueError(f"Unsupported target_object_policy: {policy}")

    if selected["point_cloud"].shape[0] == 0:
        raise RuntimeError("Selected object has zero points")

    return idx, selected


def _load_rgbd_from_files(rgb_path, depth_path, depth_value_in_meters,
                          depth_scale, intrinsics):
    if not rgb_path or not depth_path:
        raise ValueError(
            "rgbd_files 模式下需要设置 rgb_path 和 depth_path"
        )

    color_image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_image is None:
        raise FileNotFoundError(f"Failed to load RGB image: {rgb_path}")

    depth_ext = Path(depth_path).suffix.lower()
    if depth_ext == ".npy":
        depth_image = np.load(depth_path)
    else:
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise FileNotFoundError(f"Failed to load depth image: {depth_path}")

    if depth_image.ndim == 3:
        depth_image = depth_image[:, :, 0]
    depth_image = depth_image.astype(np.float32)
    if not depth_value_in_meters:
        depth_image = depth_image / depth_scale

    h, w = color_image.shape[:2]
    if depth_image.shape[:2] != (h, w):
        raise ValueError(
            f"RGB/Depth 尺寸不一致: RGB={color_image.shape[:2]}, Depth={depth_image.shape[:2]}"
        )

    intr = dict(intrinsics)
    intr["width"] = w
    intr["height"] = h
    return color_image, depth_image, intr


def _run_pipeline_from_rgbd(segmenter, color_image, depth_image, intrinsics):
    """(Deprecated proxy method) Run pipeline directly from loaded RGBD arrays."""
    return segmenter.process_rgbd(color_image, depth_image, intrinsics)


def run_segmentation(cfg=None):
    """Run segmentation and export to NPZ.

    Parameters
    ----------
    cfg : dict or None
        Configuration dict (same structure as PipelineConfig.to_json() output).
        If None, loads from GRASP_PIPELINE_CONFIG env var or uses defaults.

    Returns
    -------
    str
        Path to the output NPZ file.
    """
    if cfg is None:
        config_path = os.environ.get("GRASP_PIPELINE_CONFIG", "")
        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}

    segmenter_kwargs, use_camera = _prepare_segmenter_kwargs(cfg)
    segmenter = ObjectPointCloudSegmenter(**segmenter_kwargs, use_camera=use_camera)
    try:
        return _run_segmentation_core(segmenter, cfg)
    finally:
        segmenter.close()


def _prepare_segmenter_kwargs(cfg):
    """Extract and prepare segmenter kwargs from config dict."""
    segmenter_kwargs = dict(cfg.get("segmenter_kwargs", DEFAULT_SEGMENTER_KWARGS))
    # Inject top-level text_prompt (overrides any value inside segmenter_kwargs)
    if "text_prompt" in cfg:
        segmenter_kwargs["text_prompt"] = cfg["text_prompt"]
    if "depth_range" in segmenter_kwargs and isinstance(
        segmenter_kwargs["depth_range"], list
    ):
        segmenter_kwargs["depth_range"] = tuple(segmenter_kwargs["depth_range"])
    input_source = cfg.get("input_source", "camera")
    use_camera = input_source == "camera"
    return segmenter_kwargs, use_camera


def _run_segmentation_core(segmenter, cfg):
    """Run segmentation with a pre-loaded segmenter and save NPZ.

    Parameters
    ----------
    segmenter : ObjectPointCloudSegmenter
        Already-initialized segmenter (model loaded).
    cfg : dict
        Configuration dict.

    Returns
    -------
    str
        Path to the output NPZ file.
    """
    work_dir = cfg.get("work_dir", str(DEFAULT_WORK_DIR))
    npz_filename = cfg.get("npz_filename", "scene_object_clouds.npz")
    output_npz_path = str(Path(work_dir) / npz_filename)
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)

    input_source = cfg.get("input_source", "camera")

    if input_source == "camera":
        results = segmenter.run()
    elif input_source == "rgbd_files":
        color_image, depth_image, intrinsics = _load_rgbd_from_files(
            rgb_path=cfg.get("rgb_path", ""),
            depth_path=cfg.get("depth_path", ""),
            depth_value_in_meters=cfg.get("depth_value_in_meters", False),
            depth_scale=cfg.get("depth_scale", 1000.0),
            intrinsics=cfg.get("intrinsics", DEFAULT_INTRINSICS),
        )
        results = _run_pipeline_from_rgbd(
            segmenter, color_image, depth_image, intrinsics
        )
    else:
        raise ValueError(f"Unsupported input_source: {input_source}")

    # ── Select target object ──
    policy = cfg.get("target_object_policy", "top_conf")
    index = cfg.get("target_object_index", 0)
    selected_idx, selected = _select_result(results, policy=policy, index=index)

    # ── Save NPZ ──
    meta = {
        "timestamp": datetime.now().isoformat(),
        "selected_index": selected_idx,
        "num_candidates": len(results),
        "class_name": selected.get("class_name"),
        "confidence": float(selected.get("confidence", 0.0)),
        "bbox": np.asarray(selected.get("bbox", [])).tolist(),
        "intrinsics": selected.get("intrinsics", {}),
        "policy": policy,
    }

    np.savez_compressed(
        output_npz_path,
        scene_pc=np.asarray(selected["scene_point_cloud"], dtype=np.float32),
        scene_color=np.asarray(selected["scene_point_color"], dtype=np.uint8),
        object_pc=np.asarray(selected["point_cloud"], dtype=np.float32),
        object_color=np.asarray(selected["point_cloud_color"], dtype=np.uint8),
        meta_json=np.asarray(json.dumps(meta), dtype=np.str_),
    )

    print(f"[OK] Saved intermediate NPZ: {output_npz_path}")
    print(
        f"[INFO] Selected object #{selected_idx}: {meta['class_name']} "
        f"(conf={meta['confidence']:.3f})"
    )
    print(
        f"[INFO] scene points={selected['scene_point_cloud'].shape[0]}, "
        f"object points={selected['point_cloud'].shape[0]}"
    )

    return output_npz_path


def worker_main():
    """Worker mode: load model once, process multiple configs from stdin.

    Protocol:
      - Reads config JSON file paths from stdin (one per line).
      - Prints "__DONE__" to stdout after each run.
      - Exits on "__QUIT__" or EOF.
    """
    import sys

    segmenter = None

    for line in sys.stdin:
        config_path = line.strip()
        if not config_path:
            continue
        if config_path == "__QUIT__":
            break

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        # Lazy init: create segmenter on first request
        if segmenter is None:
            segmenter_kwargs, use_camera = _prepare_segmenter_kwargs(cfg)
            segmenter = ObjectPointCloudSegmenter(
                **segmenter_kwargs, use_camera=use_camera
            )

        # Sync attributes for each run (no model reload needed)
        segmenter_kwargs, use_camera = _prepare_segmenter_kwargs(cfg)
        if hasattr(segmenter, "text_prompt") and "text_prompt" in segmenter_kwargs:
            segmenter.text_prompt = segmenter_kwargs["text_prompt"]
        if hasattr(segmenter, "segmentation_method") and "segmentation_method" in segmenter_kwargs:
            segmenter.segmentation_method = segmenter_kwargs["segmentation_method"]

        try:
            _run_segmentation_core(segmenter, cfg)
            sys.stdout.write("__DONE__\n")
            sys.stdout.flush()
        except Exception as e:
            print(f"[ERROR] Segmentation failed: {e}", file=sys.stderr)
            sys.stdout.write("__ERROR__\n")
            sys.stdout.flush()

    if segmenter is not None:
        segmenter.close()


def main():
    run_segmentation()


if __name__ == "__main__":
    import sys

    if "--worker" in sys.argv:
        worker_main()
    else:
        main()


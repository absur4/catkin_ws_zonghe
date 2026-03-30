#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate segmentation + grasp estimation across two conda environments.

Usage:
    # As CLI (uses default PipelineConfig):
    python run_pipeline_cross_env.py

    # As Python API:
    from pipeline_config import PipelineConfig
    from run_pipeline_cross_env import run_pipeline

    config = PipelineConfig()
    config.input_source = "rgbd_files"
    config.rgb_path = "/path/to/color.png"
    config.depth_path = "/path/to/depth.png"
    config.segmenter_kwargs["text_prompt"] = "cup."
    result_json_path = run_pipeline(config)
"""

import json
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pipeline_config import PipelineConfig

SCRIPT_DIR = Path(__file__).resolve().parent


class PipelineRunner:
    """Persistent pipeline runner that keeps worker processes alive.

    Usage::

        runner = PipelineRunner(config)
        runner.start()          # Loads models (slow, once)
        runner.run()            # Fast: reuses loaded models
        runner.run(new_config)  # Can override config per run
        runner.stop()           # Release workers

    Or with context manager::

        with PipelineRunner(config) as runner:
            for img_path in image_list:
                config.rgb_path = img_path
                runner.run(config)
    """

    def __init__(self, config=None):
        self.config = config or PipelineConfig()
        self._seg_proc = None
        self._grasp_proc = None

    def start(self):
        """Start worker subprocesses and load models."""
        conda_path = shutil.which(self.config.conda_exe)
        if conda_path is None:
            raise RuntimeError(
                f"Cannot find conda executable: {self.config.conda_exe}"
            )

        Path(self.config.work_dir).mkdir(parents=True, exist_ok=True)

        shared_env = os.environ.copy()
        shared_env["PYTHONUNBUFFERED"] = "1"

        export_script = SCRIPT_DIR / "export_segment_clouds_npz.py"
        grasp_script = SCRIPT_DIR / "run_grasp_from_npz.py"

        print("[PipelineRunner] Starting segmentation worker ...")
        self._seg_proc = subprocess.Popen(
            [
                conda_path, "run", "--no-capture-output",
                "-n", self.config.seg_env_name,
                "python", str(export_script), "--worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit → real-time terminal output
            text=True,
            env=shared_env,
        )

        print("[PipelineRunner] Starting grasp estimation worker ...")
        self._grasp_proc = subprocess.Popen(
            [
                conda_path, "run", "--no-capture-output",
                "-n", self.config.grasp_env_name,
                "python", str(grasp_script), "--worker",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            env=shared_env,
        )

        print("[PipelineRunner] Workers started. Models will load on first run.")

    def run(self, text_prompt: str | None = None,
            segmentation_method: str | None = None,
            input_source: str | None = None,
            rgb_path: str | None = None,
            depth_path: str | None = None) -> dict[str, any]:
        """Run the pipeline through persistent workers.

        Parameters
        ----------
        text_prompt : str, optional
            Override the target text prompt.
        segmentation_method : str, optional
            'bbox' or 'sam2_mask' to toggle segmentation methods on the fly. If None, uses the config from __init__.
        input_source : str or None
            Override input source ("camera" / "rgbd_files").
        rgb_path : str or None
            Override RGB image path.
        depth_path : str or None
            Override depth image path.

        Returns
        -------
        str
            Path to the result JSON file.
        """
        cfg = _apply_overrides(
            self.config,
            input_source=input_source, rgb_path=rgb_path,
            depth_path=depth_path, text_prompt=text_prompt,
            segmentation_method=segmentation_method,
        )

        if self._seg_proc is None or self._grasp_proc is None:
            raise RuntimeError("Workers not started. Call start() first.")

        Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
        config_json_path = os.path.join(cfg.work_dir, ".pipeline_config.json")
        cfg.to_json(config_json_path)

        t0 = time.time()

        # ── Segmentation ──
        self._send_and_wait(self._seg_proc, config_json_path, "segmentation")

        # ── Grasp estimation ──
        self._send_and_wait(self._grasp_proc, config_json_path, "grasp")

        t_end = time.time()
        print(f"[OK] Pipeline finished in {t_end - t0:.3f} seconds")
        print(f"[OUT] {cfg.npz_path}")
        print(f"[OUT] {cfg.result_json_path}")

        return _load_result(cfg.result_json_path)

    def _send_and_wait(self, proc, config_json_path, name):
        """Send config path to worker and wait for __DONE__ marker."""
        if proc.poll() is not None:
            raise RuntimeError(f"{name} worker process has exited unexpectedly")

        proc.stdin.write(config_json_path + "\n")
        proc.stdin.flush()

        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"{name} worker process ended unexpectedly"
                )
            line = line.rstrip("\n")
            if line == "__DONE__":
                return
            if line == "__ERROR__":
                raise RuntimeError(
                    f"{name} step failed (see error output above)"
                )
            print(line)  # Forward worker stdout to parent terminal

    def stop(self):
        """Stop worker processes."""
        for name, proc in [("seg", self._seg_proc), ("grasp", self._grasp_proc)]:
            if proc is not None and proc.poll() is None:
                try:
                    proc.stdin.write("__QUIT__\n")
                    proc.stdin.flush()
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
        self._seg_proc = None
        self._grasp_proc = None
        print("[PipelineRunner] Workers stopped.")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def _apply_overrides(config, *, input_source=None, rgb_path=None,
                     depth_path=None, text_prompt=None, segmentation_method=None):
    """Return a config copy with overrides applied (original unchanged)."""
    overrides = {}
    if input_source is not None:
        overrides["input_source"] = input_source
    if rgb_path is not None:
        overrides["rgb_path"] = rgb_path
    if depth_path is not None:
        overrides["depth_path"] = depth_path
    if text_prompt is not None:
        overrides["text_prompt"] = text_prompt

    # Handle nested overrides for segmenter_kwargs
    if segmentation_method is not None:
        # Create a mutable copy of segmenter_kwargs if it exists, or an empty dict
        segmenter_kwargs_copy = dict(config.segmenter_kwargs) if config.segmenter_kwargs else {}
        segmenter_kwargs_copy["segmentation_method"] = segmentation_method
        overrides["segmenter_kwargs"] = segmenter_kwargs_copy

    return replace(config, **overrides) if overrides else config


def _load_result(result_json_path):
    """Read result JSON and return a structured dict.

    Returns
    -------
    dict
        - result_json_path : str
        - best_grasp_pose  : np.ndarray (4, 4) or None
        - best_grasp_conf  : float or None
    """
    with open(result_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = data.get("result", {})
    pose = result.get("best_grasp_pose")
    return {
        "result_json_path": result_json_path,
        "best_grasp_pose": np.array(pose, dtype=np.float64) if pose is not None else None,
        "best_grasp_conf": result.get("best_grasp_conf"),
    }


def run_pipeline(config=None, *, input_source=None, rgb_path=None,
                 depth_path=None, text_prompt=None, segmentation_method=None):
    """Run the pipeline (single-shot, no model reuse).

    For repeated runs, use PipelineRunner instead.
    """
    if config is None:
        config = PipelineConfig()
    config = _apply_overrides(
        config,
        input_source=input_source, rgb_path=rgb_path,
        depth_path=depth_path, text_prompt=text_prompt,
        segmentation_method=segmentation_method,
    )

    conda_path = shutil.which(config.conda_exe)
    if conda_path is None:
        raise RuntimeError(f"Cannot find conda executable: {config.conda_exe}")

    Path(config.work_dir).mkdir(parents=True, exist_ok=True)

    config_json_path = os.path.join(config.work_dir, ".pipeline_config.json")
    config.to_json(config_json_path)

    shared_env = os.environ.copy()
    shared_env["PYTHONUNBUFFERED"] = "1"
    shared_env["GRASP_PIPELINE_CONFIG"] = config_json_path

    export_script = SCRIPT_DIR / "export_segment_clouds_npz.py"
    grasp_script = SCRIPT_DIR / "run_grasp_from_npz.py"

    cmd_export = [
        conda_path, "run", "--no-capture-output",
        "-n", config.seg_env_name,
        "python", str(export_script),
    ]
    cmd_grasp = [
        conda_path, "run", "--no-capture-output",
        "-n", config.grasp_env_name,
        "python", str(grasp_script),
    ]

    print(f"[RUN] {' '.join(cmd_export)}")
    subprocess.run(cmd_export, check=True, env=shared_env)

    print(f"[RUN] {' '.join(cmd_grasp)}")
    subprocess.run(cmd_grasp, check=True, env=shared_env)

    print(f"[OK] Pipeline finished")
    print(f"[OUT] {config.npz_path}")
    print(f"[OUT] {config.result_json_path}")

    return _load_result(config.result_json_path)


def main():
    config = PipelineConfig()
    # result = run_pipeline(config)

    # result["best_grasp_pose"]    # numpy array (4, 4)，最优抓姿齐次矩阵
    # result["best_grasp_conf"]    # float，最优抓姿置信度
    # result["result_json_path"]   # str，完整 JSON 文件路径

    with PipelineRunner(config) as runner:
        # print("\n=== First Run (Model initialization inside worker) ===")
        # t_start = time.time()
        # runner.run()                                       # 用默认配置
        # print(f"--- First Run Total Time: {time.time() - t_start:.3f}s ---\n")

        print("\n=== Second Run (Reusing loaded models) ===")
        t_start = time.time()
        try:
            runner.run(text_prompt="cola.", input_source="rgbd_files", rgb_path="/home/h/PPC/src/GraspGen/rs_data/4object/color.png", depth_path="/home/h/PPC/src/GraspGen/rs_data/4object/depth.png", segmentation_method="sam2_mask")
        except RuntimeError as e:
            print(f"[WARN] Run skipped: {e}")
        print(f"--- Second Run Total Time: {time.time() - t_start:.3f}s ---\n")

        # print("\n=== Third Run (Reusing loaded models) ===")
        # t_start = time.time()
        # try:
        #     runner.run(text_prompt="bottled water.", input_source="rgbd_files", rgb_path="/home/h/PPC/src/GraspGen/rs_data/4object/color.png", depth_path="/home/h/PPC/src/GraspGen/rs_data/4object/depth.png")                 # 只改检测目标
        # except RuntimeError as e:
        #     print(f"[WARN] Run skipped: {e}")
        # print(f"--- Third Run Total Time: {time.time() - t_start:.3f}s ---\n")

        # print("\n=== Fourth Run (Reusing loaded models) ===")
        # t_start = time.time()
        # try:
        #     runner.run(text_prompt="yellow lays.", input_source="rgbd_files", rgb_path="/home/h/PPC/src/GraspGen/rs_data/4object/color.png", depth_path="/home/h/PPC/src/GraspGen/rs_data/4object/depth.png")                 # 只改检测目标
        # except RuntimeError as e:
        #     print(f"[WARN] Run skipped: {e}")
        # print(f"--- Fourth Run Total Time: {time.time() - t_start:.3f}s ---\n")

        # print("\n=== Fifth Run (Reusing loaded models) ===")
        # t_start = time.time()
        # try:
        #     runner.run(text_prompt="peach drink.", input_source="rgbd_files", rgb_path="/home/h/PPC/src/GraspGen/rs_data/4object/color.png", depth_path="/home/h/PPC/src/GraspGen/rs_data/4object/depth.png")                 # 只改检测目标
        # except RuntimeError as e:
        #     print(f"[WARN] Run skipped: {e}")
        # print(f"--- Fifth Run Total Time: {time.time() - t_start:.3f}s ---\n")


if __name__ == "__main__":
    main()


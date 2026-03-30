#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Top-level orchestrator for persistent detect/grasp hub processes."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import signal
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .ipc import recv_message, send_message
    from .profile_registry import ProfileRegistry
    from .protocol import CMD_DETECT, CMD_GRASP, CMD_INIT, CMD_PING, CMD_QUIT
except ImportError:  # script mode
    from ipc import recv_message, send_message
    from profile_registry import ProfileRegistry
    from protocol import CMD_DETECT, CMD_GRASP, CMD_INIT, CMD_PING, CMD_QUIT


_THIS_DIR = Path(__file__).resolve().parent
_DETECT_WORKER = _THIS_DIR / "detect_hub_worker.py"
_GRASP_WORKER = _THIS_DIR / "grasp_hub_worker.py"


class DetectGraspOrchestrator:
    """Two-process runner: detect hub + grasp hub."""

    def __init__(
        self,
        *,
        conda_exe: str = "conda",
        detect_env_name: str = "dsam2",
        grasp_env_name: str = "GraspGen",
        verbose: bool = True,
        default_profile: dict[str, Any] | None = None,
        category_profiles: dict[str, dict[str, Any]] | None = None,
    ):
        self.conda_exe = conda_exe
        self.detect_env_name = detect_env_name
        self.grasp_env_name = grasp_env_name
        self.verbose = verbose

        self.registry = ProfileRegistry(
            default_profile=default_profile,
            category_profiles=category_profiles,
        )

        self._detect_proc: subprocess.Popen | None = None
        self._grasp_proc: subprocess.Popen | None = None

    def start(self) -> None:
        if self._detect_proc is not None or self._grasp_proc is not None:
            return

        conda_path = shutil.which(self.conda_exe)
        if conda_path is None:
            raise RuntimeError(f"Cannot find conda executable: {self.conda_exe}")

        shared_env = os.environ.copy()
        shared_env["PYTHONUNBUFFERED"] = "1"

        self._detect_proc = subprocess.Popen(
            [
                conda_path,
                "run",
                "--no-capture-output",
                "-n",
                self.detect_env_name,
                "python",
                str(_DETECT_WORKER),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=shared_env,
            bufsize=0,
        )

        self._grasp_proc = subprocess.Popen(
            [
                conda_path,
                "run",
                "--no-capture-output",
                "-n",
                self.grasp_env_name,
                "python",
                str(_GRASP_WORKER),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=shared_env,
            bufsize=0,
        )

        # Quick health check.
        self._rpc(self._detect_proc, {"cmd": CMD_PING}, "detect", timeout_s=120.0)
        self._rpc(self._grasp_proc, {"cmd": CMD_PING}, "grasp", timeout_s=120.0)
        init_profile = self.registry.get("default")
        self._rpc(
            self._detect_proc,
            {"cmd": CMD_INIT, "profile": init_profile, "need_camera": False},
            "detect",
            timeout_s=300.0,
        )
        self._rpc(
            self._grasp_proc,
            {"cmd": CMD_INIT, "profile": init_profile},
            "grasp",
            timeout_s=300.0,
        )
        if self.verbose:
            print(
                f"[DetectGrasp] workers ready + preloaded: detect_env={self.detect_env_name}, "
                f"grasp_env={self.grasp_env_name}"
            )

    def _rpc(
        self,
        proc: subprocess.Popen | None,
        req: dict[str, Any],
        name: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError(f"{name} worker is not started")
        if proc.poll() is not None:
            raise RuntimeError(f"{name} worker exited unexpectedly with code {proc.returncode}")

        send_message(proc.stdin, req)

        try:
            resp = recv_message(proc.stdout, timeout_s=timeout_s)
        except TimeoutError as exc:
            raise RuntimeError(
                f"{name} worker did not respond within {timeout_s:.1f}s"
            ) from exc
        except EOFError as exc:
            rc = proc.poll()
            if rc is None:
                raise RuntimeError(f"{name} worker pipe closed unexpectedly (EOF)") from exc

            msg = f"{name} worker exited unexpectedly with code {rc}"
            # Typical OOM-kill patterns: shell return code 137 / signal 9.
            if rc in (137, -signal.SIGKILL):
                msg += " (likely killed by system, e.g. OOM)"
            raise RuntimeError(msg) from exc

        if not isinstance(resp, dict) or "ok" not in resp:
            raise RuntimeError(f"{name} worker returned malformed response")
        if not resp.get("ok"):
            err = resp.get("error", {})
            message = err.get("message", "unknown error")
            detail = err.get("detail", "")
            if detail:
                raise RuntimeError(f"{name} failed: {message}\n{detail}")
            raise RuntimeError(f"{name} failed: {message}")

        return resp.get("data", {})

    def run(
        self,
        *,
        category: str,
        input_source: str,
        rgb_path: str | None = None,
        depth_path: str | None = None,
        color_image: np.ndarray | None = None,
        depth_image: np.ndarray | None = None,
        intrinsics: dict[str, Any] | None = None,
        depth_value_in_meters: bool = False,
        depth_scale: float = 1000.0,
        profile_override: dict[str, Any] | None = None,
        grasp_mode: str = "graspgen",
        camera_to_ee_transform: Any | None = None,
        ee_to_base_transform: Any | None = None,
    ) -> dict[str, Any]:
        if self._detect_proc is None or self._grasp_proc is None:
            raise RuntimeError("Workers not started. Call start() first.")
        if grasp_mode not in ("graspgen", "tableware_pca"):
            raise ValueError(f"Unsupported grasp_mode: {grasp_mode}")

        profile = self.registry.get(category, override=profile_override)
        input_payload: dict[str, Any] = {
            "input_source": input_source,
            "depth_value_in_meters": depth_value_in_meters,
            "depth_scale": depth_scale,
            "intrinsics": dict(intrinsics or {}),
        }

        if input_source == "rgbd_files":
            input_payload["rgb_path"] = rgb_path or ""
            input_payload["depth_path"] = depth_path or ""
        elif input_source == "rgbd_arrays":
            if color_image is None or depth_image is None:
                raise ValueError("rgbd_arrays mode requires color_image and depth_image")
            input_payload["color_image"] = np.asarray(color_image)
            input_payload["depth_image"] = np.asarray(depth_image)
        elif input_source == "camera":
            pass
        else:
            raise ValueError(f"Unsupported input_source: {input_source}")

        t0 = time.perf_counter()
        if self.verbose:
            print(
                f"[DetectGrasp] rpc->detect (category={category}, input={input_source}, "
                f"mode={grasp_mode})"
            )
        detect_output = self._rpc(
            self._detect_proc,
            {
                "cmd": CMD_DETECT,
                "profile": profile,
                "input": input_payload,
                "grasp_mode": grasp_mode,
            },
            "detect",
            timeout_s=180.0,
        )
        t1 = time.perf_counter()

        method = detect_output.get("segmentation_method_used", "unknown")
        class_name = detect_output.get("selected_class_name", "unknown")
        conf = float(detect_output.get("selected_confidence", 0.0))
        npts = int(detect_output.get("selected_num_points", 0))
        detect_status = str(detect_output.get("status", "ok"))

        if detect_status != "ok":
            t2 = time.perf_counter()
            message = detect_output.get("message", "detect stage skipped")
            out = {
                "grasp_mode": grasp_mode,
                "status": detect_status,
                "message": message,
                "result_json_path": None,
                "best_grasp_pose": None,
                "best_grasp_conf": None,
                "timing": {
                    "detect_rpc_s": float(t1 - t0),
                    "grasp_rpc_s": 0.0,
                    "total_s": float(t2 - t0),
                    "detect_worker": detect_output.get("timing", {}),
                    "grasp_worker": {},
                },
                "result_payload": {
                    "mode": grasp_mode,
                    "status": detect_status,
                    "message": message,
                    "source": {
                        "selected_index": detect_output.get("selected_index"),
                        "num_candidates": int(detect_output.get("num_candidates", 0)),
                        "class_name": class_name,
                        "confidence": conf,
                        "bbox": np.asarray(detect_output.get("selected_bbox", [])).tolist(),
                        "intrinsics": detect_output.get("intrinsics", {}),
                        "policy": profile.get("target_object_policy", "top_conf"),
                        "input_source": detect_output.get("input_source", "unknown"),
                        "result_json_path": None,
                    },
                    "result": {},
                },
            }
            if self.verbose:
                print(
                    f"[DetectGrasp] detect skipped: status={detect_status}, "
                    f"message={message}"
                )
                timing = out["timing"]
                print(
                    f"[DetectGrasp] timing total={timing['total_s']:.3f}s "
                    f"(detect_rpc={timing['detect_rpc_s']:.3f}s)"
                )
            return out

        if grasp_mode == "tableware_pca":
            tableware_result = dict(detect_output.get("tableware_pca", {}))
            best_pose = tableware_result.get("transform_matrix")
            best_conf = detect_output.get("selected_confidence")
            t2 = time.perf_counter()

            result_payload = {
                "mode": "tableware_pca",
                "source": {
                    "selected_index": int(detect_output.get("selected_index", 0)),
                    "num_candidates": int(detect_output.get("num_candidates", 0)),
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": np.asarray(detect_output.get("selected_bbox", [])).tolist(),
                    "intrinsics": detect_output.get("intrinsics", {}),
                    "policy": profile.get("target_object_policy", "top_conf"),
                    "input_source": detect_output.get("input_source", "unknown"),
                    "result_json_path": None,
                },
                "result": tableware_result,
            }
            out = {
                "grasp_mode": "tableware_pca",
                "status": "ok",
                "message": "",
                "result_json_path": None,
                "best_grasp_pose": best_pose,
                "best_grasp_conf": best_conf,
                "timing": {
                    "detect_rpc_s": float(t1 - t0),
                    "grasp_rpc_s": 0.0,
                    "total_s": float(t2 - t0),
                    "detect_worker": detect_output.get("timing", {}),
                    "grasp_worker": {},
                },
                "result_payload": result_payload,
            }
            if self.verbose:
                timing = out["timing"]
                print(
                    f"[DetectGrasp] category={category} mode=tableware_pca method={method} "
                    f"class={class_name} conf={conf:.3f} points={npts}"
                )
                print(
                    f"[DetectGrasp] timing total={timing['total_s']:.3f}s "
                    f"(detect_rpc={timing['detect_rpc_s']:.3f}s)"
                )
            return out

        # 仅转发抓取端需要的字段，避免无效 IPC 负担。
        detect_to_grasp = {
            "input_source": detect_output.get("input_source"),
            "intrinsics": detect_output.get("intrinsics"),
            "selected_mask_packed": detect_output.get("selected_mask_packed"),
            "selected_mask_shape": detect_output.get("selected_mask_shape"),
            "selected_index": detect_output.get("selected_index"),
            "selected_class_name": detect_output.get("selected_class_name"),
            "selected_confidence": detect_output.get("selected_confidence"),
            "selected_bbox": detect_output.get("selected_bbox"),
            "num_candidates": detect_output.get("num_candidates", 0),
        }
        if "selected_mask" in detect_output:  # 兼容旧字段
            detect_to_grasp["selected_mask"] = detect_output.get("selected_mask")
        if "segmentation_method_used" in detect_output:
            detect_to_grasp["segmentation_method_used"] = detect_output.get(
                "segmentation_method_used"
            )
        if "selected_num_points" in detect_output:
            detect_to_grasp["selected_num_points"] = detect_output.get("selected_num_points")
        if detect_output.get("input_source") == "rgbd_files":
            detect_to_grasp["rgb_path"] = detect_output.get("rgb_path")
            detect_to_grasp["depth_path"] = detect_output.get("depth_path")
            detect_to_grasp["depth_value_in_meters"] = detect_output.get(
                "depth_value_in_meters", False
            )
            detect_to_grasp["depth_scale"] = detect_output.get("depth_scale", 1000.0)
        else:
            detect_to_grasp["color_image"] = detect_output.get("color_image")
            detect_to_grasp["depth_image"] = detect_output.get("depth_image")

        if self.verbose:
            print("[DetectGrasp] rpc->grasp")
        grasp_output = self._rpc(
            self._grasp_proc,
            {
                "cmd": CMD_GRASP,
                "profile": profile,
                "detect_output": detect_to_grasp,
                "frame_transforms": {
                    "camera_to_ee_transform": camera_to_ee_transform,
                    "ee_to_base_transform": ee_to_base_transform,
                },
            },
            "grasp",
            timeout_s=180.0,
        )
        t2 = time.perf_counter()
        grasp_output["grasp_mode"] = "graspgen"
        grasp_output.setdefault("status", "ok")
        grasp_output.setdefault("message", "")

        grasp_output["timing"] = {
            "detect_rpc_s": float(t1 - t0),
            "grasp_rpc_s": float(t2 - t1),
            "total_s": float(t2 - t0),
            "detect_worker": detect_output.get("timing", {}),
            "grasp_worker": grasp_output.get("timing", {}),
        }

        if self.verbose:
            best_conf = grasp_output.get("best_grasp_conf")
            timing = grasp_output["timing"]
            grasp_status = grasp_output.get("status", "ok")
            print(
                f"[DetectGrasp] category={category} mode=graspgen method={method} "
                f"class={class_name} conf={conf:.3f} points={npts} "
                f"best_conf={best_conf}"
            )
            if grasp_status != "ok":
                print(
                    f"[DetectGrasp] grasp skipped: status={grasp_status}, "
                    f"message={grasp_output.get('message', '')}"
                )
            print(
                f"[DetectGrasp] timing total={timing['total_s']:.3f}s "
                f"(detect_rpc={timing['detect_rpc_s']:.3f}s, "
                f"grasp_rpc={timing['grasp_rpc_s']:.3f}s)"
            )

        return grasp_output

    def stop(self) -> None:
        for name, proc in (("detect", self._detect_proc), ("grasp", self._grasp_proc)):
            if proc is None:
                continue
            try:
                if proc.poll() is None and proc.stdin is not None and proc.stdout is not None:
                    send_message(proc.stdin, {"cmd": CMD_QUIT})
                    _ = recv_message(proc.stdout)
                    proc.wait(timeout=10)
            except Exception:
                proc.kill()
        self._detect_proc = None
        self._grasp_proc = None

    def __enter__(self) -> "DetectGraspOrchestrator":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

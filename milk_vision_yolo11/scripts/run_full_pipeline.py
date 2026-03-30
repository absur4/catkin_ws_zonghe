#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(p)).absolute()


def now_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def q(x: Path) -> str:
    return shlex.quote(str(x))


def run_bash(command: str, desc: str, cwd: Optional[Path] = None) -> None:
    print(f"\n[STEP] {desc}")
    print(f"[CMD ] {command}\n")

    proc = subprocess.Popen(
        ["bash", "-lc", command],
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")

    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{desc} 失败，返回码: {ret}")


def find_single_file(directory: Path, pattern: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"在目录 {directory} 中未找到文件: {pattern}")
    if len(files) > 1:
        # 如果有多个，取最新修改时间的那个
        files = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def parse_summary(summary_path: Path) -> List[Dict]:
    """
    解析 run_yolo11_seg_demo.py 输出的 summary.txt
    行格式类似：
    instance 1: class=cell phone, conf=0.6600, box=[...], mask_area_pixels=12345, mask_path=/abs/path.png
    """
    if not summary_path.is_file():
        raise FileNotFoundError(f"summary.txt 不存在: {summary_path}")

    pattern = re.compile(
        r"^instance\s+(\d+):\s+class=(.*?),\s+conf=([0-9.]+),\s+box=\[(.*?)\],\s+mask_area_pixels=(\d+),\s+mask_path=(.*)$"
    )

    detections = []
    with open(summary_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = pattern.match(line)
            if not m:
                continue

            instance_id = int(m.group(1))
            cls_name = m.group(2).strip()
            conf = float(m.group(3))
            box_str = m.group(4).strip()
            area = int(m.group(5))
            mask_path = Path(m.group(6).strip())

            detections.append(
                {
                    "instance_id": instance_id,
                    "class_name": cls_name,
                    "conf": conf,
                    "box_str": box_str,
                    "mask_area_pixels": area,
                    "mask_path": mask_path,
                }
            )

    if not detections:
        raise RuntimeError(f"summary.txt 解析失败或没有检测结果: {summary_path}")

    return detections


def select_detection(
    detections: List[Dict],
    target_class: str,
    selection_rule: str,
    allow_fallback: bool,
) -> Dict:
    matches = [d for d in detections if d["class_name"] == target_class]

    if not matches:
        found_classes = sorted(set(d["class_name"] for d in detections))
        if not allow_fallback:
            raise RuntimeError(
                f"没有找到目标类别 '{target_class}'。\n"
                f"当前检测到的类别有: {found_classes}\n"
                f"你可以改 --target-class，或者加 --allow-fallback。"
            )
        print(
            f"[WARN] 未找到目标类别 '{target_class}'，将从全部检测结果中按规则 '{selection_rule}' 回退选择。"
        )
        matches = detections

    if selection_rule == "largest_area":
        matches = sorted(matches, key=lambda d: d["mask_area_pixels"], reverse=True)
    elif selection_rule == "highest_conf":
        matches = sorted(matches, key=lambda d: d["conf"], reverse=True)
    elif selection_rule == "first":
        matches = sorted(matches, key=lambda d: d["instance_id"])
    else:
        raise ValueError(f"未知 selection_rule: {selection_rule}")

    return matches[0]


def parse_position_result(position_result_txt: Path) -> List[float]:
    if not position_result_txt.is_file():
        raise FileNotFoundError(f"position_result.txt 不存在: {position_result_txt}")

    content = position_result_txt.read_text(encoding="utf-8")
    m = re.search(r"position_cam=\[([^\]]+)\]", content)
    if not m:
        raise RuntimeError(f"无法从 {position_result_txt} 中解析 position_cam")

    values = [float(x.strip()) for x in m.group(1).split(",")]
    if len(values) != 3:
        raise RuntimeError(f"position_cam 不是 3 个值: {values}")
    return values


def main() -> int:
    this_file = Path(__file__).resolve()
    script_dir = this_file.parent
    src_dir = this_file.parents[2]  # .../catkin_ws/src

    default_capture_script = src_dir / "l515_snapshot_tool" / "capture_l515_once.py"
    default_yolo_script = script_dir / "run_yolo11_seg_demo.py"
    default_estimate_script = script_dir / "estimate_position_from_mask_depth.py"
    default_base_outdir = script_dir.parent / "pipeline_runs"

    parser = argparse.ArgumentParser(
        description="一键执行: L515拍照 -> YOLO11分割 -> 目标mask筛选 -> 3D位置反推"
    )

    # 环境相关
    parser.add_argument("--ros-setup", default="/opt/ros/noetic/setup.bash")
    parser.add_argument("--ws-setup", default="~/catkin_ws/devel/setup.bash")
    parser.add_argument("--conda-sh", default="~/miniconda3/etc/profile.d/conda.sh")
    parser.add_argument("--conda-env", default="milk_yolo11")

    # 脚本路径
    parser.add_argument("--capture-script", default=str(default_capture_script))
    parser.add_argument("--yolo-script", default=str(default_yolo_script))
    parser.add_argument("--estimate-script", default=str(default_estimate_script))

    # 运行输出目录
    parser.add_argument("--base-outdir", default=str(default_base_outdir))
    parser.add_argument("--run-id", default="", help="可手动指定本次运行ID；默认按时间生成")

    # Capture 参数
    parser.add_argument("--color-topic", default="/camera/color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--launch-pkg", default="realsense2_camera")
    parser.add_argument("--launch-file", default="l515.launch")
    parser.add_argument("--timeout", type=float, default=40.0)
    parser.add_argument("--startup-delay", type=float, default=8.0)
    parser.add_argument("--skip-launch", action="store_true")

    # YOLO 参数
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)

    # 目标选择参数
    parser.add_argument(
        "--target-class",
        default="cell phone",
        help="要选取的目标类别名，当前你的测试目标可先用 'cell phone'"
    )
    parser.add_argument(
        "--selection-rule",
        default="largest_area",
        choices=["largest_area", "highest_conf", "first"],
        help="若同类目标有多个，如何选择"
    )
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="如果没找到 target-class，则允许从全部检测中按 selection-rule 回退选择"
    )

    # 相机内参 / 深度参数
    parser.add_argument("--fx", type=float, default=905.534912109375)
    parser.add_argument("--fy", type=float, default=905.7691650390625)
    parser.add_argument("--cx", type=float, default=656.93017578125)
    parser.add_argument("--cy", type=float, default=356.93890380859375)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    parser.add_argument("--min-depth-raw", type=int, default=300)
    parser.add_argument("--max-depth-raw", type=int, default=4000)

    args = parser.parse_args()

    ros_setup = expand_path(args.ros_setup)
    ws_setup = expand_path(args.ws_setup)
    conda_sh = expand_path(args.conda_sh)

    capture_script = expand_path(args.capture_script)
    yolo_script = expand_path(args.yolo_script)
    estimate_script = expand_path(args.estimate_script)

    base_outdir = expand_path(args.base_outdir)
    base_outdir.mkdir(parents=True, exist_ok=True)

    run_id = args.run_id.strip() if args.run_id.strip() else now_run_id()
    run_dir = base_outdir / run_id
    if run_dir.exists():
        raise FileExistsError(f"本次运行目录已存在，请换一个 --run-id: {run_dir}")

    capture_dir = run_dir / "capture"
    seg_dir = run_dir / "seg"
    position_dir = run_dir / "position"

    capture_dir.mkdir(parents=True, exist_ok=False)
    seg_dir.mkdir(parents=True, exist_ok=False)
    position_dir.mkdir(parents=True, exist_ok=False)

    for p in [ros_setup, ws_setup, conda_sh, capture_script, yolo_script, estimate_script]:
        if not p.exists():
            raise FileNotFoundError(f"路径不存在: {p}")

    try:
        # ---------------------------------------------------------------------
        # Step 1. 拍照
        # ---------------------------------------------------------------------
        capture_cmd = (
            f"source {q(ros_setup)} && "
            f"source {q(ws_setup)} && "
            f"python3 {q(capture_script)} "
            f"--output-dir {q(capture_dir)} "
            f"--color-topic {shlex.quote(args.color_topic)} "
            f"--depth-topic {shlex.quote(args.depth_topic)} "
            f"--launch-pkg {shlex.quote(args.launch_pkg)} "
            f"--launch-file {shlex.quote(args.launch_file)} "
            f"--timeout {args.timeout} "
            f"--startup-delay {args.startup_delay}"
        )
        if args.skip_launch:
            capture_cmd += " --skip-launch"

        run_bash(capture_cmd, "L515 拍照")

        # 找拍照输出
        captured_rgb = find_single_file(capture_dir, "*_color.png")
        captured_depth = find_single_file(capture_dir, "*_depth.png")
        captured_depth_npy = find_single_file(capture_dir, "*_depth.npy")
        captured_depth_vis = find_single_file(capture_dir, "*_depth_vis.png")

        # 生成稳定命名，方便下游接口使用
        scene_rgb = run_dir / "scene_rgb.png"
        scene_depth = run_dir / "scene_depth.png"
        scene_depth_npy = run_dir / "scene_depth.npy"
        scene_depth_vis = run_dir / "scene_depth_vis.png"

        copy_if_exists(captured_rgb, scene_rgb)
        copy_if_exists(captured_depth, scene_depth)
        copy_if_exists(captured_depth_npy, scene_depth_npy)
        copy_if_exists(captured_depth_vis, scene_depth_vis)

        print("\n[INFO] 已生成稳定输入文件:")
        print(f"       RGB   : {scene_rgb}")
        print(f"       Depth : {scene_depth}")
        print(f"       NPY   : {scene_depth_npy}")
        print(f"       Vis   : {scene_depth_vis}")

        # ---------------------------------------------------------------------
        # Step 2. YOLO 分割
        # ---------------------------------------------------------------------
        yolo_cmd = (
            f"source {q(conda_sh)} && "
            f"conda activate {shlex.quote(args.conda_env)} && "
            f"python {q(yolo_script)} "
            f"--image {q(scene_rgb)} "
            f"--outdir {q(seg_dir)} "
            f"--model {shlex.quote(args.model)} "
            f"--conf {args.conf} "
            f"--imgsz {args.imgsz}"
        )
        run_bash(yolo_cmd, "YOLO11 实例分割")

        summary_txt = seg_dir / "summary.txt"
        detections = parse_summary(summary_txt)

        print("\n[INFO] YOLO 检测结果摘要:")
        for d in detections:
            print(
                f"       instance={d['instance_id']}, "
                f"class={d['class_name']}, "
                f"conf={d['conf']:.4f}, "
                f"area={d['mask_area_pixels']}, "
                f"mask={d['mask_path']}"
            )

        selected = select_detection(
            detections=detections,
            target_class=args.target_class,
            selection_rule=args.selection_rule,
            allow_fallback=args.allow_fallback,
        )

        selected_mask = selected["mask_path"]
        if not selected_mask.is_file():
            raise FileNotFoundError(f"选中的 mask 文件不存在: {selected_mask}")

        target_mask = run_dir / "target_mask.png"
        copy_if_exists(selected_mask, target_mask)

        print("\n[INFO] 选中的目标实例:")
        print(f"       class          : {selected['class_name']}")
        print(f"       conf           : {selected['conf']:.4f}")
        print(f"       mask_area      : {selected['mask_area_pixels']}")
        print(f"       original_mask  : {selected_mask}")
        print(f"       target_mask    : {target_mask}")

        # ---------------------------------------------------------------------
        # Step 3. 3D 位置反推
        # ---------------------------------------------------------------------
        estimate_cmd = (
            f"source {q(conda_sh)} && "
            f"conda activate {shlex.quote(args.conda_env)} && "
            f"python {q(estimate_script)} "
            f"--rgb {q(scene_rgb)} "
            f"--depth {q(scene_depth)} "
            f"--mask {q(target_mask)} "
            f"--outdir {q(position_dir)} "
            f"--fx {args.fx} "
            f"--fy {args.fy} "
            f"--cx {args.cx} "
            f"--cy {args.cy} "
            f"--depth_scale {args.depth_scale} "
            f"--min_depth_raw {args.min_depth_raw} "
            f"--max_depth_raw {args.max_depth_raw}"
        )
        run_bash(estimate_cmd, "根据 mask + depth 反推 3D 位置")

        position_result_txt = position_dir / "position_result.txt"
        position_cam = parse_position_result(position_result_txt)

        final_result = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "scene_rgb": str(scene_rgb),
            "scene_depth": str(scene_depth),
            "target_class_requested": args.target_class,
            "selection_rule": args.selection_rule,
            "selected_detection": {
                "instance_id": selected["instance_id"],
                "class_name": selected["class_name"],
                "conf": selected["conf"],
                "mask_area_pixels": selected["mask_area_pixels"],
                "mask_path": str(selected_mask),
                "target_mask": str(target_mask),
            },
            "camera_intrinsics": {
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
                "depth_scale": args.depth_scale,
            },
            "position_cam": {
                "x": position_cam[0],
                "y": position_cam[1],
                "z": position_cam[2],
            },
            "position_result_txt": str(position_result_txt),
            "position_debug_png": str(position_dir / "position_debug.png"),
            "seg_summary_txt": str(summary_txt),
            "seg_overlay_jpg": str(seg_dir / "result_overlay.jpg"),
        }

        final_result_json = run_dir / "final_result.json"
        with open(final_result_json, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        latest_result_json = base_outdir / "latest_result.json"
        shutil.copy2(final_result_json, latest_result_json)

        print("\n" + "=" * 80)
        print("[FINAL RESULT]")
        print(
            f"position_cam = "
            f"[{position_cam[0]:.6f}, {position_cam[1]:.6f}, {position_cam[2]:.6f}]"
        )
        print(f"final_result_json = {final_result_json}")
        print(f"latest_result_json = {latest_result_json}")
        print("=" * 80)

        return 0

    except Exception as exc:
        print(f"\n[ERROR] 整体流程失败: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rospy

import grasp_only_vision as gov


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GRASPGEN_ROOT = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "piper_ros", "src", "GraspGen")
)


_VISER_VIEWER_CODE = r'''
import sys
import time
import numpy as np

npz_path = sys.argv[1]
port = int(sys.argv[2])
top_k = int(sys.argv[3])
graspgen_root = sys.argv[4]

if graspgen_root not in sys.path:
    sys.path.insert(0, graspgen_root)

from grasp_gen.utils.viser_utils import create_visualizer, make_frame, visualize_grasp, visualize_pointcloud


def _maybe_add_cloud(vis, name, points, colors, size):
    if points.size == 0:
        return
    visualize_pointcloud(vis, name, points, colors, size=size)


data = np.load(npz_path, allow_pickle=False)
scene_pc = np.asarray(data["scene_pc"], dtype=np.float32)
scene_color = np.asarray(data["scene_color"], dtype=np.uint8)
object_pc = np.asarray(data["object_pc"], dtype=np.float32)
object_color = np.asarray(data["object_color"], dtype=np.uint8)
grasp_poses = np.asarray(data["grasp_poses"], dtype=np.float64)
grasp_conf = np.asarray(data["grasp_conf"], dtype=np.float64)
best_pose = np.asarray(data["best_pose"], dtype=np.float64)
gripper_name = str(data["gripper_name"].item())

vis = create_visualizer(clear=True, port=port)
vis.scene.reset()

_maybe_add_cloud(vis, "scene_pc", scene_pc, scene_color, size=0.0025)
_maybe_add_cloud(vis, "object_pc", object_pc, object_color, size=0.004)
make_frame(vis, "camera_frame", h=0.12, radius=0.004, T=np.eye(4, dtype=np.float64))

if grasp_poses.ndim == 3 and grasp_poses.shape[0] > 0:
    order = np.argsort(-grasp_conf) if grasp_conf.size == grasp_poses.shape[0] else np.arange(grasp_poses.shape[0])
    top_idx = order[: max(1, min(int(top_k), grasp_poses.shape[0]))]
    best_idx = int(top_idx[0])
    for rank, idx in enumerate(top_idx):
        pose = np.asarray(grasp_poses[idx], dtype=np.float64)
        conf = float(grasp_conf[idx]) if grasp_conf.size == grasp_poses.shape[0] else float("nan")
        name = "best_grasp" if idx == best_idx else f"grasps/{rank:03d}"
        color = [0, 255, 0] if idx == best_idx else [255, 165, 0]
        width = 10.0 if idx == best_idx else 6.0
        print(f"[VISER] grasp rank={rank + 1} idx={idx} conf={conf:.4f}")
        visualize_grasp(vis, name, pose, color=color, gripper_name=gripper_name, linewidth=width)

if best_pose.shape == (4, 4):
    make_frame(vis, "best_grasp_frame", h=0.08, radius=0.003, T=best_pose)

print(f"[VISER] Ready at http://localhost:{port}")
while True:
    time.sleep(1.0)
'''


def _build_clouds(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Dict[str, Any],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    h, w = depth_m.shape

    u, v = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    z = depth_m.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_map = np.stack([x, y, z], axis=-1)

    depth_valid = z > 0
    object_valid = mask.astype(bool) & depth_valid

    scene_pc = points_map[depth_valid]
    scene_color = color_bgr[depth_valid][:, ::-1].astype(np.uint8)
    object_pc = points_map[object_valid]
    object_color = color_bgr[object_valid][:, ::-1].astype(np.uint8)
    return scene_pc, scene_color, object_pc, object_color


def _downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, colors
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx], colors[idx]


def _write_vis_payload(
    *,
    scene_pc: np.ndarray,
    scene_color: np.ndarray,
    object_pc: np.ndarray,
    object_color: np.ndarray,
    grasp_poses: np.ndarray,
    grasp_conf: np.ndarray,
    best_pose: np.ndarray,
    gripper_name: str,
) -> str:
    fd, payload_path = tempfile.mkstemp(prefix="grasp_only_viser_", suffix=".npz")
    os.close(fd)
    np.savez_compressed(
        payload_path,
        scene_pc=np.asarray(scene_pc, dtype=np.float32),
        scene_color=np.asarray(scene_color, dtype=np.uint8),
        object_pc=np.asarray(object_pc, dtype=np.float32),
        object_color=np.asarray(object_color, dtype=np.uint8),
        grasp_poses=np.asarray(grasp_poses, dtype=np.float64),
        grasp_conf=np.asarray(grasp_conf, dtype=np.float64),
        best_pose=np.asarray(best_pose, dtype=np.float64),
        gripper_name=np.array(gripper_name),
    )
    return payload_path


def _launch_viser_subprocess(
    *,
    conda_exe: str,
    grasp_env: str,
    payload_path: str,
    port: int,
    top_k: int,
) -> Tuple[Optional[subprocess.Popen], Optional[str]]:
    if not os.path.isdir(_GRASPGEN_ROOT):
        rospy.logwarn(f"GraspGen 根目录不存在，跳过 Viser 可视化: {_GRASPGEN_ROOT}")
        return None, None

    log_path = payload_path + ".log"
    cmd = [
        conda_exe,
        "run",
        "-n",
        grasp_env,
        "python",
        "-c",
        _VISER_VIEWER_CODE,
        payload_path,
        str(port),
        str(top_k),
        _GRASPGEN_ROOT,
    ]

    try:
        log_file = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        return proc, log_path
    except Exception as exc:
        rospy.logwarn(f"启动 Viser 可视化失败: {exc}")
        return None, log_path


def _maybe_launch_viser(
    *,
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Dict[str, Any],
    mask: np.ndarray,
    result_data: Dict[str, Any],
    conda_exe: str,
    grasp_env: str,
) -> None:
    enabled = bool(rospy.get_param("~visualization/enabled", True))
    if not enabled:
        return

    payload = dict(result_data.get("result_payload", {}))
    result_json = dict(payload.get("result", {}))
    grasp_poses = np.asarray(result_json.get("grasp_poses", []), dtype=np.float64)
    grasp_conf = np.asarray(result_json.get("grasp_conf", []), dtype=np.float64)
    best_pose_raw = result_data.get("best_grasp_pose")
    if best_pose_raw is None:
        rospy.logwarn("没有 best_grasp_pose，跳过 Viser 可视化")
        return

    best_pose = np.asarray(best_pose_raw, dtype=np.float64)
    gripper_name = str(result_json.get("gripper_name", "robotiq_2f_140"))

    scene_pc, scene_color, object_pc, object_color = _build_clouds(
        color_bgr,
        depth_m,
        intrinsics,
        mask,
    )
    if object_pc.shape[0] == 0:
        rospy.logwarn("目标点云为空，跳过 Viser 可视化")
        return

    rng = np.random.default_rng(int(rospy.get_param("~visualization/random_seed", 0)))
    max_scene_points = int(rospy.get_param("~visualization/max_scene_points", 120000))
    max_object_points = int(rospy.get_param("~visualization/max_object_points", 60000))
    top_k = int(rospy.get_param("~visualization/num_visualize_grasps", 20))
    port = int(rospy.get_param("~visualization/port", 8080))

    scene_pc, scene_color = _downsample_points(scene_pc, scene_color, max_scene_points, rng)
    object_pc, object_color = _downsample_points(object_pc, object_color, max_object_points, rng)

    payload_path = _write_vis_payload(
        scene_pc=scene_pc,
        scene_color=scene_color,
        object_pc=object_pc,
        object_color=object_color,
        grasp_poses=grasp_poses,
        grasp_conf=grasp_conf,
        best_pose=best_pose,
        gripper_name=gripper_name,
    )
    proc, log_path = _launch_viser_subprocess(
        conda_exe=conda_exe,
        grasp_env=grasp_env,
        payload_path=payload_path,
        port=port,
        top_k=top_k,
    )
    if proc is None:
        return

    rospy.loginfo(
        f"Viser 可视化已启动，访问 http://localhost:{port} "
        f"(pid={proc.pid}, log={log_path}, payload={payload_path})"
    )


def main() -> None:
    rospy.init_node("grasp_only_vision_viser")
    rospy.loginfo("========== Grasp Only Vision Viser ==========")

    if not rospy.has_param("~depth_topic"):
        rospy.set_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
    if not rospy.has_param("~require_depth"):
        rospy.set_param("~require_depth", True)

    vc = gov.VisionContext()

    conda_exe = rospy.get_param("~grasp/conda_exe", "/home/songfei/miniconda3/bin/conda")
    grasp_env = rospy.get_param("~grasp/grasp_env_name", "GraspGen")
    profile_bundle_path = rospy.get_param(
        "~grasp/profile_bundle_path",
        "/home/songfei/catkin_ws/src/piper_ros/src/piper/scripts/detect_grasp/profile_bundle.json",
    )
    gripper_config = rospy.get_param(
        "~grasp/gripper_config",
        "/home/songfei/catkin_ws/src/piper_ros/src/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml",
    )

    target_name = rospy.get_param("~target_name", "milk")
    confidence_threshold = rospy.get_param("~confidence_threshold", 0.3)
    rgbd_max_dt = float(rospy.get_param("~debug/rgbd_max_dt", 0.08))

    rospy.loginfo(f"RGB topic: {vc.rgb_topic}, Depth topic: {vc.depth_topic}")
    if not vc.wait_for_images(timeout_sec=15.0):
        rospy.logerr("未能获取相机图像，退出")
        return
    if not gov._wait_for_synced_rgbd(vc, timeout_sec=15.0, max_dt_sec=rgbd_max_dt):
        rospy.logerr(f"未能获取时间差小于 {rgbd_max_dt:.3f}s 的 RGB/Depth 图像，退出")
        return

    try:
        obj, mask, rgb_cv, depth_cv = gov._select_detection_with_mask(
            vc,
            target_name=target_name,
            confidence_threshold=confidence_threshold,
            rgbd_max_dt_sec=rgbd_max_dt,
        )
    except Exception as exc:
        rospy.logerr(f"目标检测/分割失败: {exc}")
        return

    if depth_cv.shape[:2] != rgb_cv.shape[:2]:
        rospy.logwarn(
            f"Depth/RGB size mismatch: depth={depth_cv.shape[:2]} rgb={rgb_cv.shape[:2]}, resizing depth"
        )
        depth_cv = cv2.resize(
            depth_cv,
            (rgb_cv.shape[1], rgb_cv.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    depth_m = depth_cv.astype(np.float32) * float(vc.depth_scale)

    if not obj.bbox:
        rospy.logerr("检测结果没有 bbox，无法构造 mask")
        return
    intr = gov._resolve_intrinsics(vc, rgb_cv.shape[1], rgb_cv.shape[0])
    rospy.loginfo(
        f"Grasp input summary: label={obj.class_name} conf={obj.confidence:.4f} "
        f"bbox={obj.bbox} intrinsics=(fx={intr['fx']:.3f}, fy={intr['fy']:.3f}, "
        f"cx={intr['cx']:.3f}, cy={intr['cy']:.3f})"
    )

    grasp_client = gov.GraspOnlyClient(
        conda_exe=conda_exe,
        grasp_env=grasp_env,
        profile_bundle_path=profile_bundle_path,
        gripper_config=gripper_config,
    )
    grasp_client.start()
    try:
        result = grasp_client.run_grasp(
            color_bgr=rgb_cv,
            depth_m=depth_m,
            intrinsics=intr,
            mask=mask,
            bbox=obj.bbox,
            class_name=obj.class_name,
            confidence=obj.confidence,
        )
    finally:
        grasp_client.stop()

    if not result.get("ok", False):
        rospy.logerr(f"抓取失败: {result}")
        return

    data = result.get("data", {})
    best_pose = data.get("best_grasp_pose")
    best_conf = data.get("best_grasp_conf")
    rospy.loginfo(f"best_grasp_conf: {best_conf}")
    if best_pose is None:
        rospy.logwarn("未生成抓取姿态")
        return

    rospy.loginfo(f"best_grasp_pose:\n{np.asarray(best_pose)}")
    _maybe_launch_viser(
        color_bgr=rgb_cv,
        depth_m=depth_m,
        intrinsics=intr,
        mask=mask,
        result_data=data,
        conda_exe=conda_exe,
        grasp_env=grasp_env,
    )

    enable_motion = rospy.get_param("~motion/enable", False)
    if not enable_motion:
        rospy.logwarn("motion/enable=false，未执行机械臂动作")
        return

    base_frame = rospy.get_param("~motion/base_frame", "base_link")
    camera_frame = rospy.get_param("~motion/camera_frame", vc.camera_frame_id)
    rospy.loginfo(f"vc.camera_frame_id: {vc.camera_frame_id}")
    rospy.loginfo(f"motion/camera_frame: {camera_frame}, motion/base_frame: {base_frame}")
    pregrasp_offset = float(rospy.get_param("~motion/pregrasp_offset", 0.10))
    gripper_open = float(rospy.get_param("~motion/gripper_open", 0.035))
    vel = float(rospy.get_param("~motion/max_velocity", 0.3))
    acc = float(rospy.get_param("~motion/max_acceleration", 0.3))

    grasp_T_cam = np.asarray(best_pose, dtype=np.float64)
    try:
        tf_stamped = vc.tf_buffer.lookup_transform(base_frame, camera_frame, rospy.Time(0), rospy.Duration(1.0))
        base_T_cam = gov._matrix_from_tf(tf_stamped)
    except Exception as exc:
        rospy.logwarn(f"TF 变换失败（{base_frame}<-{camera_frame}），使用相机坐标: {exc}")
        base_T_cam = np.eye(4, dtype=np.float64)

    base_T_grasp = base_T_cam @ grasp_T_cam
    rospy.loginfo(f"best_grasp_pose (base_link):\n{base_T_grasp}")
    pre_T = base_T_grasp.copy()
    pre_T[:3, 3] -= pregrasp_offset * pre_T[:3, 2]

    pre_endpose = gov._pose_from_matrix(pre_T)

    rospy.loginfo("执行机械臂抓取动作...")
    gov._call_joint_moveit_ctrl_gripper(gripper_open, max_velocity=vel, max_acceleration=acc)
    if not gov._call_joint_moveit_ctrl_endpose(pre_endpose, max_velocity=vel, max_acceleration=acc):
        rospy.logerr("移动到预抓取位失败")
        return
    rospy.loginfo("已移动到预抓取位，按当前配置不继续执行抓取位与闭爪动作")


if __name__ == "__main__":
    main()

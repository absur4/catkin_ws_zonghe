#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rospy
from std_srvs.srv import Trigger

import grasp_only_vision as gov
import grasp_only_vision_viser as gov_viser

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OPEN_VOCAB_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "open_vocabulary"))
if _OPEN_VOCAB_ROOT not in sys.path:
    sys.path.insert(0, _OPEN_VOCAB_ROOT)

try:
    from open_vocabulary.API.utils.plane_estimation import fit_plane_tls
except Exception:
    fit_plane_tls = None


def _depth_to_points(
    depth_m: np.ndarray,
    intrinsics: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
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
    depth_valid = z > 1e-6
    return points_map, depth_valid


def _sample_plane_points(
    depth_m: np.ndarray,
    intrinsics: Dict[str, Any],
    object_mask: np.ndarray,
    *,
    max_points: int,
    min_depth: float,
    max_depth: float,
    rng: np.random.Generator,
) -> np.ndarray:
    points_map, depth_valid = _depth_to_points(depth_m, intrinsics)
    if object_mask is None:
        bg_mask = depth_valid
    else:
        bg_mask = depth_valid & (~object_mask.astype(bool))
    if not np.any(bg_mask):
        return np.zeros((0, 3), dtype=np.float64)

    points = points_map[bg_mask]
    z = points[:, 2]
    depth_ok = (z > min_depth) & (z < max_depth)
    points = points[depth_ok]
    if points.shape[0] == 0:
        return points.astype(np.float64)

    if max_points > 0 and points.shape[0] > max_points:
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points.astype(np.float64)


def _fit_plane_ransac(
    points: np.ndarray,
    *,
    iters: int,
    inlier_thresh: float,
    min_inliers: int,
    rng: np.random.Generator,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    if points.shape[0] < 3:
        return None

    best_inliers = None
    best_count = 0
    best_plane = None
    n_points = points.shape[0]

    for _ in range(iters):
        idx = rng.choice(n_points, size=3, replace=False)
        p1, p2, p3 = points[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)
        if norm < 1e-8:
            continue
        n = n / norm
        d = -float(np.dot(n, p1))
        dist = np.abs(points @ n + d)
        inliers = dist <= inlier_thresh
        count = int(np.sum(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = (n, d)

    if best_plane is None or best_inliers is None:
        return None
    if best_count < min_inliers:
        return None

    inlier_points = points[best_inliers]
    if fit_plane_tls is not None and inlier_points.shape[0] >= 3:
        n_refined, d_refined = fit_plane_tls(inlier_points, np.array([0.0, 1.0, 0.0]))
        return n_refined, float(d_refined), inlier_points

    n, d = best_plane
    return n, float(d), inlier_points


def _select_best_grasp_above_plane(
    grasp_poses: np.ndarray,
    grasp_conf: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    object_centroid: np.ndarray,
    *,
    min_clearance: float,
    require_approach: bool,
    min_approach_dot: float,
    prefer_mode: str,
    prefer_dot: float,
) -> Optional[int]:
    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        return None
    if grasp_conf.ndim != 1 or grasp_conf.shape[0] != grasp_poses.shape[0]:
        return None

    n = plane_normal.astype(np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        return None
    n = n / n_norm
    d = float(plane_d)

    obj_side = float(np.dot(n, object_centroid) + d)
    if obj_side < 0.0:
        n = -n
        d = -d

    positions = grasp_poses[:, :3, 3]
    signed_dist = positions @ n + d
    above_mask = signed_dist > float(min_clearance)

    if require_approach:
        approach = grasp_poses[:, :3, 2]
        approach_dot = np.sum(approach * (-n[None, :]), axis=1)
        above_mask &= approach_dot >= float(min_approach_dot)

    if not np.any(above_mask):
        return None

    valid_idx = np.where(above_mask)[0]
    if prefer_mode:
        mode = str(prefer_mode).strip().lower()
    else:
        mode = "none"

    if mode in ("front", "side"):
        approach = grasp_poses[:, :3, 2]
        if mode == "front":
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        dot = np.abs(approach @ axis)
        prefer_mask = dot >= float(prefer_dot)
        prefer_idx = np.where(above_mask & prefer_mask)[0]
        if prefer_idx.size > 0:
            valid_idx = prefer_idx

    best_local = int(valid_idx[np.argmax(grasp_conf[valid_idx])])
    return best_local


def _normalize_category_name(name: Any) -> str:
    return str(name).strip().lower().replace(".", "")


def _select_best_grasp_by_direction(
    grasp_poses: np.ndarray,
    grasp_conf: np.ndarray,
    *,
    prefer_mode: str,
    prefer_dot: float,
) -> Optional[int]:
    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        return None
    if grasp_conf.ndim != 1 or grasp_conf.shape[0] != grasp_poses.shape[0]:
        return None
    if grasp_poses.shape[0] == 0:
        return None

    mode = str(prefer_mode).strip().lower()
    if mode == "front":
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    elif mode == "side":
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        return int(np.argmax(grasp_conf))

    approach = grasp_poses[:, :3, 2]
    alignment = np.abs(approach @ axis)
    preferred_idx = np.where(alignment >= float(prefer_dot))[0]
    valid_idx = preferred_idx if preferred_idx.size > 0 else np.arange(grasp_poses.shape[0])
    return int(valid_idx[np.argmax(grasp_conf[valid_idx])])


def _resolve_selection_strategy(
    *,
    target_name: str,
    detected_class_name: str,
    requested_strategy: str,
) -> str:
    strategy = str(requested_strategy).strip().lower()
    if strategy and strategy != "auto":
        return strategy

    category = _normalize_category_name(detected_class_name) or _normalize_category_name(target_name)
    if category == "milk":
        return "front"
    if category == "plate":
        return "top"
    return "top"


def _clamp_gripper_width(width_m: float, *, max_width_m: float, param_name: str) -> float:
    width = float(width_m)
    max_width = max(0.0, float(max_width_m))
    if width < 0.0:
        rospy.logwarn(f"{param_name}={width:.4f} < 0，已钳制到 0")
        return 0.0
    if width > max_width:
        rospy.logwarn(
            f"{param_name}={width:.4f} 超过夹爪最大开口 {max_width:.4f}，已钳制到最大值"
        )
        return max_width
    return width


def _rank_indices_by_conf(indices: np.ndarray, grasp_conf: np.ndarray) -> list[int]:
    if indices.size == 0:
        return []
    ordered = indices[np.argsort(-grasp_conf[indices])]
    return [int(idx) for idx in ordered]


def _rank_grasp_candidates_above_plane(
    grasp_poses: np.ndarray,
    grasp_conf: np.ndarray,
    plane_normal: np.ndarray,
    plane_d: float,
    object_centroid: np.ndarray,
    *,
    min_clearance: float,
    require_approach: bool,
    min_approach_dot: float,
    prefer_mode: str,
    prefer_dot: float,
) -> list[int]:
    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        return []
    if grasp_conf.ndim != 1 or grasp_conf.shape[0] != grasp_poses.shape[0]:
        return []

    n = plane_normal.astype(np.float64)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-8:
        return []
    n = n / n_norm
    d = float(plane_d)

    obj_side = float(np.dot(n, object_centroid) + d)
    if obj_side < 0.0:
        n = -n
        d = -d

    positions = grasp_poses[:, :3, 3]
    signed_dist = positions @ n + d
    above_mask = signed_dist > float(min_clearance)

    if require_approach:
        approach = grasp_poses[:, :3, 2]
        approach_dot = np.sum(approach * (-n[None, :]), axis=1)
        above_mask &= approach_dot >= float(min_approach_dot)

    valid_idx = np.where(above_mask)[0]
    if valid_idx.size == 0:
        return []

    mode = str(prefer_mode).strip().lower()
    if mode not in ("front", "side"):
        return _rank_indices_by_conf(valid_idx, grasp_conf)

    approach = grasp_poses[:, :3, 2]
    axis = np.array([0.0, 0.0, 1.0] if mode == "front" else [1.0, 0.0, 0.0], dtype=np.float64)
    alignment = np.abs(approach @ axis)
    preferred_idx = valid_idx[alignment[valid_idx] >= float(prefer_dot)]
    remaining_idx = valid_idx[alignment[valid_idx] < float(prefer_dot)]
    return _rank_indices_by_conf(preferred_idx, grasp_conf) + _rank_indices_by_conf(remaining_idx, grasp_conf)


def _rank_grasp_candidates_by_direction(
    grasp_poses: np.ndarray,
    grasp_conf: np.ndarray,
    *,
    prefer_mode: str,
    prefer_dot: float,
) -> list[int]:
    if grasp_poses.ndim != 3 or grasp_poses.shape[1:] != (4, 4):
        return []
    if grasp_conf.ndim != 1 or grasp_conf.shape[0] != grasp_poses.shape[0]:
        return []
    if grasp_poses.shape[0] == 0:
        return []

    all_idx = np.arange(grasp_poses.shape[0])
    mode = str(prefer_mode).strip().lower()
    if mode not in ("front", "side"):
        return _rank_indices_by_conf(all_idx, grasp_conf)

    approach = grasp_poses[:, :3, 2]
    axis = np.array([0.0, 0.0, 1.0] if mode == "front" else [1.0, 0.0, 0.0], dtype=np.float64)
    alignment = np.abs(approach @ axis)
    preferred_idx = all_idx[alignment >= float(prefer_dot)]
    remaining_idx = all_idx[alignment < float(prefer_dot)]
    return _rank_indices_by_conf(preferred_idx, grasp_conf) + _rank_indices_by_conf(remaining_idx, grasp_conf)


def _lookup_frame_transform(
    tf_buffer,
    *,
    target_frame: str,
    source_frame: str,
    timeout_sec: float,
) -> Optional[np.ndarray]:
    try:
        tf_stamped = tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            rospy.Time(0),
            rospy.Duration(timeout_sec),
        )
        return gov._matrix_from_tf(tf_stamped)
    except Exception as exc:
        rospy.logwarn(
            f"TF 读取失败（{target_frame}<-{source_frame}），跳过卡住检测: {exc}"
        )
        return None


def _transform_delta(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    translation_delta = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    relative_rot = before[:3, :3].T @ after[:3, :3]
    trace_val = float(np.trace(relative_rot))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace_val - 1.0)))
    rotation_delta_deg = float(np.rad2deg(np.arccos(cos_theta)))
    return translation_delta, rotation_delta_deg


def _call_stop_arm_motion(service_name: str) -> bool:
    try:
        rospy.wait_for_service(service_name, timeout=1.0)
        srv = rospy.ServiceProxy(service_name, Trigger)
        resp = srv()
        if not resp.success:
            rospy.logwarn(f"停止机械臂运动失败: {resp.message}")
        return bool(resp.success)
    except Exception as exc:
        rospy.logwarn(f"调用停止机械臂服务失败: {exc}")
        return False


def _call_endpose_with_stuck_detection(
    *,
    endpose: list[float],
    max_velocity: float,
    max_acceleration: float,
    tf_buffer,
    base_frame: str,
    monitor_frame: str,
    stuck_timeout_sec: float,
    translation_epsilon_m: float,
    rotation_epsilon_deg: float,
    stop_service_name: str,
) -> tuple[bool, bool]:
    if stuck_timeout_sec <= 0.0:
        return gov._call_joint_moveit_ctrl_endpose(
            endpose,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        ), False

    before = _lookup_frame_transform(
        tf_buffer,
        target_frame=base_frame,
        source_frame=monitor_frame,
        timeout_sec=1.0,
    )
    if before is None:
        return gov._call_joint_moveit_ctrl_endpose(
            endpose,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        ), False

    call_state = {"done": False, "result": False}

    def _worker() -> None:
        call_state["result"] = gov._call_joint_moveit_ctrl_endpose(
            endpose,
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
        )
        call_state["done"] = True

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    deadline = time.monotonic() + float(stuck_timeout_sec)
    while worker.is_alive() and time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(0.05)

    if worker.is_alive():
        after = _lookup_frame_transform(
            tf_buffer,
            target_frame=base_frame,
            source_frame=monitor_frame,
            timeout_sec=0.5,
        )
        if after is not None:
            translation_delta, rotation_delta_deg = _transform_delta(before, after)
            rospy.loginfo(
                "motion monitor: "
                f"frame={monitor_frame} waited={stuck_timeout_sec:.2f}s "
                f"translation_delta={translation_delta:.4f}m rotation_delta={rotation_delta_deg:.2f}deg"
            )
            if (
                translation_delta <= float(translation_epsilon_m)
                and rotation_delta_deg <= float(rotation_epsilon_deg)
            ):
                rospy.logwarn(
                    f"检测到 {monitor_frame} 相对 {base_frame} 在 {stuck_timeout_sec:.1f}s 内几乎未移动，放弃当前抓姿"
                )
                _call_stop_arm_motion(stop_service_name)
                worker.join(timeout=2.0)
                return False, True

    worker.join()
    return bool(call_state["result"]), False


def _rot_z(theta_rad: float) -> np.ndarray:
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_gripper_base_offset(offset_m: float, rot_z_deg: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[2, 3] = float(offset_m)
    r = np.eye(4, dtype=np.float64)
    r[:3, :3] = _rot_z(np.deg2rad(float(rot_z_deg)))
    return t @ r


def _normalize_joint_list(value: Any, expected_len: int = 6) -> list[float]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            rospy.logwarn("home_joints 解析失败，使用零位")
            value = []
    if not isinstance(value, (list, tuple)):
        rospy.logwarn("home_joints 类型无效，使用零位")
        value = []
    joints = [float(item) for item in value]
    if len(joints) < expected_len:
        joints = joints + [0.0] * (expected_len - len(joints))
    elif len(joints) > expected_len:
        joints = joints[:expected_len]
    return joints


def _call_joint_moveit_ctrl_arm(
    joint_states: list[float],
    *,
    max_velocity: float,
    max_acceleration: float,
    service_name: str = "joint_moveit_ctrl_arm",
    gripper: float = 0.0,
) -> bool:
    rospy.wait_for_service(service_name)
    try:
        srv = rospy.ServiceProxy(service_name, gov.JointMoveitCtrl)
        req = gov.JointMoveitCtrlRequest()
        req.joint_states = list(joint_states)
        req.gripper = float(gripper)
        req.max_velocity = max_velocity
        req.max_acceleration = max_acceleration
        resp = srv(req)
        return bool(resp.status)
    except Exception as exc:
        rospy.logerr(f"arm 服务调用失败: {exc}")
        return False


def main() -> None:
    rospy.init_node("grasp_vision_table_up_goal_max")
    rospy.loginfo("========== Grasp Vision Table Up Goal Max ==========")

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
    if best_pose is None:
        rospy.logwarn("未生成抓取姿态")
        return

    payload = dict(data.get("result_payload", {}))
    result_json = dict(payload.get("result", {}))
    grasp_poses = np.asarray(result_json.get("grasp_poses", []), dtype=np.float64)
    grasp_conf = np.asarray(result_json.get("grasp_conf", []), dtype=np.float64)

    rng = np.random.default_rng(int(rospy.get_param("~plane/random_seed", 0)))
    plane_max_points = int(rospy.get_param("~plane/max_points", 60000))
    plane_min_depth = float(rospy.get_param("~plane/min_depth", 0.15))
    plane_max_depth = float(rospy.get_param("~plane/max_depth", 3.0))
    ransac_iters = int(rospy.get_param("~plane/ransac_iters", 200))
    inlier_thresh = float(rospy.get_param("~plane/inlier_thresh", 0.01))
    min_inliers = int(rospy.get_param("~plane/min_inliers", 800))
    min_clearance = float(rospy.get_param("~plane/min_clearance", 0.005))
    require_approach = bool(rospy.get_param("~plane/require_approach", False))
    min_approach_dot = float(rospy.get_param("~plane/min_approach_dot", 0.2))
    selection_strategy_param = str(rospy.get_param("~selection/strategy", "auto"))
    prefer_mode = str(rospy.get_param("~selection/prefer_mode", "none"))
    prefer_dot = float(rospy.get_param("~selection/prefer_dot", 0.6))

    selection_strategy = _resolve_selection_strategy(
        target_name=target_name,
        detected_class_name=str(obj.class_name),
        requested_strategy=selection_strategy_param,
    )
    normalized_prefer_mode = str(prefer_mode).strip().lower()
    if selection_strategy == "front" and normalized_prefer_mode not in ("front", "side"):
        effective_prefer_mode = "front"
    else:
        effective_prefer_mode = normalized_prefer_mode or "none"

    rospy.loginfo(
        "grasp selection: "
        f"requested_strategy={selection_strategy_param} effective_strategy={selection_strategy} "
        f"detected_class={_normalize_category_name(obj.class_name)} prefer_mode={effective_prefer_mode}"
    )

    plane_points = _sample_plane_points(
        depth_m,
        intr,
        mask,
        max_points=plane_max_points,
        min_depth=plane_min_depth,
        max_depth=plane_max_depth,
        rng=rng,
    )

    candidate_indices: list[int] = []
    if selection_strategy == "front":
        candidate_indices = _rank_grasp_candidates_by_direction(
            grasp_poses,
            grasp_conf,
            prefer_mode=effective_prefer_mode,
            prefer_dot=prefer_dot,
        )
        if candidate_indices:
            rospy.loginfo(
                f"前抓优先候选数: {len(candidate_indices)}，最高分 idx={candidate_indices[0]} conf={float(grasp_conf[candidate_indices[0]]):.4f}"
            )
    else:
        if plane_points.shape[0] >= 3:
            plane_fit = _fit_plane_ransac(
                plane_points,
                iters=ransac_iters,
                inlier_thresh=inlier_thresh,
                min_inliers=min_inliers,
                rng=rng,
            )
        else:
            plane_fit = None

        if plane_fit is None:
            rospy.logwarn("桌面平面拟合失败，退化为按抓取分数排序")
            candidate_indices = _rank_indices_by_conf(np.arange(grasp_poses.shape[0]), grasp_conf)
        else:
            n, d, inliers = plane_fit
            rospy.loginfo(
                f"plane fit: inliers={inliers.shape[0]} n={n.tolist()} d={d:.4f}"
            )
            if hasattr(obj, "pose") and hasattr(obj.pose, "position"):
                centroid = np.array(
                    [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z],
                    dtype=np.float64,
                )
            elif hasattr(obj, "centroid"):
                centroid = np.array([obj.centroid.x, obj.centroid.y, obj.centroid.z], dtype=np.float64)
            else:
                centroid = np.zeros(3, dtype=np.float64)
            candidate_indices = _rank_grasp_candidates_above_plane(
                grasp_poses,
                grasp_conf,
                n,
                d,
                centroid,
                min_clearance=min_clearance,
                require_approach=require_approach,
                min_approach_dot=min_approach_dot,
                prefer_mode=effective_prefer_mode,
                prefer_dot=prefer_dot,
            )
            rospy.loginfo(f"桌面上方抓姿候选数: {len(candidate_indices)}")

    if not candidate_indices and best_pose is not None:
        best_idx = int(np.argmax(grasp_conf)) if grasp_conf.size > 0 else -1
        if best_idx >= 0:
            candidate_indices = [best_idx]
            rospy.logwarn("候选抓姿为空，退化为使用分数最高的抓姿")

    if not candidate_indices:
        rospy.logerr("未找到可执行的抓姿候选，退出")
        return

    tool_depth = float(rospy.get_param("~tool/depth_m", 0.195))
    tool_offset_ratio = float(rospy.get_param("~tool/offset_ratio", 0.5))
    tool_offset_m_param = rospy.get_param("~tool/offset_m", None)
    tool_offset_m = float(tool_offset_m_param) if tool_offset_m_param is not None else tool_depth * tool_offset_ratio
    tool_rot_z_deg = float(rospy.get_param("~tool/rot_z_deg", -90.0))
    use_tool_offset = bool(rospy.get_param("~tool/enable_offset", True))
    grasp_to_gripper = _build_gripper_base_offset(tool_offset_m, tool_rot_z_deg) if use_tool_offset else np.eye(4, dtype=np.float64)
    cam_T_gripper = chosen_pose @ grasp_to_gripper

    vis_data = dict(data)
    vis_data["best_grasp_pose"] = cam_T_gripper
    if chosen_conf is not None:
        vis_data["best_grasp_conf"] = float(chosen_conf)
    gov_viser._maybe_launch_viser(
        color_bgr=rgb_cv,
        depth_m=depth_m,
        intrinsics=intr,
        mask=mask,
        result_data=vis_data,
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
    gripper_max_width = float(rospy.get_param("~motion/gripper_max_width", 0.08))
    gripper_open_each_side = rospy.get_param("~motion/gripper_open_each_side", None)
    if gripper_open_each_side is not None:
        requested_gripper_open = 2.0 * float(gripper_open_each_side)
        rospy.loginfo(
            f"使用单侧夹爪开口 motion/gripper_open_each_side={float(gripper_open_each_side):.4f}m，"
            f"换算总开口={requested_gripper_open:.4f}m"
        )
    else:
        requested_gripper_open = float(rospy.get_param("~motion/gripper_open", 0.08))
    gripper_open = _clamp_gripper_width(
        requested_gripper_open,
        max_width_m=gripper_max_width,
        param_name="motion/gripper_open",
    )
    gripper_close = _clamp_gripper_width(
        float(rospy.get_param("~motion/gripper_close", 0.01)),
        max_width_m=gripper_max_width,
        param_name="motion/gripper_close",
    )
    vel = float(rospy.get_param("~motion/max_velocity", 0.3))
    acc = float(rospy.get_param("~motion/max_acceleration", 0.3))
    execute_grasp = bool(rospy.get_param("~motion/execute_grasp", True))
    return_home = bool(rospy.get_param("~motion/return_home", True))
    stuck_detection_enable = bool(rospy.get_param("~motion/stuck_detection_enable", True))
    stuck_timeout_sec = float(rospy.get_param("~motion/stuck_timeout_sec", 3.0))
    stuck_monitor_frame = str(rospy.get_param("~motion/stuck_monitor_frame", "gripper_base"))
    stuck_translation_epsilon_m = float(rospy.get_param("~motion/stuck_translation_epsilon_m", 0.002))
    stuck_rotation_epsilon_deg = float(rospy.get_param("~motion/stuck_rotation_epsilon_deg", 2.0))
    stop_service_name = str(rospy.get_param("~motion/stop_service", "joint_moveit_stop_arm"))

    try:
        tf_stamped = vc.tf_buffer.lookup_transform(base_frame, camera_frame, rospy.Time(0), rospy.Duration(1.0))
        base_T_cam = gov._matrix_from_tf(tf_stamped)
    except Exception as exc:
        rospy.logwarn(f"TF 变换失败（{base_frame}<-{camera_frame}），使用相机坐标: {exc}")
        base_T_cam = np.eye(4, dtype=np.float64)

    success_idx = None
    success_base_T_grasp = None
    success_grasp_conf = None
    for rank, idx in enumerate(candidate_indices, start=1):
        candidate_pose = np.asarray(grasp_poses[idx], dtype=np.float64)
        candidate_conf = float(grasp_conf[idx])
        grasp_to_gripper = _build_gripper_base_offset(tool_offset_m, tool_rot_z_deg) if use_tool_offset else np.eye(4, dtype=np.float64)
        cam_T_gripper = candidate_pose @ grasp_to_gripper

        vis_data = dict(data)
        vis_data["best_grasp_pose"] = cam_T_gripper
        vis_data["best_grasp_conf"] = candidate_conf
        gov_viser._maybe_launch_viser(
            color_bgr=rgb_cv,
            depth_m=depth_m,
            intrinsics=intr,
            mask=mask,
            result_data=vis_data,
            conda_exe=conda_exe,
            grasp_env=grasp_env,
        )

        base_T_grasp = base_T_cam @ cam_T_gripper
        rospy.loginfo(
            f"尝试抓姿 {rank}/{len(candidate_indices)} idx={idx} conf={candidate_conf:.4f}"
        )
        rospy.loginfo(f"selected_grasp_pose (base_link):\n{base_T_grasp}")
        pre_T = base_T_grasp.copy()
        pre_T[:3, 3] -= pregrasp_offset * pre_T[:3, 2]
        pre_endpose = gov._pose_from_matrix(pre_T)
        grasp_endpose = gov._pose_from_matrix(base_T_grasp)

        rospy.loginfo("执行机械臂抓取动作...")
        gov._call_joint_moveit_ctrl_gripper(gripper_open, max_velocity=vel, max_acceleration=acc)
        if stuck_detection_enable:
            pre_ok, pre_stuck = _call_endpose_with_stuck_detection(
                endpose=pre_endpose,
                max_velocity=vel,
                max_acceleration=acc,
                tf_buffer=vc.tf_buffer,
                base_frame=base_frame,
                monitor_frame=stuck_monitor_frame,
                stuck_timeout_sec=stuck_timeout_sec,
                translation_epsilon_m=stuck_translation_epsilon_m,
                rotation_epsilon_deg=stuck_rotation_epsilon_deg,
                stop_service_name=stop_service_name,
            )
        else:
            pre_ok = gov._call_joint_moveit_ctrl_endpose(pre_endpose, max_velocity=vel, max_acceleration=acc)
            pre_stuck = False
        if not pre_ok:
            rospy.logwarn(
                f"候选抓姿 idx={idx} 预抓取失败{'（3秒未动）' if pre_stuck else ''}，尝试下一条"
            )
            continue
        rospy.loginfo("已移动到预抓取位")
        if not execute_grasp:
            rospy.logwarn("motion/execute_grasp=false，跳过抓取位与闭爪动作")
            return

        if stuck_detection_enable:
            grasp_ok, grasp_stuck = _call_endpose_with_stuck_detection(
                endpose=grasp_endpose,
                max_velocity=vel,
                max_acceleration=acc,
                tf_buffer=vc.tf_buffer,
                base_frame=base_frame,
                monitor_frame=stuck_monitor_frame,
                stuck_timeout_sec=stuck_timeout_sec,
                translation_epsilon_m=stuck_translation_epsilon_m,
                rotation_epsilon_deg=stuck_rotation_epsilon_deg,
                stop_service_name=stop_service_name,
            )
        else:
            grasp_ok = gov._call_joint_moveit_ctrl_endpose(grasp_endpose, max_velocity=vel, max_acceleration=acc)
            grasp_stuck = False
        if not grasp_ok:
            rospy.logwarn(
                f"候选抓姿 idx={idx} 抓取位失败{'（3秒未动）' if grasp_stuck else ''}，尝试下一条"
            )
            continue

        gov._call_joint_moveit_ctrl_gripper(gripper_close, max_velocity=vel, max_acceleration=acc)
        rospy.loginfo("已移动到抓取位并闭爪")
        success_idx = idx
        success_base_T_grasp = base_T_grasp
        success_grasp_conf = candidate_conf
        break

    if success_idx is None or success_base_T_grasp is None:
        rospy.logerr("所有抓姿候选都执行失败，程序退出")
        return

    base_T_grasp = success_base_T_grasp
    chosen_conf = success_grasp_conf
    if return_home:
        retreat_offset = float(rospy.get_param("~motion/retreat_offset", 0.05))
        home_service = rospy.get_param("~motion/home_service", "joint_moveit_ctrl_arm")
        home_joints_raw = rospy.get_param("~motion/home_joints", [0.0] * 6)
        home_joints = _normalize_joint_list(home_joints_raw, expected_len=6)
        home_gripper = float(rospy.get_param("~motion/home_gripper", gripper_close))
        rospy.loginfo("抓取完成，先抬起再回零位...")
        retreat_T = base_T_grasp.copy()
        retreat_T[:3, 3] -= retreat_offset * base_T_grasp[:3, 2]
        retreat_pose = gov._pose_from_matrix(retreat_T)
        if not gov._call_joint_moveit_ctrl_endpose(retreat_pose, max_velocity=vel, max_acceleration=acc):
            rospy.logwarn("抬起失败，尝试回零位")
        if not _call_joint_moveit_ctrl_arm(
            home_joints,
            max_velocity=vel,
            max_acceleration=acc,
            service_name=str(home_service),
            gripper=home_gripper if str(home_service) == "joint_moveit_ctrl_piper" else 0.0,
        ):
            rospy.logwarn("回零位失败")
        if str(home_service) != "joint_moveit_ctrl_piper":
            gov._call_joint_moveit_ctrl_gripper(home_gripper, max_velocity=vel, max_acceleration=acc)


if __name__ == "__main__":
    main()

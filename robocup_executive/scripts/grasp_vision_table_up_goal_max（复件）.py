#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import os
import sys
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import rospy

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
    prefer_mode = str(rospy.get_param("~selection/prefer_mode", "none"))
    prefer_dot = float(rospy.get_param("~selection/prefer_dot", 0.6))

    plane_points = _sample_plane_points(
        depth_m,
        intr,
        mask,
        max_points=plane_max_points,
        min_depth=plane_min_depth,
        max_depth=plane_max_depth,
        rng=rng,
    )

    chosen_pose = np.asarray(best_pose, dtype=np.float64)
    chosen_conf = float(best_conf) if best_conf is not None else None

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
        rospy.logwarn("桌面平面拟合失败，使用原始 best_grasp_pose")
    else:
        n, d, inliers = plane_fit
        rospy.loginfo(
            f"plane fit: inliers={inliers.shape[0]} n={n.tolist()} d={d:.4f}"
        )
        centroid = None
        if hasattr(obj, "pose") and hasattr(obj.pose, "position"):
            centroid = np.array(
                [obj.pose.position.x, obj.pose.position.y, obj.pose.position.z],
                dtype=np.float64,
            )
        elif hasattr(obj, "centroid"):
            centroid = np.array([obj.centroid.x, obj.centroid.y, obj.centroid.z], dtype=np.float64)
        else:
            centroid = np.zeros(3, dtype=np.float64)
        idx = _select_best_grasp_above_plane(
            grasp_poses,
            grasp_conf,
            n,
            d,
            centroid,
            min_clearance=min_clearance,
            require_approach=require_approach,
            min_approach_dot=min_approach_dot,
            prefer_mode=prefer_mode,
            prefer_dot=prefer_dot,
        )
        if idx is None:
            rospy.logwarn("未找到桌面上方的抓姿，使用原始 best_grasp_pose")
        else:
            chosen_pose = np.asarray(grasp_poses[idx], dtype=np.float64)
            chosen_conf = float(grasp_conf[idx])
            rospy.loginfo(
                f"选择桌面上方抓姿 idx={idx} conf={chosen_conf:.4f}"
            )

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
    gripper_open = float(rospy.get_param("~motion/gripper_open", 0.035))
    gripper_close = float(rospy.get_param("~motion/gripper_close", 0.01))
    vel = float(rospy.get_param("~motion/max_velocity", 0.3))
    acc = float(rospy.get_param("~motion/max_acceleration", 0.3))
    execute_grasp = bool(rospy.get_param("~motion/execute_grasp", True))
    return_home = bool(rospy.get_param("~motion/return_home", True))

    grasp_T_cam = np.asarray(cam_T_gripper, dtype=np.float64)
    try:
        tf_stamped = vc.tf_buffer.lookup_transform(base_frame, camera_frame, rospy.Time(0), rospy.Duration(1.0))
        base_T_cam = gov._matrix_from_tf(tf_stamped)
    except Exception as exc:
        rospy.logwarn(f"TF 变换失败（{base_frame}<-{camera_frame}），使用相机坐标: {exc}")
        base_T_cam = np.eye(4, dtype=np.float64)

    base_T_grasp = base_T_cam @ grasp_T_cam
    rospy.loginfo(f"selected_grasp_pose (base_link):\n{base_T_grasp}")
    pre_T = base_T_grasp.copy()
    pre_T[:3, 3] -= pregrasp_offset * pre_T[:3, 2]

    pre_endpose = gov._pose_from_matrix(pre_T)
    grasp_endpose = gov._pose_from_matrix(base_T_grasp)

    rospy.loginfo("执行机械臂抓取动作...")
    gov._call_joint_moveit_ctrl_gripper(gripper_open, max_velocity=vel, max_acceleration=acc)
    if not gov._call_joint_moveit_ctrl_endpose(pre_endpose, max_velocity=vel, max_acceleration=acc):
        rospy.logerr("移动到预抓取位失败")
        return
    rospy.loginfo("已移动到预抓取位")
    if not execute_grasp:
        rospy.logwarn("motion/execute_grasp=false，跳过抓取位与闭爪动作")
        return
    if not gov._call_joint_moveit_ctrl_endpose(grasp_endpose, max_velocity=vel, max_acceleration=acc):
        rospy.logerr("移动到抓取位失败")
        return
    gov._call_joint_moveit_ctrl_gripper(gripper_close, max_velocity=vel, max_acceleration=acc)
    rospy.loginfo("已移动到抓取位并闭爪")
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

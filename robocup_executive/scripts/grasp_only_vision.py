#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
import tf2_ros
from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlRequest
from tf.transformations import quaternion_from_matrix

# 复用 pick_place_run_vision 中的检测逻辑
PICK_PLACE_PATH = os.path.join(
    os.path.dirname(__file__),
    "pick_place_run_vision.py",
)
if os.path.exists(PICK_PLACE_PATH):
    sys.path.insert(0, os.path.dirname(PICK_PLACE_PATH))
try:
    from pick_place_run_vision import VisionContext, MockDetectedObject  # noqa: F401
except Exception as exc:
    raise RuntimeError(
        "无法导入 pick_place_run_vision.py 中的 VisionContext。"
        "请确认该文件存在且可被 Python 导入。"
    ) from exc

# detect_grasp 抓取端 IPC
DETECT_GRASP_DIR = "/home/songfei/catkin_ws/src/piper_ros/src/piper/scripts/detect_grasp"
if DETECT_GRASP_DIR not in sys.path:
    sys.path.insert(0, DETECT_GRASP_DIR)
from ipc import send_message, recv_message  # noqa: E402
from protocol import CMD_INIT, CMD_GRASP, CMD_QUIT  # noqa: E402
from profile_registry import ProfileRegistry  # noqa: E402


def _bbox_to_mask(bbox: List[int], shape_hw: Tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = shape_hw
    x1 = max(0, min(int(x1), w - 1))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h - 1))
    y2 = max(0, min(int(y2), h))
    mask = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def _normalize_label(label: str) -> str:
    return str(label).lower().strip().replace('.', '')


def _clip_bbox(bbox: List[float], shape_hw: Tuple[int, int]) -> List[int]:
    x1, y1, x2, y2 = bbox
    h, w = shape_hw
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    return [x1, y1, x2, y2]


def _msg_stamp_sec(msg: Optional[Image]) -> Optional[float]:
    if msg is None:
        return None
    try:
        return float(msg.header.stamp.to_sec())
    except Exception:
        return None


def _wait_for_synced_rgbd(vc: VisionContext, timeout_sec: float = 15.0, max_dt_sec: float = 0.08) -> bool:
    deadline = rospy.Time.now() + rospy.Duration(timeout_sec)
    rate = rospy.Rate(30)
    while rospy.Time.now() < deadline and not rospy.is_shutdown():
        rgb_msg = vc.rgb_image
        depth_msg = vc.depth_image
        if rgb_msg is None or depth_msg is None:
            rate.sleep()
            continue
        rgb_t = _msg_stamp_sec(rgb_msg)
        depth_t = _msg_stamp_sec(depth_msg)
        if rgb_t is None or depth_t is None:
            return True
        if abs(rgb_t - depth_t) <= max_dt_sec:
            return True
        rate.sleep()
    return False


def _select_detection_with_mask(
    vc: VisionContext,
    *,
    target_name: str,
    confidence_threshold: float,
    rgbd_max_dt_sec: float,
) -> Tuple[MockDetectedObject, np.ndarray, np.ndarray, np.ndarray]:
    if vc.gsam_api is None:
        raise RuntimeError("GroundingSAM API 不可用，无法生成 SAM mask")

    if not _wait_for_synced_rgbd(vc, timeout_sec=15.0, max_dt_sec=rgbd_max_dt_sec):
        raise RuntimeError("未能获取时间接近的 RGB/Depth 图像")

    if vc.rgb_image is None or vc.depth_image is None:
        raise RuntimeError("RGB/Depth 图像缺失")

    rgb_cv = vc.bridge.imgmsg_to_cv2(vc.rgb_image, "bgr8")
    depth_cv = vc.bridge.imgmsg_to_cv2(vc.depth_image, "16UC1")

    fd_rgb, temp_rgb = tempfile.mkstemp(prefix="grasp_only_rgb_", suffix=".jpg")
    os.close(fd_rgb)
    cv2.imwrite(temp_rgb, rgb_cv)

    try:
        text_prompt = vc._build_prompt([target_name])
        rospy.loginfo(f"检测提示词: {text_prompt}")
        results = vc.gsam_api.segment(
            image_path=temp_rgb,
            text_prompt=text_prompt,
            output_dir=None,
            save_mask=False,
            save_annotated=False,
            save_json=False,
        )
    finally:
        if os.path.exists(temp_rgb):
            try:
                os.remove(temp_rgb)
            except OSError:
                pass

    boxes = results.get("boxes", [])
    confidences = results.get("confidences", [])
    labels = results.get("labels", [])
    masks = results.get("masks", None)

    if masks is None or len(boxes) == 0:
        raise RuntimeError("GroundingSAM 未返回有效检测或分割 mask")

    depth_for_stats = depth_cv
    if depth_for_stats.shape[:2] != rgb_cv.shape[:2]:
        rospy.logwarn(
            f"Depth/RGB size mismatch during selection: depth={depth_for_stats.shape[:2]} rgb={rgb_cv.shape[:2]}"
        )
        depth_for_stats = cv2.resize(
            depth_for_stats,
            (rgb_cv.shape[1], rgb_cv.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    candidates = []
    for index, box in enumerate(boxes):
        confidence = float(confidences[index])
        if confidence < confidence_threshold:
            continue

        label = _normalize_label(labels[index])
        bbox = _clip_bbox(box, rgb_cv.shape[:2])
        raw_mask = np.asarray(masks[index])
        if raw_mask.ndim > 2:
            raw_mask = np.squeeze(raw_mask)
        if raw_mask.shape[:2] != rgb_cv.shape[:2]:
            rospy.logwarn(
                f"SAM mask shape mismatch for candidate {index}: mask={raw_mask.shape} rgb={rgb_cv.shape[:2]}, fallback to bbox mask"
            )
            mask = _bbox_to_mask(bbox, rgb_cv.shape[:2])
        else:
            mask = raw_mask.astype(bool)

        mask_pixels = int(np.count_nonzero(mask))
        valid_depth_pixels = int(np.count_nonzero(mask & (depth_for_stats > 0)))

        candidates.append(
            {
                "index": index,
                "label": label,
                "confidence": confidence,
                "bbox": bbox,
                "mask": mask,
                "mask_pixels": mask_pixels,
                "valid_depth_pixels": valid_depth_pixels,
            }
        )

    if not candidates:
        raise RuntimeError("没有满足阈值的检测候选")

    candidates.sort(
        key=lambda item: (
            item["confidence"],
            item["valid_depth_pixels"],
            item["mask_pixels"],
            -item["index"],
        ),
        reverse=True,
    )
    selected = candidates[0]

    obj = MockDetectedObject(selected["label"], selected["confidence"])
    obj.bbox = list(selected["bbox"])

    rgb_t = _msg_stamp_sec(vc.rgb_image)
    depth_t = _msg_stamp_sec(vc.depth_image)
    dt = None if rgb_t is None or depth_t is None else abs(rgb_t - depth_t)
    rospy.loginfo(
        "Selected detection: "
        f"index={selected['index']} label={selected['label']} conf={selected['confidence']:.4f} "
        f"bbox={selected['bbox']} mask_pixels={selected['mask_pixels']} "
        f"valid_depth_pixels={selected['valid_depth_pixels']} rgb_depth_dt={dt}"
    )

    return obj, selected["mask"], rgb_cv, depth_cv


class GraspOnlyClient:
    def __init__(
        self,
        *,
        conda_exe: str,
        grasp_env: str,
        profile_bundle_path: str,
        gripper_config: str,
    ) -> None:
        self.conda_exe = conda_exe
        self.grasp_env = grasp_env
        self.profile_bundle_path = profile_bundle_path
        self.gripper_config = gripper_config
        self.proc: Optional[subprocess.Popen] = None
        self.registry: Optional[ProfileRegistry] = None

    def start(self) -> None:
        if not os.path.exists(self.profile_bundle_path):
            raise FileNotFoundError(self.profile_bundle_path)
        with open(self.profile_bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)
        self.registry = ProfileRegistry(
            default_profile=bundle.get("default_profile", {}),
            category_profiles=bundle.get("category_profiles", {}),
        )

        grasp_worker = os.path.join(DETECT_GRASP_DIR, "grasp_hub_worker.py")
        env = os.environ.copy()
        extra_py_paths = [
            "/home/songfei/catkin_ws/src/piper_ros/src/piper/scripts",
            "/home/songfei/catkin_ws/src/piper_ros/src/GraspGen",
        ]
        env["PYTHONPATH"] = ":".join(extra_py_paths + [env.get("PYTHONPATH", "")])
        env.pop("LD_LIBRARY_PATH", None)
        conda_sh = "/home/songfei/miniconda3/etc/profile.d/conda.sh"
        launch_cmd = (
            f"source {conda_sh} && conda activate {self.grasp_env} && "
            f"python {grasp_worker}"
        )
        self.proc = subprocess.Popen(
            ["bash", "-lc", launch_cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        profile = self.registry.get(
            "default",
            override={
                "enable_visualization": False,
                "estimator_config": {"gripper_config": self.gripper_config},
            },
        )
        send_message(self.proc.stdin, {"cmd": CMD_INIT, "profile": profile})
        try:
            _ = recv_message(self.proc.stdout)
        except EOFError:
            err = b""
            if self.proc.stderr is not None:
                try:
                    err = self.proc.stderr.read()
                except Exception:
                    err = b""
            raise RuntimeError(
                "Grasp worker 启动失败，stderr:\n" + err.decode("utf-8", errors="replace")
            )

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            send_message(self.proc.stdin, {"cmd": CMD_QUIT})
            _ = recv_message(self.proc.stdout)
        finally:
            self.proc = None

    def run_grasp(
        self,
        *,
        color_bgr: np.ndarray,
        depth_m: np.ndarray,
        intrinsics: Dict[str, Any],
        mask: np.ndarray,
        bbox: List[int],
        class_name: str,
        confidence: float,
    ) -> Dict[str, Any]:
        if self.proc is None or self.registry is None:
            raise RuntimeError("GraspOnlyClient 未启动")

        packed = np.packbits(mask.astype(np.uint8), axis=None).tolist()
        detect_output = {
            "input_source": "rgbd_arrays",
            "color_image": color_bgr,
            "depth_image": depth_m.astype(np.float32),
            "intrinsics": intrinsics,
            "selected_mask_packed": packed,
            "selected_mask_shape": list(mask.shape),
            "selected_bbox": [int(v) for v in bbox],
            "selected_class_name": class_name,
            "selected_confidence": float(confidence),
            "selected_index": 0,
            "num_candidates": 1,
        }

        # 类别映射：餐具统一进 tableware 配置
        lower = str(class_name).lower()
        if lower in ("fork", "spoon", "knife"):
            category = "tableware"
        else:
            category = lower

        profile = self.registry.get(
            category,
            override={
                "enable_visualization": False,
                "estimator_config": {"gripper_config": self.gripper_config},
            },
        )

        send_message(self.proc.stdin, {"cmd": CMD_GRASP, "profile": profile, "detect_output": detect_output})
        return recv_message(self.proc.stdout)


def _build_intrinsics(vc: VisionContext, w: int, h: int) -> Dict[str, Any]:
    return {
        "fx": float(vc.fx),
        "fy": float(vc.fy),
        "cx": float(vc.cx),
        "cy": float(vc.cy),
        "width": int(w),
        "height": int(h),
    }


def _resolve_intrinsics(vc: VisionContext, w: int, h: int) -> Dict[str, Any]:
    use_camera_info = bool(rospy.get_param("~camera_intrinsics/use_camera_info", True))
    if not use_camera_info:
        return _build_intrinsics(vc, w, h)

    camera_info_topic = rospy.get_param("~camera_info_topic", "/camera/color/camera_info")
    camera_info_timeout = float(rospy.get_param("~camera_intrinsics/camera_info_timeout", 2.0))
    try:
        msg = rospy.wait_for_message(camera_info_topic, CameraInfo, timeout=camera_info_timeout)
        if len(msg.K) >= 9:
            intr = {
                "fx": float(msg.K[0]),
                "fy": float(msg.K[4]),
                "cx": float(msg.K[2]),
                "cy": float(msg.K[5]),
                "width": int(w),
                "height": int(h),
            }
            rospy.loginfo(
                f"Using CameraInfo intrinsics from {camera_info_topic}: "
                f"fx={intr['fx']:.3f}, fy={intr['fy']:.3f}, cx={intr['cx']:.3f}, cy={intr['cy']:.3f}"
            )
            return intr
    except Exception as exc:
        rospy.logwarn(f"读取 CameraInfo 失败，回退到私有参数内参: {exc}")

    return _build_intrinsics(vc, w, h)


def _matrix_from_tf(transform) -> np.ndarray:
    t = transform.transform.translation
    q = transform.transform.rotation
    x, y, z = t.x, t.y, t.z
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    R = np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def _pose_from_matrix(T: np.ndarray) -> List[float]:
    q = quaternion_from_matrix(T)
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def _call_joint_moveit_ctrl_endpose(endpose, max_velocity=0.3, max_acceleration=0.3) -> bool:
    rospy.wait_for_service("joint_moveit_ctrl_endpose")
    try:
        srv = rospy.ServiceProxy("joint_moveit_ctrl_endpose", JointMoveitCtrl)
        req = JointMoveitCtrlRequest()
        req.joint_states = [0.0] * 6
        req.gripper = 0.0
        req.max_velocity = max_velocity
        req.max_acceleration = max_acceleration
        req.joint_endpose = endpose
        resp = srv(req)
        return bool(resp.status)
    except rospy.ServiceException as e:
        rospy.logerr(f"endpose 服务调用失败: {e}")
        return False


def _call_joint_moveit_ctrl_gripper(gripper_position, max_velocity=0.3, max_acceleration=0.3) -> bool:
    rospy.wait_for_service("joint_moveit_ctrl_gripper")
    try:
        srv = rospy.ServiceProxy("joint_moveit_ctrl_gripper", JointMoveitCtrl)
        req = JointMoveitCtrlRequest()
        req.joint_states = [0.0] * 6
        req.gripper = gripper_position
        req.max_velocity = max_velocity
        req.max_acceleration = max_acceleration
        resp = srv(req)
        return bool(resp.status)
    except rospy.ServiceException as e:
        rospy.logerr(f"gripper 服务调用失败: {e}")
        return False


def main() -> None:
    rospy.init_node("grasp_only_vision")
    rospy.loginfo("========== Grasp Only Vision ==========")

    if not rospy.has_param("~depth_topic"):
        rospy.set_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
    if not rospy.has_param("~require_depth"):
        rospy.set_param("~require_depth", True)

    vc = VisionContext()

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
    if not _wait_for_synced_rgbd(vc, timeout_sec=15.0, max_dt_sec=rgbd_max_dt):
        rospy.logerr(f"未能获取时间差小于 {rgbd_max_dt:.3f}s 的 RGB/Depth 图像，退出")
        return

    try:
        obj, mask, rgb_cv, depth_cv = _select_detection_with_mask(
            vc,
            target_name=target_name,
            confidence_threshold=confidence_threshold,
            rgbd_max_dt_sec=rgbd_max_dt,
        )
    except Exception as exc:
        rospy.logerr(f"目标检测/分割失败: {exc}")
        return

    # 对齐深度尺寸到 RGB（防止索引尺寸不匹配）
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
    intr = _resolve_intrinsics(vc, rgb_cv.shape[1], rgb_cv.shape[0])
    rospy.loginfo(
        f"Grasp input summary: label={obj.class_name} conf={obj.confidence:.4f} "
        f"bbox={obj.bbox} intrinsics=(fx={intr['fx']:.3f}, fy={intr['fy']:.3f}, cx={intr['cx']:.3f}, cy={intr['cy']:.3f})"
    )

    grasp_client = GraspOnlyClient(
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

    grasp_T_cam = np.asarray(best_pose, dtype=np.float64)
    try:
        tf_stamped = vc.tf_buffer.lookup_transform(base_frame, camera_frame, rospy.Time(0), rospy.Duration(1.0))
        base_T_cam = _matrix_from_tf(tf_stamped)
    except Exception as e:
        rospy.logwarn(f"TF 变换失败（{base_frame}<-{camera_frame}），使用相机坐标: {e}")
        base_T_cam = np.eye(4, dtype=np.float64)

    base_T_grasp = base_T_cam @ grasp_T_cam
    rospy.loginfo(f"best_grasp_pose (base_link):\n{base_T_grasp}")
    # 预抓取位姿：沿抓取姿态负 z 方向回退
    pre_T = base_T_grasp.copy()
    pre_T[:3, 3] -= pregrasp_offset * pre_T[:3, 2]

    pre_endpose = _pose_from_matrix(pre_T)
    grasp_endpose = _pose_from_matrix(base_T_grasp)

    rospy.loginfo("执行机械臂抓取动作...")
    _call_joint_moveit_ctrl_gripper(gripper_open, max_velocity=vel, max_acceleration=acc)
    if not _call_joint_moveit_ctrl_endpose(pre_endpose, max_velocity=vel, max_acceleration=acc):
        rospy.logerr("移动到预抓取位失败")
        return
    rospy.loginfo("已移动到预抓取位，按当前配置不继续执行抓取位与闭爪动作")


if __name__ == "__main__":
    main()

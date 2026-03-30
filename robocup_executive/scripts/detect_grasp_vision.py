#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo
import tf2_ros
from moveit_ctrl.srv import JointMoveitCtrl, JointMoveitCtrlRequest
from tf.transformations import quaternion_from_matrix

PICK_PLACE_PATH = os.path.join(
    os.path.dirname(__file__),
    "pick_place_run_vision.py",
)
if os.path.exists(PICK_PLACE_PATH):
    sys.path.insert(0, os.path.dirname(PICK_PLACE_PATH))
try:
    from pick_place_run_vision import VisionContext
except Exception as exc:
    raise RuntimeError(
        "无法导入 pick_place_run_vision.py 中的 VisionContext。"
        "请确认该文件存在且可被 Python 导入。"
    ) from exc

DETECT_GRASP_DIR = "/home/songfei/catkin_ws/src/piper_ros/src/piper/scripts/detect_grasp"
if DETECT_GRASP_DIR not in sys.path:
    sys.path.insert(0, DETECT_GRASP_DIR)
from pipeline import DetectGraspConfig, DetectGraspRunner  # noqa: E402


MESHCAT_HOST = "127.0.0.1"
MESHCAT_ZMQ_PORT = 6000
MESHCAT_WEB_PORT = 7000
MESHCAT_URL = f"http://{MESHCAT_HOST}:{MESHCAT_WEB_PORT}/static/"


def _is_tcp_port_open(host: str, port: int, timeout_sec: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_sec):
            return True
    except OSError:
        return False


def _maybe_start_meshcat_server(
    *,
    grasp_env: str,
    enable_visualization: bool,
    auto_start_meshcat: bool,
    open_browser: bool,
) -> Optional[subprocess.Popen]:
    if not enable_visualization:
        return None

    web_ready = _is_tcp_port_open(MESHCAT_HOST, MESHCAT_WEB_PORT)
    zmq_ready = _is_tcp_port_open(MESHCAT_HOST, MESHCAT_ZMQ_PORT)
    if web_ready and zmq_ready:
        rospy.loginfo(f"检测到现有 MeshCat server: {MESHCAT_URL}")
        if open_browser:
            webbrowser.open(MESHCAT_URL, new=2)
        return None

    if not auto_start_meshcat:
        rospy.logwarn(
            "可视化已启用，但未检测到 MeshCat server。"
            f"请先在 {grasp_env} 环境中运行 `meshcat-server`，然后打开 {MESHCAT_URL}"
        )
        return None

    conda_sh = "/home/songfei/miniconda3/etc/profile.d/conda.sh"
    launch_cmd = f"source {conda_sh} && conda activate {grasp_env} && meshcat-server"
    rospy.loginfo("启动 MeshCat server 用于浏览器可视化...")
    proc = subprocess.Popen(
        ["bash", "-lc", launch_cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=os.environ.copy(),
    )

    deadline = time.time() + 8.0
    while time.time() < deadline and not rospy.is_shutdown():
        web_ready = _is_tcp_port_open(MESHCAT_HOST, MESHCAT_WEB_PORT)
        zmq_ready = _is_tcp_port_open(MESHCAT_HOST, MESHCAT_ZMQ_PORT)
        if web_ready and zmq_ready:
            rospy.loginfo(f"MeshCat server 已就绪: {MESHCAT_URL}")
            if open_browser:
                webbrowser.open(MESHCAT_URL, new=2)
            return proc
        time.sleep(0.2)

    rospy.logwarn(
        "等待 MeshCat server 超时。"
        f"你可以稍后手动打开 {MESHCAT_URL} 查看。"
    )
    return proc


def _msg_stamp_sec(msg) -> Optional[float]:
    if msg is None:
        return None
    try:
        return float(msg.header.stamp.to_sec())
    except Exception:
        return None


def _wait_for_synced_rgbd(
    vc: VisionContext,
    timeout_sec: float = 15.0,
    max_dt_sec: float = 0.08,
) -> bool:
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
    camera_info_timeout = float(
        rospy.get_param("~camera_intrinsics/camera_info_timeout", 2.0)
    )
    try:
        msg = rospy.wait_for_message(
            camera_info_topic,
            CameraInfo,
            timeout=camera_info_timeout,
        )
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
                f"fx={intr['fx']:.3f}, fy={intr['fy']:.3f}, "
                f"cx={intr['cx']:.3f}, cy={intr['cy']:.3f}"
            )
            return intr
    except Exception as exc:
        rospy.logwarn(f"读取 CameraInfo 失败，回退到私有参数内参: {exc}")

    return _build_intrinsics(vc, w, h)


def _load_rgbd_from_files(
    rgb_path: str,
    depth_path: str,
    depth_value_in_meters: bool,
    depth_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB 文件不存在: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth 文件不存在: {depth_path}")

    color_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError(f"无法读取 RGB 图像: {rgb_path}")

    if depth_path.lower().endswith(".npy"):
        depth = np.load(depth_path)
    else:
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"无法读取 Depth 图像: {depth_path}")

    depth = np.asarray(depth)
    if depth_value_in_meters:
        depth_m = depth.astype(np.float32)
    else:
        depth_m = depth.astype(np.float32) / float(depth_scale)

    if depth_m.shape[:2] != color_bgr.shape[:2]:
        rospy.logwarn(
            f"Depth/RGB size mismatch: depth={depth_m.shape[:2]} rgb={color_bgr.shape[:2]}, resizing depth"
        )
        depth_m = cv2.resize(
            depth_m,
            (color_bgr.shape[1], color_bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return color_bgr, depth_m


def _normalize_target_name(target_name: str) -> str:
    return str(target_name).lower().strip().replace(".", "")


def _build_profile_override(
    *,
    target_name: str,
    enable_visualization: bool,
    num_visualize_grasps: int,
    block_after_visualization: bool,
    gripper_config: str,
    text_prompt: str,
) -> Dict[str, Any]:
    prompt = text_prompt.strip() if text_prompt.strip() else f"{target_name}."
    if not prompt.endswith("."):
        prompt += "."
    return {
        "text_prompt": prompt,
        "enable_visualization": bool(enable_visualization),
        "num_visualize_grasps": int(num_visualize_grasps),
        "block_after_visualization": bool(block_after_visualization),
        "estimator_config": {"gripper_config": gripper_config},
    }


def _select_category(target_name: str, explicit_category: str) -> str:
    if explicit_category.strip():
        return explicit_category.strip()
    lower = _normalize_target_name(target_name)
    if lower in ("fork", "spoon", "knife"):
        return "tableware"
    return lower or "default"


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
    return [
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        float(q[3]),
    ]


def _call_joint_moveit_ctrl_endpose(
    endpose,
    max_velocity: float = 0.3,
    max_acceleration: float = 0.3,
) -> bool:
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
    except rospy.ServiceException as exc:
        rospy.logerr(f"endpose 服务调用失败: {exc}")
        return False


def _call_joint_moveit_ctrl_gripper(
    gripper_position: float,
    max_velocity: float = 0.3,
    max_acceleration: float = 0.3,
) -> bool:
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
    except rospy.ServiceException as exc:
        rospy.logerr(f"gripper 服务调用失败: {exc}")
        return False


def _log_detect_summary(result: Dict[str, Any]) -> None:
    payload = result.get("result_payload", {})
    source = payload.get("source", {})
    rospy.loginfo(
        "detect_grasp summary: "
        f"class={source.get('class_name')} conf={source.get('confidence')} "
        f"bbox={source.get('bbox')} policy={source.get('policy')} "
        f"input={source.get('input_source')}"
    )
    top5 = result.get("top5_grasp_confs", [])
    if top5:
        rospy.loginfo(f"top5 grasp confs: {top5}")


def main() -> None:
    rospy.init_node("detect_grasp_vision")
    rospy.loginfo("========== Detect+Grasp Vision (detect_grasp API) ==========")

    input_source = rospy.get_param("~input_source", "rgbd_arrays")
    if input_source not in ("rgbd_arrays", "rgbd_files", "camera"):
        rospy.logerr(f"不支持的 input_source: {input_source}")
        return

    target_name = rospy.get_param("~target_name", "milk")
    category = _select_category(
        target_name,
        rospy.get_param("~category", ""),
    )
    text_prompt = rospy.get_param("~text_prompt", "")
    grasp_mode = rospy.get_param("~grasp_mode", "graspgen")

    conda_exe = rospy.get_param("~conda_exe", "/home/songfei/miniconda3/bin/conda")
    detect_env = rospy.get_param("~detect_env_name", "ganzhi")
    grasp_env = rospy.get_param("~grasp_env_name", "Grasp")
    profile_bundle_path = rospy.get_param(
        "~profile_bundle_path",
        "/home/songfei/catkin_ws/src/piper_ros/src/piper/scripts/detect_grasp/profile_bundle.json",
    )
    gripper_config = rospy.get_param(
        "~gripper_config",
        "/home/songfei/catkin_ws/src/piper_ros/src/GraspGenModels/checkpoints/graspgen_robotiq_2f_140.yml",
    )

    enable_visualization = bool(rospy.get_param("~enable_visualization", True))
    num_visualize_grasps = int(rospy.get_param("~num_visualize_grasps", 10))
    block_after_visualization = bool(
        rospy.get_param("~block_after_visualization", False)
    )
    auto_start_meshcat = bool(rospy.get_param("~auto_start_meshcat", True))
    open_browser = bool(rospy.get_param("~open_browser", True))

    profile_bundle = {}
    if os.path.exists(profile_bundle_path):
        with open(profile_bundle_path, "r", encoding="utf-8") as f:
            profile_bundle = json.load(f)
    else:
        rospy.logwarn(f"profile_bundle 不存在，将只使用默认 profile: {profile_bundle_path}")

    meshcat_proc = _maybe_start_meshcat_server(
        grasp_env=grasp_env,
        enable_visualization=enable_visualization,
        auto_start_meshcat=auto_start_meshcat,
        open_browser=open_browser,
    )

    try:
        cfg = DetectGraspConfig(
            category=category,
            input_source=input_source,
            conda_exe=conda_exe,
            detect_env_name=detect_env,
            grasp_env_name=grasp_env,
            default_profile=profile_bundle.get("default_profile", {}),
            category_profiles=profile_bundle.get("category_profiles", {}),
            verbose=True,
            grasp_mode=grasp_mode,
        )

        profile_override = _build_profile_override(
            target_name=target_name,
            enable_visualization=enable_visualization,
            num_visualize_grasps=num_visualize_grasps,
            block_after_visualization=block_after_visualization,
            gripper_config=gripper_config,
            text_prompt=text_prompt,
        )

        color_bgr = None
        depth_m = None
        intrinsics = None
        rgb_path = None
        depth_path = None
        depth_value_in_meters = False
        depth_scale = float(rospy.get_param("~depth_scale", 1000.0))
        camera_frame = "camera_color_optical_frame"
        tf_buffer = None

        if input_source == "rgbd_arrays":
            if not rospy.has_param("~depth_topic"):
                rospy.set_param("~depth_topic", "/camera/aligned_depth_to_color/image_raw")
            if not rospy.has_param("~require_depth"):
                rospy.set_param("~require_depth", True)
            vc = VisionContext()
            camera_frame = vc.camera_frame_id
            tf_buffer = vc.tf_buffer
            rospy.loginfo(f"RGB topic: {vc.rgb_topic}, Depth topic: {vc.depth_topic}")
            if not vc.wait_for_images(timeout_sec=15.0):
                rospy.logerr("未能获取相机图像，退出")
                return
            rgbd_max_dt = float(rospy.get_param("~rgbd_max_dt", 0.08))
            if not _wait_for_synced_rgbd(vc, timeout_sec=15.0, max_dt_sec=rgbd_max_dt):
                rospy.logerr(
                    f"未能获取时间差小于 {rgbd_max_dt:.3f}s 的 RGB/Depth 图像，退出"
                )
                return

            color_bgr = vc.bridge.imgmsg_to_cv2(vc.rgb_image, "bgr8")
            depth_cv = vc.bridge.imgmsg_to_cv2(vc.depth_image, "16UC1")
            if depth_cv.shape[:2] != color_bgr.shape[:2]:
                rospy.logwarn(
                    f"Depth/RGB size mismatch: depth={depth_cv.shape[:2]} rgb={color_bgr.shape[:2]}, resizing depth"
                )
                depth_cv = cv2.resize(
                    depth_cv,
                    (color_bgr.shape[1], color_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            depth_m = depth_cv.astype(np.float32) * float(vc.depth_scale)
            intrinsics = _resolve_intrinsics(vc, color_bgr.shape[1], color_bgr.shape[0])
        elif input_source == "rgbd_files":
            rgb_path = rospy.get_param("~rgb_path", "")
            depth_path = rospy.get_param("~depth_path", "")
            depth_value_in_meters = bool(rospy.get_param("~depth_value_in_meters", False))
            if not rgb_path or not depth_path:
                rospy.logerr("rgbd_files 模式需要设置 ~rgb_path 和 ~depth_path")
                return
            color_bgr, depth_m = _load_rgbd_from_files(
                rgb_path,
                depth_path,
                depth_value_in_meters,
                depth_scale,
            )
            intrinsics = {
                "fx": float(rospy.get_param("~camera_intrinsics/fx")),
                "fy": float(rospy.get_param("~camera_intrinsics/fy")),
                "cx": float(rospy.get_param("~camera_intrinsics/cx")),
                "cy": float(rospy.get_param("~camera_intrinsics/cy")),
                "width": int(color_bgr.shape[1]),
                "height": int(color_bgr.shape[0]),
            }
        else:
            base_intrinsics = {}
            for key in ("fx", "fy", "cx", "cy", "width", "height"):
                param = f"~camera_intrinsics/{key}"
                if rospy.has_param(param):
                    base_intrinsics[key] = rospy.get_param(param)
            if base_intrinsics:
                cfg.intrinsics = base_intrinsics

        rospy.loginfo(
            f"DetectGrasp input summary: target={target_name} category={category} "
            f"mode={grasp_mode} input_source={input_source} detect_env={detect_env} grasp_env={grasp_env}"
        )

        with DetectGraspRunner(cfg) as runner:
            result = runner.run(
                category=category,
                input_source=input_source,
                rgb_path=rgb_path,
                depth_path=depth_path,
                color_image=color_bgr,
                depth_image=depth_m,
                intrinsics=intrinsics,
                depth_value_in_meters=depth_value_in_meters,
                depth_scale=depth_scale,
                profile_override=profile_override,
                grasp_mode=grasp_mode,
            )

        rospy.loginfo(f"detect_grasp status: {result.get('status')} message: {result.get('message', '')}")
        _log_detect_summary(result)

        best_pose = result.get("best_grasp_pose")
        best_conf = result.get("best_grasp_conf")
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
        camera_frame = rospy.get_param("~motion/camera_frame", camera_frame)
        pregrasp_offset = float(rospy.get_param("~motion/pregrasp_offset", 0.10))
        gripper_open = float(rospy.get_param("~motion/gripper_open", 0.035))
        vel = float(rospy.get_param("~motion/max_velocity", 0.3))
        acc = float(rospy.get_param("~motion/max_acceleration", 0.3))

        grasp_T_cam = np.asarray(best_pose, dtype=np.float64)
        try:
            if tf_buffer is None:
                tf_buffer = tf2_ros.Buffer()
                tf2_ros.TransformListener(tf_buffer)
                rospy.sleep(0.5)
            tf_stamped = tf_buffer.lookup_transform(
                base_frame,
                camera_frame,
                rospy.Time(0),
                rospy.Duration(1.0),
            )
            base_T_cam = _matrix_from_tf(tf_stamped)
        except Exception as exc:
            rospy.logwarn(
                f"TF 变换失败（{base_frame}<-{camera_frame}），使用相机坐标: {exc}"
            )
            base_T_cam = np.eye(4, dtype=np.float64)

        base_T_grasp = base_T_cam @ grasp_T_cam
        rospy.loginfo(f"best_grasp_pose (base_link):\n{base_T_grasp}")

        pre_T = base_T_grasp.copy()
        pre_T[:3, 3] -= pregrasp_offset * pre_T[:3, 2]

        pre_endpose = _pose_from_matrix(pre_T)
        _call_joint_moveit_ctrl_gripper(
            gripper_open,
            max_velocity=vel,
            max_acceleration=acc,
        )
        if not _call_joint_moveit_ctrl_endpose(
            pre_endpose,
            max_velocity=vel,
            max_acceleration=acc,
        ):
            rospy.logerr("移动到预抓取位失败")
            return
        rospy.loginfo("已移动到预抓取位，按当前配置不继续执行抓取位与闭爪动作")
    finally:
        if meshcat_proc is not None:
            try:
                meshcat_proc.terminate()
                meshcat_proc.wait(timeout=2.0)
            except Exception:
                try:
                    meshcat_proc.kill()
                except Exception:
                    pass


if __name__ == "__main__":
    main()

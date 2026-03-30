import json

try:
    from sg_pkg.scripts.detect_grasp.pipeline import DetectGraspConfig, DetectGraspRunner
except ModuleNotFoundError:
    # 兼容直接运行本目录脚本：python /abs/path/usage_one.py
    from pipeline import DetectGraspConfig, DetectGraspRunner
import numpy as np

profile_bundle_path = "/home/h/PPC/src/sg_pkg/scripts/detect_grasp/profile_bundle_template.json"
rgb_path = "/home/h/PPC/src/GraspGen/rs_data/4object/color.png"
depth_path = "/home/h/PPC/src/GraspGen/rs_data/4object/depth.png"

# 加载配置文件
with open(profile_bundle_path, "r", encoding="utf-8") as f:
    bundle = json.load(f)

cfg = DetectGraspConfig(
    conda_exe="conda",
    detect_env_name="dsam2",
    grasp_env_name="GraspGen",
    input_source="rgbd_files",
    default_profile=bundle.get("default_profile", {}),
    category_profiles=bundle.get("category_profiles", {}),
)


def _format_matrix_4x4(mat) -> str:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.shape != (4, 4):
        return np.array2string(arr, separator=", ")
    rows = ["[" + ", ".join(f"{v: .8f}" for v in row) + "]" for row in arr]
    return "[\n  " + ",\n  ".join(rows) + "\n]"


with DetectGraspRunner(cfg) as runner:
    # 同一 runner 内可按需切换模式，模型复用不重载。
    # # 2
    result2 = runner.run(
        category="plate",
        input_source="camera",
        rgb_path=rgb_path,
        depth_path=depth_path,
        profile_override={"text_prompt": "plate."},
        grasp_mode="graspgen",
    )
    pose2 = result2["best_grasp_pose"]
    pose2_str = "None"
    if pose2 is not None:
        pose2_str = _format_matrix_4x4(pose2)
    print(f"\n[{result2.get('grasp_mode')}] best_grasp_conf: {result2['best_grasp_conf']}")
    if result2.get("status") != "ok":
        print(f"[{result2.get('grasp_mode')}] message: {result2.get('message')}")
    print(f"[{result2.get('grasp_mode')}] best_grasp_pose:\n{pose2_str}")

    # 3
    # result2 = runner.run(
    #     category="tableware",
    #     input_source="camera",
    #     rgb_path=rgb_path,
    #     depth_path=depth_path,
    #     profile_override={"text_prompt": "spoon."},
    #     grasp_mode="tableware_pca",
    # )
    # pose2 = result2["best_grasp_pose"]
    # pose2_str = "None"
    # if pose2 is not None:
    #     pose2_str = _format_matrix_4x4(pose2)
    # print(f"\n[{result2.get('grasp_mode')}] best_grasp_conf: {result2['best_grasp_conf']}")
    # if result2.get("status") != "ok":
    #     print(f"[{result2.get('grasp_mode')}] message: {result2.get('message')}")
    # print(f"[{result2.get('grasp_mode')}] best_grasp_pose:\n{pose2_str}")


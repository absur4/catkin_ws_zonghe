import os
import cv2
import argparse
import numpy as np


def find_nearest_valid_pixel(u_target, v_target, us, vs):
    d2 = (us - u_target) ** 2 + (vs - v_target) ** 2
    idx = np.argmin(d2)
    return int(us[idx]), int(vs[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=str, required=True, help="Path to RGB image")
    parser.add_argument("--depth", type=str, required=True, help="Path to aligned depth image (uint16)")
    parser.add_argument("--mask", type=str, required=True, help="Path to binary mask image")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")

    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)

    parser.add_argument(
        "--depth_scale",
        type=float,
        default=0.001,
        help="Scale from raw depth unit to meters. For uint16 depth in mm, use 0.001"
    )
    parser.add_argument("--min_depth_raw", type=int, default=300, help="Minimum valid raw depth")
    parser.add_argument("--max_depth_raw", type=int, default=4000, help="Maximum valid raw depth")

    args = parser.parse_args()

    rgb_path = os.path.abspath(args.rgb)
    depth_path = os.path.abspath(args.depth)
    mask_path = os.path.abspath(args.mask)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask image: {mask_path}")

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if depth.ndim != 2:
        raise RuntimeError(f"Depth image must be single-channel. Got shape: {depth.shape}")

    if rgb.shape[:2] != depth.shape[:2]:
        raise RuntimeError(f"RGB/depth shape mismatch: rgb={rgb.shape[:2]}, depth={depth.shape[:2]}")

    if mask.shape[:2] != depth.shape[:2]:
        raise RuntimeError(f"Mask/depth shape mismatch: mask={mask.shape[:2]}, depth={depth.shape[:2]}")

    print("[INFO] RGB shape:", rgb.shape, rgb.dtype)
    print("[INFO] Depth shape:", depth.shape, depth.dtype)
    print("[INFO] Mask shape:", mask.shape, mask.dtype)

    mask_bin = mask > 127

    valid = (
        mask_bin &
        (depth > args.min_depth_raw) &
        (depth < args.max_depth_raw)
    )

    valid_count = int(valid.sum())
    print("[INFO] valid mask-depth pixel count:", valid_count)

    if valid_count < 30:
        raise RuntimeError(
            f"Too few valid depth pixels inside mask: {valid_count}. "
            "Check mask/depth alignment or depth validity."
        )

    vs, us = np.where(valid)
    z_raw = depth[vs, us].astype(np.float32)

    # 先做一层稳健裁剪，去掉深度边缘噪声
    if len(z_raw) >= 50:
        q10, q90 = np.percentile(z_raw, [10, 90])
        keep = (z_raw >= q10) & (z_raw <= q90)
    else:
        keep = np.ones_like(z_raw, dtype=bool)

    us_f = us[keep]
    vs_f = vs[keep]
    z_raw_f = z_raw[keep]

    if len(z_raw_f) < 20:
        raise RuntimeError(
            f"Too few pixels after robust filtering: {len(z_raw_f)}"
        )

    z_m = z_raw_f * args.depth_scale

    x_m = (us_f.astype(np.float32) - args.cx) * z_m / args.fx
    y_m = (vs_f.astype(np.float32) - args.cy) * z_m / args.fy

    # 代表点：mask 内有效3D点的中位数
    X_med = float(np.median(x_m))
    Y_med = float(np.median(y_m))
    Z_med = float(np.median(z_m))

    # 代表像素：mask 内有效像素的中位数位置附近最近有效点
    u_med = float(np.median(us_f))
    v_med = float(np.median(vs_f))
    u_pick, v_pick = find_nearest_valid_pixel(u_med, v_med, us_f, vs_f)

    z_pick_raw = int(depth[v_pick, u_pick])
    z_pick_m = z_pick_raw * args.depth_scale
    X_pick = (u_pick - args.cx) * z_pick_m / args.fx
    Y_pick = (v_pick - args.cy) * z_pick_m / args.fy

    print("\n[RESULT] Representative 3D point (median of 3D cloud):")
    print(f"position_cam = [{X_med:.6f}, {Y_med:.6f}, {Z_med:.6f}]  (meters)")

    print("\n[DEBUG] Representative pixel-based 3D point:")
    print(f"pixel_uv = [{u_pick}, {v_pick}]")
    print(f"depth_raw = {z_pick_raw}")
    print(f"position_cam_from_pixel = [{X_pick:.6f}, {Y_pick:.6f}, {z_pick_m:.6f}]  (meters)")

    # 保存文本结果
    txt_path = os.path.join(outdir, "position_result.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("[INPUT]\n")
        f.write(f"rgb={rgb_path}\n")
        f.write(f"depth={depth_path}\n")
        f.write(f"mask={mask_path}\n\n")

        f.write("[INTRINSICS]\n")
        f.write(f"fx={args.fx}\n")
        f.write(f"fy={args.fy}\n")
        f.write(f"cx={args.cx}\n")
        f.write(f"cy={args.cy}\n")
        f.write(f"depth_scale={args.depth_scale}\n\n")

        f.write("[STATS]\n")
        f.write(f"valid_count={valid_count}\n")
        f.write(f"filtered_count={len(z_raw_f)}\n")
        f.write(f"depth_raw_min={int(np.min(z_raw_f))}\n")
        f.write(f"depth_raw_median={float(np.median(z_raw_f))}\n")
        f.write(f"depth_raw_max={int(np.max(z_raw_f))}\n\n")

        f.write("[RESULT_MEDIAN_3D]\n")
        f.write(f"position_cam=[{X_med:.6f}, {Y_med:.6f}, {Z_med:.6f}]\n\n")

        f.write("[RESULT_PIXEL_3D]\n")
        f.write(f"pixel_uv=[{u_pick}, {v_pick}]\n")
        f.write(f"depth_raw={z_pick_raw}\n")
        f.write(f"position_cam_from_pixel=[{X_pick:.6f}, {Y_pick:.6f}, {z_pick_m:.6f}]\n")

    # 保存调试图
    dbg = rgb.copy()

    # 画原始mask轮廓
    contours, _ = cv2.findContours(
        (mask_bin.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(dbg, contours, -1, (0, 255, 0), 2)

    # 画有效深度区域轮廓
    valid_u8 = (valid.astype(np.uint8) * 255)
    valid_contours, _ = cv2.findContours(
        valid_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(dbg, valid_contours, -1, (255, 0, 0), 2)

    # 画代表像素
    cv2.circle(dbg, (u_pick, v_pick), 6, (0, 0, 255), -1)
    cv2.putText(
        dbg,
        f"pos_cam=({X_med:.3f}, {Y_med:.3f}, {Z_med:.3f})m",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2
    )

    dbg_path = os.path.join(outdir, "position_debug.png")
    cv2.imwrite(dbg_path, dbg)

    print(f"\n[INFO] Saved result txt to: {txt_path}")
    print(f"[INFO] Saved debug image to: {dbg_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

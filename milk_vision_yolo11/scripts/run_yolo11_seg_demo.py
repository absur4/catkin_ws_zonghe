import os
import cv2
import argparse
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input RGB image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-seg.pt",
        help="YOLO11 segmentation model path"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )
    args = parser.parse_args()

    image_path = os.path.abspath(args.image)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    print(f"[INFO] Loading model: {args.model}")
    model = YOLO(args.model)

    print(f"[INFO] Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"cv2.imread failed: {image_path}")

    h, w = image.shape[:2]
    print(f"[INFO] Image shape: {w} x {h}")

    print("[INFO] Running segmentation inference...")
    results = model.predict(
        source=image_path,
        conf=args.conf,
        imgsz=args.imgsz,
        device=0,
        retina_masks=True,
        verbose=False
    )

    if len(results) == 0:
        print("[WARN] No results returned by model.")
        return

    result = results[0]

    # 1) 保存整体可视化结果图
    vis = result.plot()
    vis_path = os.path.join(outdir, "result_overlay.jpg")
    cv2.imwrite(vis_path, vis)
    print(f"[INFO] Saved overlay image to: {vis_path}")

    # 2) 提取 boxes / masks
    if result.boxes is None or len(result.boxes) == 0:
        print("[WARN] No detections found.")
        return

    if result.masks is None:
        print("[WARN] Detections exist, but masks is None.")
        return

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    boxes_cls = result.boxes.cls.cpu().numpy().astype(int)
    boxes_conf = result.boxes.conf.cpu().numpy()
    masks = result.masks.data.cpu().numpy()  # shape: [N, H, W], with retina_masks=True -> original image size
    names = result.names

    print("\n[INFO] Detection summary:")
    print(f"  Number of instances: {len(boxes_xyxy)}")

    summary_lines = []

    for i in range(len(boxes_xyxy)):
        cls_id = int(boxes_cls[i])
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        conf = float(boxes_conf[i])
        x1, y1, x2, y2 = boxes_xyxy[i].tolist()

        mask = masks[i]
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        area_pixels = int((mask_bin > 0).sum())

        mask_path = os.path.join(outdir, f"mask_{i:02d}_{cls_name}.png")
        cv2.imwrite(mask_path, mask_bin)

        line = (
            f"instance {i}: "
            f"class={cls_name}, conf={conf:.4f}, "
            f"box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], "
            f"mask_area_pixels={area_pixels}, "
            f"mask_path={mask_path}"
        )
        print(line)
        summary_lines.append(line)

    summary_txt = os.path.join(outdir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        for line in summary_lines:
            f.write(line + "\n")

    print(f"\n[INFO] Saved detection summary to: {summary_txt}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

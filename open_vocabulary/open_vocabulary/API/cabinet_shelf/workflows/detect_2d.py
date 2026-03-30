#!/usr/bin/env python3
"""2D cabinet shelf detection workflow."""

import json
import os
from typing import Dict, Optional

try:
    from ..runtime import (
        box_convert,
        cluster_horizontal_lines,
        compute_shelf_layers,
        cv2,
        detect_shelf_lines,
        draw_annotated_image,
        draw_shelf_lines_image,
        generate_equal_layers,
        load_image,
        mask_util,
        np,
        predict,
        torch,
    )
    from ...utils.shelf_geometry import assign_object_to_layer
except ImportError:
    from cabinet_shelf.runtime import (
        box_convert,
        cluster_horizontal_lines,
        compute_shelf_layers,
        cv2,
        detect_shelf_lines,
        draw_annotated_image,
        draw_shelf_lines_image,
        generate_equal_layers,
        load_image,
        mask_util,
        np,
        predict,
        torch,
    )
    from utils.shelf_geometry import assign_object_to_layer
def detect(detector, image_path: str, object_prompt: Optional[str] = None,
           cabinet_prompt: str = "cabinet.", output_dir: Optional[str] = None,
           box_threshold: float = 0.3, text_threshold: float = 0.25,
           num_shelves_hint: int = 4, min_coverage: float = 0.4,
           cluster_min_gap: int = 100, min_lines_hint: int = 2,
           hough_fallback: bool = True, canny_low: int = 50,
           canny_high: int = 150, hough_threshold: int = 60,
           hough_min_line_len_ratio: float = 0.5, hough_max_line_gap: int = 20,
           horizontal_tol: int = 2, min_len_ratio: float = 0.15,
           max_angle_deg: float = 12.0, ransac_tol: float = 10.0,
           ransac_iters: int = 80, ransac_min_inliers: int = 20,
           dbscan_eps: float = 28.0, dbscan_min_samples: int = 4) -> Dict:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    img_h, img_w = image.shape[:2]

    print("\n[CabinetShelfDetector] Step 1: 检测柜子 (Grounding DINO only)...")
    cabinet_prompt_clean = cabinet_prompt if cabinet_prompt.endswith(".") else cabinet_prompt + "."
    _, image_transformed = load_image(image_path)
    boxes_raw, _, _labels_raw = predict(
        model=detector.api.grounding_model,
        image=image_transformed,
        caption=cabinet_prompt_clean.lower(),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=detector.api.device,
    )
    if len(boxes_raw) > 0:
        boxes_abs = boxes_raw * torch.Tensor([img_w, img_h, img_w, img_h])
        cabinet_boxes = box_convert(boxes=boxes_abs, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()
        areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in cabinet_boxes]
        best_idx = int(np.argmax(areas))
        cabinet_bbox = tuple(int(round(v)) for v in cabinet_boxes[best_idx])
        print(f"  检测到柜子: {cabinet_bbox}")
    else:
        print("  未检测到柜子，使用整张图像作为 ROI")
        cabinet_bbox = (0, 0, img_w, img_h)

    print("\n[CabinetShelfDetector] Step 2: 检测层板线...")
    debug_dir = os.path.join(output_dir, "debug") if output_dir else None
    raw_y = detect_shelf_lines(
        image,
        roi=cabinet_bbox,
        min_coverage=min_coverage,
        debug_dir=debug_dir,
        min_lines_hint=min_lines_hint,
        hough_fallback=hough_fallback,
        canny_low=canny_low,
        canny_high=canny_high,
        hough_threshold=hough_threshold,
        hough_min_line_len_ratio=hough_min_line_len_ratio,
        hough_max_line_gap=hough_max_line_gap,
        horizontal_tol=horizontal_tol,
        min_len_ratio=min_len_ratio,
        max_angle_deg=max_angle_deg,
        ransac_tol=ransac_tol,
        ransac_iters=ransac_iters,
        ransac_min_inliers=ransac_min_inliers,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
    )
    print(f"  原始检测: {len(raw_y)} 条候选线")
    shelf_y = cluster_horizontal_lines(raw_y, min_gap=cluster_min_gap)
    print(f"  检测到 {len(shelf_y)} 条层板线: {shelf_y}")

    if shelf_y:
        shelf_layers = compute_shelf_layers(cabinet_bbox, shelf_y)
    else:
        print(f"  未检测到层板线，按 {num_shelves_hint} 层等分柜子")
        shelf_layers = generate_equal_layers(cabinet_bbox, num_shelves_hint)
        shelf_y = [layer["y_bottom"] for layer in shelf_layers[:-1]]

    num_layers = len(shelf_layers)
    print(f"  划分为 {num_layers} 层")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        shelf_only = draw_shelf_lines_image(image, cabinet_bbox, shelf_y, shelf_layers)
        shelf_lines_path = os.path.join(output_dir, "shelf_lines.jpg")
        cv2.imwrite(shelf_lines_path, shelf_only)
        print(f"  层板线标注图已保存至: {shelf_lines_path}")

    result = {
        "cabinet": {"bbox": list(cabinet_bbox)},
        "shelf_lines_y": shelf_y,
        "shelf_layers": shelf_layers,
        "num_layers": num_layers,
        "objects": [],
    }

    if not object_prompt:
        if output_dir is not None:
            json_path = os.path.join(output_dir, "shelf_results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"JSON 结果已保存至: {json_path}")
        print(f"\n[CabinetShelfDetector] 完成！共 {num_layers} 层（仅层板检测模式）")
        return result

    print("\n[CabinetShelfDetector] Step 3: 检测物品...")
    object_result = detector.api.segment(
        image_path=image_path,
        text_prompt=object_prompt,
        output_dir=None,
        save_mask=False,
        save_annotated=False,
        save_json=False,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    print("\n[CabinetShelfDetector] Step 4: 分配物品到对应层...")
    objects_info = []
    if object_result["boxes"]:
        for i, (bbox, conf, label) in enumerate(
            zip(object_result["boxes"], object_result["confidences"], object_result["labels"])
        ):
            bbox_int = tuple(int(round(v)) for v in bbox)
            layer = assign_object_to_layer(bbox_int, shelf_layers)
            obj = {
                "class_name": label.strip(),
                "bbox": list(bbox_int),
                "confidence": round(conf, 4),
                "shelf_layer": layer,
            }
            if object_result["masks"] is not None and i < len(object_result["masks"]):
                mask = object_result["masks"][i]
                rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                obj["mask_rle"] = rle
            objects_info.append(obj)
            print(f"  {label.strip()} (conf={conf:.2f}) → 第{layer}层")

    result["objects"] = objects_info

    if output_dir is not None:
        annotated = draw_annotated_image(image, cabinet_bbox, shelf_y, shelf_layers, objects_info)
        ann_path = os.path.join(output_dir, "shelf_annotated.jpg")
        cv2.imwrite(ann_path, annotated)
        print(f"\n标注图像已保存至: {ann_path}")
        json_path = os.path.join(output_dir, "shelf_results.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"JSON 结果已保存至: {json_path}")

    print(f"\n[CabinetShelfDetector] 完成！共 {num_layers} 层, {len(objects_info)} 个物品")
    return result



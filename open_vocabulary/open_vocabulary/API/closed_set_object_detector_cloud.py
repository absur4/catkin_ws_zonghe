#!/usr/bin/env python3
"""
Cloud closed-set object detector.

Pipeline:
1. Use DeepDataSpace GroundingDino-1.6-Edge API for detection.
2. Classify cropped detections with Google SigLIP.
3. Optionally override final results with dedicated plate/spoon detections.
"""

import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from requests.exceptions import HTTPError, RequestException
from transformers import AutoProcessor, SiglipModel

try:
    from .utils.bbox_utils import nms_xyxy_numpy
except ImportError:
    from utils.bbox_utils import nms_xyxy_numpy


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_API_MODEL = "GroundingDino-1.6-Edge"
DEFAULT_OVERRIDE_PROMPTS = ("plate.", "spoon.")


@dataclass
class SupportGallery:
    class_names: List[str]
    image_paths: List[str]
    image_features: torch.Tensor
    text_features: torch.Tensor
    support_features: torch.Tensor
    labels: torch.Tensor


class DeepDataSpaceGroundingDinoClient:
    """Async Grounding DINO API client."""

    def __init__(
        self,
        token: str,
        model: str = DEFAULT_API_MODEL,
        timeout_sec: float = 180.0,
        poll_interval_sec: float = 1.0,
        max_retries: int = 4,
        retry_backoff_sec: float = 2.0,
    ):
        if not token:
            raise ValueError("需要提供 DeepDataSpace Token")
        self.token = token
        self.model = model
        self.timeout_sec = timeout_sec
        self.poll_interval_sec = poll_interval_sec
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.create_url = "https://api.deepdataspace.com/v2/task/grounding_dino/detection"
        self.status_url = "https://api.deepdataspace.com/v2/task_status/{task_uuid}"

    def detect(
        self,
        image_path: str,
        prompt: str,
        bbox_threshold: float = 0.28,
        iou_threshold: float = 0.8,
        session_id: Optional[str] = None,
    ) -> Tuple[List[Dict], Optional[str]]:
        headers = {
            "Token": self.token,
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "prompt": {
                "type": "text",
                "text": prompt.rstrip("."),
            },
            "targets": ["bbox"],
            "bbox_threshold": float(bbox_threshold),
            "iou_threshold": float(iou_threshold),
        }

        if session_id:
            payload["session_id"] = session_id
        else:
            payload["image"] = self._image_to_base64(image_path)

        data = self._request_json_with_retry(
            method="post",
            url=self.create_url,
            headers=headers,
            json_body=payload,
            timeout=60,
        )
        if data.get("code") != 0:
            raise RuntimeError(f"创建检测任务失败: {data}")
        task_data = data.get("data") or {}
        if task_data.get("status") == "success" and isinstance(task_data.get("result"), dict):
            result = task_data.get("result", {})
            objects = result.get("objects", [])
            return objects, task_data.get("session_id")

        task_uuid = task_data.get("uuid") or task_data.get("task_uuid") or task_data.get("id")
        if not task_uuid:
            raise RuntimeError(
                "创建检测任务响应里没有 `uuid`。"
                f"完整响应: {json.dumps(data, ensure_ascii=False)}"
            )
        task = self._poll_task(task_uuid)
        result = task.get("result", {})
        objects = result.get("objects", [])
        return objects, task.get("session_id")

    def _poll_task(self, task_uuid: str) -> Dict:
        headers = {"Token": self.token}
        deadline = time.time() + self.timeout_sec
        while time.time() < deadline:
            url = self.status_url.format(task_uuid=task_uuid)
            data = self._request_json_with_retry(
                method="get",
                url=url,
                headers=headers,
                json_body=None,
                timeout=30,
            )
            if data.get("code") != 0:
                raise RuntimeError(f"查询任务失败: {data}")
            task = data["data"]
            status = task.get("status")
            if status == "success":
                return task
            if status in {"failed", "error", "canceled"}:
                raise RuntimeError(f"检测任务失败: {task}")
            time.sleep(self.poll_interval_sec)
        raise TimeoutError(f"检测任务超时: {task_uuid}")

    def _request_json_with_retry(
        self,
        method: str,
        url: str,
        headers: Dict[str, str],
        json_body: Optional[Dict],
        timeout: float,
    ) -> Dict:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if method == "post":
                    resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
                else:
                    resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except HTTPError as exc:
                response = exc.response
                status_code = response.status_code if response is not None else None
                response_text = response.text if response is not None else ""
                if status_code is not None and 400 <= status_code < 500:
                    raise RuntimeError(
                        f"DeepDataSpace {method.upper()} request returned {status_code}. "
                        f"Response body: {response_text}"
                    ) from exc
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.retry_backoff_sec * attempt
                print(
                    f"[DeepDataSpace] {method.upper()} failed "
                    f"(attempt {attempt}/{self.max_retries}), retry in {sleep_sec:.1f}s: {exc}"
                )
                time.sleep(sleep_sec)
            except RequestException as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                sleep_sec = self.retry_backoff_sec * attempt
                print(
                    f"[DeepDataSpace] {method.upper()} failed "
                    f"(attempt {attempt}/{self.max_retries}), retry in {sleep_sec:.1f}s: {exc}"
                )
                time.sleep(sleep_sec)
        raise RuntimeError(
            f"DeepDataSpace {method.upper()} request failed after {self.max_retries} attempts: {last_error}"
        )

    def _image_to_base64(self, image_path: str) -> str:
        mime = self._guess_mime(image_path)
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    def _guess_mime(self, image_path: str) -> str:
        suffix = Path(image_path).suffix.lower()
        if suffix == ".png":
            return "image/png"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".webp":
            return "image/webp"
        return "image/png"


class ClosedSetObjectDetectorCloud:
    """Cloud detection + SigLIP classification."""

    def __init__(
        self,
        train_root: str,
        api_token: str,
        device: Optional[str] = None,
        siglip_model_name: str = DEFAULT_SIGLIP_MODEL,
        api_model: str = DEFAULT_API_MODEL,
        class_aliases: Optional[Dict[str, Sequence[str]]] = None,
        min_similarity: Optional[float] = None,
        api_timeout_sec: float = 180.0,
        api_poll_interval_sec: float = 1.0,
        api_max_retries: int = 4,
        api_retry_backoff_sec: float = 2.0,
    ):
        self.train_root = os.path.abspath(train_root)
        self.device = device or self._auto_device()
        self.class_aliases = class_aliases or {}
        self.min_similarity = min_similarity

        self.dino_client = DeepDataSpaceGroundingDinoClient(
            token=api_token,
            model=api_model,
            timeout_sec=api_timeout_sec,
            poll_interval_sec=api_poll_interval_sec,
            max_retries=api_max_retries,
            retry_backoff_sec=api_retry_backoff_sec,
        )
        self.siglip_model = SiglipModel.from_pretrained(siglip_model_name).to(self.device)
        self.siglip_model.eval()
        try:
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name)
        except ImportError as exc:
            raise ImportError(
                "SigLIP 需要 `sentencepiece`。请先安装: "
                "`pip install sentencepiece` 或 `conda install -c conda-forge sentencepiece`"
            ) from exc
        self.gallery = self._build_support_gallery()
        self.proker_beta = self._estimate_beta(self.gallery.support_features)
        self._proker_state = self._build_proker_state()

    def detect_and_classify(
        self,
        image_path: str,
        objects: Sequence[str] | str,
        output_path: Optional[str] = None,
        save_json_path: Optional[str] = None,
        bbox_threshold: float = 0.28,
        api_iou_threshold: float = 0.8,
        final_nms_iou: float = 0.45,
        classify_crop_shrink: float = 0.12,
        min_crop_size: int = 8,
        override_prompts: Optional[Sequence[str]] = None,
        override_iou: float = 0.3,
        override_containment: float = 0.6,
    ) -> Dict:
        t0 = time.perf_counter()
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")
        h, w = image_bgr.shape[:2]

        primary_prompt = self._build_primary_prompt(objects)
        primary_objects, session_id = self.dino_client.detect(
            image_path=image_path,
            prompt=primary_prompt,
            bbox_threshold=bbox_threshold,
            iou_threshold=api_iou_threshold,
            session_id=None,
        )

        primary_boxes, primary_scores, primary_labels = self._objects_to_arrays(primary_objects)
        keep = nms_xyxy_numpy(primary_boxes, primary_scores, iou_threshold=final_nms_iou) if len(primary_boxes) else []

        detections = []
        accepted = []
        for idx in keep:
            x1, y1, x2, y2 = [int(round(v)) for v in primary_boxes[idx].tolist()]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
                continue

            classify_bbox = self._shrink_bbox(
                (x1, y1, x2, y2),
                image_size=(w, h),
                shrink_ratio=classify_crop_shrink,
                min_crop_size=min_crop_size,
            )
            crop_pil = self._crop_to_pil(image_bgr, classify_bbox)
            class_name, class_confidence, all_scores = self._classify_crop(crop_pil)
            accepted_flag = True if self.min_similarity is None else class_confidence >= self.min_similarity

            det = {
                "bbox": [x1, y1, x2, y2],
                "dino_label": str(primary_labels[idx]).strip(),
                "dino_confidence": float(primary_scores[idx]),
                "predicted_class": class_name,
                "class_confidence": float(class_confidence),
                "classification_bbox": list(classify_bbox),
                "decision_source": "siglip",
                "accepted": accepted_flag,
                "class_scores": all_scores,
            }
            detections.append(det)
            if accepted_flag:
                accepted.append(det)

        override_prompts = list(override_prompts) if override_prompts is not None else list(DEFAULT_OVERRIDE_PROMPTS)
        override_detections = self._run_override_detections(
            image_path=image_path,
            session_id=session_id,
            prompts=override_prompts,
            bbox_threshold=bbox_threshold,
            api_iou_threshold=api_iou_threshold,
            final_nms_iou=final_nms_iou,
        )
        accepted = self._apply_override_detections(
            accepted=accepted,
            override_detections=override_detections,
            iou_threshold=override_iou,
            containment_threshold=override_containment,
        )

        accepted_lookup = {tuple(det["bbox"]): det for det in accepted}
        for det in detections:
            det["accepted"] = tuple(det["bbox"]) in accepted_lookup and det["accepted"]

        result = {
            "image_path": os.path.abspath(image_path),
            "train_root": self.train_root,
            "api_model": self.dino_client.model,
            "siglip_model": self.siglip_model.config.name_or_path,
            "primary_detection_prompt": primary_prompt,
            "override_prompts": override_prompts,
            "bbox_threshold": float(bbox_threshold),
            "api_iou_threshold": float(api_iou_threshold),
            "final_nms_iou": float(final_nms_iou),
            "classify_crop_shrink": float(classify_crop_shrink),
            "min_similarity": self.min_similarity,
            "override_iou": float(override_iou),
            "override_containment": float(override_containment),
            "class_names": self.gallery.class_names,
            "override_detections": override_detections,
            "detections": detections,
            "accepted_detections": accepted,
            "elapsed_time_sec": round(time.perf_counter() - t0, 3),
        }

        if output_path:
            annotated = self._draw_results(image_bgr, accepted)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            cv2.imwrite(output_path, annotated)
            result["output_path"] = os.path.abspath(output_path)

        if save_json_path:
            os.makedirs(os.path.dirname(save_json_path) or ".", exist_ok=True)
            with open(save_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            result["json_path"] = os.path.abspath(save_json_path)

        return result

    def _build_support_gallery(self) -> SupportGallery:
        class_dirs = [
            p for p in sorted(Path(self.train_root).iterdir())
            if p.is_dir() and not p.name.startswith(".")
        ]
        if not class_dirs:
            raise ValueError(f"train_root 下没有类别目录: {self.train_root}")

        class_names = [p.name for p in class_dirs]
        image_paths: List[str] = []
        per_class_features: List[torch.Tensor] = []

        for class_dir in class_dirs:
            class_images = [
                p for p in sorted(class_dir.iterdir())
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            if not class_images:
                raise ValueError(f"类别目录没有图像: {class_dir}")
            class_features = []
            for image_path in class_images:
                image_paths.append(str(image_path))
                image = Image.open(image_path).convert("RGB")
                class_features.append(self._encode_image(image).cpu())
            class_stack = torch.stack(class_features, dim=0).mean(dim=0)
            class_stack = class_stack / class_stack.norm(dim=-1, keepdim=True)
            per_class_features.append(class_stack)

        text_features = self._build_text_features(class_names).cpu()
        image_features = torch.stack(per_class_features, dim=0).cpu()
        support_rows = []
        support_labels = []
        for class_idx, class_dir in enumerate(class_dirs):
            class_images = [
                p for p in sorted(class_dir.iterdir())
                if p.suffix.lower() in IMAGE_EXTENSIONS
            ]
            for image_path in class_images:
                image = Image.open(image_path).convert("RGB")
                support_rows.append(self._encode_image(image).cpu())
                support_labels.append(class_idx)
        support_features = torch.stack(support_rows, dim=0).cpu()
        labels = torch.tensor(support_labels, dtype=torch.long)

        return SupportGallery(
            class_names=class_names,
            image_paths=image_paths,
            image_features=image_features,
            text_features=text_features,
            support_features=support_features,
            labels=labels,
        )

    def _build_text_features(self, class_names: Sequence[str]) -> torch.Tensor:
        texts = []
        for class_name in class_names:
            prompt_name = class_name.replace("_", " ")
            texts.append(f"a photo of a {prompt_name}")

        with torch.no_grad():
            inputs = self.siglip_processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            text_features = self.siglip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _classify_crop(self, image: Image.Image) -> Tuple[str, float, Dict[str, float]]:
        image_feature = self._encode_image(image)
        scores = self._proker_logits(image_feature.unsqueeze(0)).squeeze(0)
        probs = torch.softmax(scores, dim=0)
        class_index = int(torch.argmax(scores).item())
        class_name = self.gallery.class_names[class_index]
        score_map = {
            self.gallery.class_names[i]: float(probs[i].item())
            for i in range(len(self.gallery.class_names))
        }
        return class_name, float(probs[class_index].item()), score_map

    def _build_proker_state(self) -> Dict[str, torch.Tensor]:
        solve_device = "cpu" if str(self.device) == "mps" else self.device
        support = self.gallery.support_features.to(solve_device)
        labels = self.gallery.labels.to(solve_device)
        text_weights = self.gallery.text_features.to(solve_device)
        num_classes = len(self.gallery.class_names)
        cache = F.one_hot(labels, num_classes=num_classes).float()
        logits_text_shots = torch.einsum("sd,cd->sc", support.float(), text_weights.float())
        kernel_ss = self._rbf_kernel(support, support, self.proker_beta)
        alpha = torch.linalg.solve(
            kernel_ss + 0.1 * torch.eye(support.size(0), device=solve_device),
            cache - logits_text_shots,
        )
        return {
            "device": solve_device,
            "support": support,
            "text_weights": text_weights,
            "alpha": alpha,
        }

    def _proker_logits(self, test_features: torch.Tensor) -> torch.Tensor:
        solve_device = self._proker_state["device"]
        support = self._proker_state["support"]
        text_weights = self._proker_state["text_weights"]
        alpha = self._proker_state["alpha"]
        test_features = test_features.to(solve_device)
        kernel_xs = self._rbf_kernel(test_features, support, self.proker_beta)
        logits_text = torch.einsum("bd,cd->bc", test_features.float(), text_weights.float())
        return (logits_text + kernel_xs @ alpha).to(self.device)

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor, beta: float) -> torch.Tensor:
        return torch.exp(-beta * (1 - x.float() @ y.float().T))

    def _estimate_beta(self, support_features: torch.Tensor) -> float:
        if support_features.shape[0] <= 1:
            return 5.0
        sims = support_features @ support_features.T
        dists = 1 - sims
        mask = ~torch.eye(dists.shape[0], dtype=torch.bool)
        valid = dists[mask]
        median_dist = float(torch.median(valid).item()) if valid.numel() else 0.2
        median_dist = max(median_dist, 1e-3)
        return float(np.clip(1.0 / median_dist, 1.0, 100.0))

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
            features = self.siglip_model.get_image_features(**inputs).squeeze(0).float()
        return features / features.norm(dim=-1, keepdim=True)

    def _build_primary_prompt(self, objects: Sequence[str] | str) -> str:
        if isinstance(objects, str):
            terms = [p.strip() for p in objects.replace(",", ".").split(".") if p.strip()]
        else:
            terms = [str(x).strip() for x in objects if str(x).strip()]
        if not terms:
            raise ValueError("需要提供 --objects 作为第一次全检测的 prompt")
        dedup = []
        seen = set()
        for term in terms:
            norm = term.lower().strip().rstrip(".")
            if not norm or norm in seen:
                continue
            seen.add(norm)
            dedup.append(norm)
        return ". ".join(dedup) + "."

    def _run_override_detections(
        self,
        image_path: str,
        session_id: Optional[str],
        prompts: Sequence[str],
        bbox_threshold: float,
        api_iou_threshold: float,
        final_nms_iou: float,
    ) -> List[Dict]:
        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_labels: List[str] = []
        session = session_id
        for prompt in prompts:
            objects, session = self.dino_client.detect(
                image_path=image_path,
                prompt=prompt,
                bbox_threshold=bbox_threshold,
                iou_threshold=api_iou_threshold,
                session_id=session,
            )
            boxes_np, scores_np, labels = self._objects_to_arrays(objects)
            if len(boxes_np) == 0:
                continue
            all_boxes.extend(boxes_np.tolist())
            all_scores.extend(scores_np.tolist())
            all_labels.extend(labels)

        if not all_boxes:
            return []

        boxes_np = np.array(all_boxes, dtype=np.float32)
        scores_np = np.array(all_scores, dtype=np.float32)
        keep = nms_xyxy_numpy(boxes_np, scores_np, iou_threshold=final_nms_iou)
        rows = []
        for idx in keep:
            rows.append({
                "bbox": [int(round(v)) for v in boxes_np[idx].tolist()],
                "predicted_class": str(all_labels[idx]).strip().lower(),
                "class_confidence": float(scores_np[idx]),
                "dino_label": str(all_labels[idx]).strip(),
                "dino_confidence": float(scores_np[idx]),
                "decision_source": "override_detection",
                "accepted": True,
            })
        return rows

    def _apply_override_detections(
        self,
        accepted: Sequence[Dict],
        override_detections: Sequence[Dict],
        iou_threshold: float,
        containment_threshold: float,
    ) -> List[Dict]:
        kept = [dict(det) for det in accepted]
        for override in override_detections:
            override_bbox = tuple(override["bbox"])
            next_kept = []
            for det in kept:
                det_bbox = tuple(det["bbox"])
                inter = self._bbox_intersection_area(det_bbox, override_bbox)
                if inter <= 0:
                    next_kept.append(det)
                    continue
                area_det = max(1.0, self._bbox_area(det_bbox))
                area_override = max(1.0, self._bbox_area(override_bbox))
                union = area_det + area_override - inter
                iou = inter / max(1.0, union)
                containment = inter / min(area_det, area_override)
                if iou >= iou_threshold or containment >= containment_threshold:
                    continue
                next_kept.append(det)
            kept = next_kept
            if tuple(override["bbox"]) not in {tuple(det["bbox"]) for det in kept}:
                kept.append(dict(override))
        return kept

    def _objects_to_arrays(self, objects: Sequence[Dict]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        boxes = []
        scores = []
        labels = []
        for obj in objects:
            bbox = obj.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            boxes.append([float(v) for v in bbox])
            scores.append(float(obj.get("score", 0.0)))
            labels.append(str(obj.get("category", "")).strip())
        boxes_np = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        scores_np = np.array(scores, dtype=np.float32) if scores else np.zeros((0,), dtype=np.float32)
        return boxes_np, scores_np, labels

    def _crop_to_pil(self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = bbox
        crop_bgr = image_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(crop_rgb)

    def _shrink_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        image_size: Tuple[int, int],
        shrink_ratio: float,
        min_crop_size: int,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        bw = x2 - x1
        bh = y2 - y1
        dx = int(round(bw * shrink_ratio * 0.5))
        dy = int(round(bh * shrink_ratio * 0.5))
        nx1 = min(max(0, x1 + dx), img_w - 1)
        ny1 = min(max(0, y1 + dy), img_h - 1)
        nx2 = max(nx1 + min_crop_size, min(img_w, x2 - dx))
        ny2 = max(ny1 + min_crop_size, min(img_h, y2 - dy))
        nx2 = min(nx2, img_w)
        ny2 = min(ny2, img_h)
        return (nx1, ny1, nx2, ny2)

    def _bbox_area(self, bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    def _bbox_intersection_area(
        self,
        a: Tuple[int, int, int, int],
        b: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        return float(max(0, ix2 - ix1) * max(0, iy2 - iy1))

    def _draw_results(self, image_bgr: np.ndarray, detections: Sequence[Dict]) -> np.ndarray:
        canvas = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 200, 60), 2)
            suffix = "override" if det.get("decision_source") == "override_detection" else "siglip"
            text = f"{det['predicted_class']} p={det['class_confidence']:.2f} {suffix}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(0, y1 - th - 6)
            cv2.rectangle(canvas, (x1, y0), (x1 + tw + 4, y1), (30, 200, 60), -1)
            cv2.putText(canvas, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return canvas

    def _auto_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cloud closed-set detector with GroundingDino API + SigLIP")
    parser.add_argument("--image", required=True, help="待检测图像")
    parser.add_argument("--train-root", required=True, help="按类别分目录的 support 图像目录")
    parser.add_argument("--objects", required=True, help="第一次全检测的 prompt，例如 'milk. cup. plate. apple. hot dog. crop.'")
    parser.add_argument("--token", default=os.environ.get("DEEPDATASPACE_TOKEN"), help="DeepDataSpace API token")
    parser.add_argument("--device", default=None, help="显式指定设备: cpu / mps / cuda")
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL, help="SigLIP 模型名")
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL, help="云端 Grounding DINO 模型版本")
    parser.add_argument("--bbox-threshold", type=float, default=0.28)
    parser.add_argument("--api-iou-threshold", type=float, default=0.8)
    parser.add_argument("--api-timeout-sec", type=float, default=180.0)
    parser.add_argument("--api-poll-interval-sec", type=float, default=1.0)
    parser.add_argument("--api-max-retries", type=int, default=4)
    parser.add_argument("--api-retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--final-nms-iou", type=float, default=0.45)
    parser.add_argument("--classify-crop-shrink", type=float, default=0.12)
    parser.add_argument("--min-similarity", type=float, default=None, help="SigLIP 最低保留相似度；不传则不过滤")
    parser.add_argument("--override-prompts", nargs="*", default=list(DEFAULT_OVERRIDE_PROMPTS), help="最后覆盖阶段额外检测的类别词")
    parser.add_argument("--override-iou", type=float, default=0.3)
    parser.add_argument("--override-containment", type=float, default=0.6)
    parser.add_argument("--output", default=None, help="标注图输出路径")
    parser.add_argument("--json", default=None, help="结果 JSON 输出路径")
    args = parser.parse_args()

    detector = ClosedSetObjectDetectorCloud(
        train_root=args.train_root,
        api_token=args.token,
        device=args.device,
        siglip_model_name=args.siglip_model,
        api_model=args.api_model,
        min_similarity=args.min_similarity,
        api_timeout_sec=args.api_timeout_sec,
        api_poll_interval_sec=args.api_poll_interval_sec,
        api_max_retries=args.api_max_retries,
        api_retry_backoff_sec=args.api_retry_backoff_sec,
    )
    result = detector.detect_and_classify(
        image_path=args.image,
        objects=args.objects,
        output_path=args.output,
        save_json_path=args.json,
        bbox_threshold=args.bbox_threshold,
        api_iou_threshold=args.api_iou_threshold,
        final_nms_iou=args.final_nms_iou,
        classify_crop_shrink=args.classify_crop_shrink,
        override_prompts=args.override_prompts,
        override_iou=args.override_iou,
        override_containment=args.override_containment,
    )
    print(json.dumps({
        "accepted": len(result["accepted_detections"]),
        "all_detections": len(result["detections"]),
        "elapsed_time_sec": result["elapsed_time_sec"],
        "api_model": result["api_model"],
        "siglip_model": result["siglip_model"],
        "classes": result["class_names"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

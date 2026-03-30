#!/usr/bin/env python3
"""
Closed-set object detector:
1. Grounding DINO detects candidate boxes.
2. ProKeR-style training-free classification with SigLIP.
3. Optional override detections keep plate/spoon results.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, SiglipModel
from torchvision import transforms

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from .grounding_dino_api import GroundingDINOAPI
    from .utils.bbox_utils import nms_xyxy_numpy
except ImportError:
    from grounding_dino_api import GroundingDINOAPI
    from utils.bbox_utils import nms_xyxy_numpy


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROKER_ROOT = PROJECT_ROOT / "ProKeR"
if str(PROKER_ROOT) not in sys.path:
    sys.path.insert(0, str(PROKER_ROOT))

try:
    from trainers.proker import RBF_Kernel
except ImportError:
    def RBF_Kernel(X, Y, beta):
        return (-beta * (1 - X.float() @ Y.float().T)).exp()


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIGLIP_MODEL = "google/siglip-base-patch16-224"
DEFAULT_SUPPORT_CACHE_DIR = SCRIPT_DIR / ".cache" / "closed_set_support"


@dataclass
class SupportGallery:
    class_names: List[str]
    image_paths: List[str]
    labels: torch.Tensor
    support_features: torch.Tensor
    text_features: torch.Tensor
    whiten_log_likes: torch.Tensor


class ClosedSetObjectDetector:
    """Closed-set detection with DINO + ProKeR-style SigLIP classification."""

    def __init__(
        self,
        train_root: str,
        dino_api: Optional[GroundingDINOAPI] = None,
        device: Optional[str] = None,
        siglip_model_name: str = DEFAULT_SIGLIP_MODEL,
        cache_dir: Optional[str] = None,
        support_cache_dir: Optional[str] = None,
        class_aliases: Optional[Dict[str, Sequence[str]]] = None,
        box_threshold: float = 0.28,
        text_threshold: float = 0.2,
        proker_beta: Optional[float] = None,
        proker_lambda: float = 0.1,
        min_similarity: Optional[float] = None,
        whitening_log_like_threshold: Optional[float] = -13000.0,
        augment_epoch: int = 1,
    ):
        self.train_root = os.path.abspath(train_root)
        self.device = device or self._auto_device()
        self.cache_dir = cache_dir
        self.support_cache_dir = os.path.abspath(support_cache_dir) if support_cache_dir else str(DEFAULT_SUPPORT_CACHE_DIR)
        self.class_aliases = class_aliases or {}
        self.proker_beta = proker_beta
        self.proker_lambda = proker_lambda
        self.min_similarity = min_similarity
        self.whitening_log_like_threshold = whitening_log_like_threshold
        self.augment_epoch = max(0, int(augment_epoch))
        self.dino_api = dino_api or GroundingDINOAPI(
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device,
        )
        self.siglip_model = SiglipModel.from_pretrained(siglip_model_name, cache_dir=self.cache_dir).to(self.device)
        self.siglip_model.eval()
        for param in self.siglip_model.parameters():
            param.requires_grad = False
        try:
            self.siglip_processor = AutoProcessor.from_pretrained(siglip_model_name, cache_dir=self.cache_dir)
        except ImportError as exc:
            raise ImportError(
                "SigLIP 需要 `sentencepiece`。请先安装: "
                "`pip install sentencepiece` 或 `conda install -c conda-forge sentencepiece`"
            ) from exc
        self.support_augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
        self.gallery = self._build_support_gallery()
        self.proker_beta = self.proker_beta or self._estimate_beta(self.gallery.support_features)
        self.whiten_mean, self.whiten_mat = self._build_whitening_stats(self.gallery.support_features)
        self._proker_state = self._build_proker_state()

    def detect_and_classify(
        self,
        image_path: str,
        objects: Optional[Sequence[str] | str] = None,
        output_path: Optional[str] = None,
        save_json_path: Optional[str] = None,
        detection_terms: Optional[Sequence[str]] = None,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        nms_iou: float = 0.45,
        min_crop_size: int = 8,
        classify_crop_shrink: float = 0.15,
        overlap_containment_ratio: float = 0.8,
        overlap_area_ratio: float = 1.35,
        override_prompts: Optional[Sequence[str]] = None,
        override_box_threshold: float = 0.45,
        override_text_threshold: Optional[float] = None,
        override_iou: float = 0.3,
        override_containment: float = 0.6,
    ) -> Dict:
        t0 = time.perf_counter()
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        generic_prompt = self._build_primary_detection_prompt(objects=objects, detection_terms=detection_terms)
        boxes_np, scores_np, labels = self._run_detection_passes(
            image_path=image_path,
            prompt=generic_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        keep = nms_xyxy_numpy(boxes_np, scores_np, iou_threshold=nms_iou) if len(boxes_np) else []

        detections = []
        accepted = []
        h, w = image_bgr.shape[:2]
        for idx in keep:
            x1, y1, x2, y2 = [int(round(v)) for v in boxes_np[idx].tolist()]
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
            image_feature = self._encode_crop(crop_pil)
            class_scores = self._proker_logits(image_feature.unsqueeze(0)).squeeze(0)
            probs = torch.softmax(class_scores, dim=0)
            class_index = int(torch.argmax(class_scores).item())
            class_name = self.gallery.class_names[class_index]
            class_confidence = float(probs[class_index].item())
            whiten_log_like = self._compute_whiten_log_like(image_feature)
            accepted_flag = True
            if self.min_similarity is not None:
                accepted_flag = accepted_flag and class_confidence >= float(self.min_similarity)
            if self.whitening_log_like_threshold is not None:
                accepted_flag = accepted_flag and whiten_log_like >= float(self.whitening_log_like_threshold)

            det = {
                "bbox": [x1, y1, x2, y2],
                "dino_label": str(labels[idx]).strip(),
                "dino_confidence": float(scores_np[idx]),
                "predicted_class": class_name,
                "class_confidence": class_confidence,
                "classification_confidence": class_confidence,
                "whitening_log_like": whiten_log_like,
                "classification_bbox": list(classify_bbox),
                "decision_source": "siglip",
                "accepted": accepted_flag,
            }
            detections.append(det)
            if accepted_flag:
                accepted.append(det)

        accepted = self._filter_ambiguous_overlaps(
            accepted,
            containment_ratio=overlap_containment_ratio,
            area_ratio=overlap_area_ratio,
        )
        override_prompts = list(override_prompts) if override_prompts is not None else ["plate.", "spoon."]
        override_detections = self._run_override_detections(
            image_path=image_path,
            prompts=override_prompts,
            box_threshold=override_box_threshold,
            text_threshold=override_text_threshold if override_text_threshold is not None else text_threshold,
            nms_iou=nms_iou,
        )
        accepted = self._apply_override_detections(
            accepted=accepted,
            override_detections=override_detections,
            iou_threshold=override_iou,
            containment_threshold=override_containment,
        )
        accepted_lookup = {tuple(det["bbox"]): det for det in accepted}
        for det in detections:
            det["accepted"] = tuple(det["bbox"]) in accepted_lookup

        result = {
            "image_path": os.path.abspath(image_path),
            "train_root": self.train_root,
            "primary_detection_prompt": generic_prompt,
            "class_names": self.gallery.class_names,
            "proker_beta": float(self.proker_beta),
            "proker_lambda": float(self.proker_lambda),
            "siglip_model": self.siglip_model.config.name_or_path,
            "min_similarity": self.min_similarity,
            "whitening_log_like_threshold": self.whitening_log_like_threshold,
            "augment_epoch": int(self.augment_epoch),
            "support_whitening_log_like_stats": {
                "min": float(self.gallery.whiten_log_likes.min().item()),
                "max": float(self.gallery.whiten_log_likes.max().item()),
                "mean": float(self.gallery.whiten_log_likes.mean().item()),
            },
            "classify_crop_shrink": float(classify_crop_shrink),
            "overlap_containment_ratio": float(overlap_containment_ratio),
            "overlap_area_ratio": float(overlap_area_ratio),
            "override_prompts": list(override_prompts),
            "override_box_threshold": float(override_box_threshold),
            "override_text_threshold": float(override_text_threshold if override_text_threshold is not None else (text_threshold if text_threshold is not None else 0.2)),
            "override_iou": float(override_iou),
            "override_containment": float(override_containment),
            "override_detections": override_detections,
            "detections": detections,
            "accepted_detections": accepted,
        }
        result["elapsed_time_sec"] = round(time.perf_counter() - t0, 3)

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
        label_ids: List[int] = []

        for class_id, class_dir in enumerate(class_dirs):
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                image_paths.append(str(file_path))
                label_ids.append(class_id)

        if not image_paths:
            raise ValueError(f"train_root 下没有可用图像: {self.train_root}")

        support_features, labels = self._load_or_build_augmented_support_features(image_paths, label_ids)
        text_features = self._build_text_features(class_names).cpu().float()
        whiten_mean, whiten_mat = self._build_whitening_stats(support_features)
        whiten_log_likes = self._compute_whiten_log_like_batch(
            support_features,
            whiten_mean=whiten_mean,
            whiten_mat=whiten_mat,
        ).cpu().float()

        gallery = SupportGallery(
            class_names=class_names,
            image_paths=image_paths,
            labels=labels,
            support_features=support_features,
            text_features=text_features,
            whiten_log_likes=whiten_log_likes,
        )
        return gallery

    def _load_or_build_augmented_support_features(
        self,
        image_paths: Sequence[str],
        label_ids: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        os.makedirs(self.support_cache_dir, exist_ok=True)
        cache_path = os.path.join(self.support_cache_dir, self._support_cache_name(image_paths, label_ids))
        if os.path.exists(cache_path):
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            return cache["support_features"].float(), cache["labels"].long()

        support_rows: List[torch.Tensor] = []
        support_labels: List[int] = []
        for image_path, label_id in zip(image_paths, label_ids):
            image = Image.open(image_path).convert("RGB")
            support_rows.append(self._encode_crop(image).cpu())
            support_labels.append(int(label_id))
            for _ in range(self.augment_epoch):
                aug_image = self.support_augment_transform(image)
                support_rows.append(self._encode_crop(aug_image).cpu())
                support_labels.append(int(label_id))

        support_features = torch.stack(support_rows, dim=0).float()
        labels = torch.tensor(support_labels, dtype=torch.long)
        torch.save(
            {
                "support_features": support_features.cpu(),
                "labels": labels.cpu(),
                "augment_epoch": int(self.augment_epoch),
                "siglip_model": self.siglip_model.config.name_or_path,
                "train_root": self.train_root,
                "image_paths": list(image_paths),
            },
            cache_path,
        )
        return support_features, labels

    def _support_cache_name(self, image_paths: Sequence[str], label_ids: Sequence[int]) -> str:
        signature_rows = []
        for image_path, label_id in zip(image_paths, label_ids):
            stat = os.stat(image_path)
            signature_rows.append(
                f"{image_path}|{label_id}|{stat.st_size}|{int(stat.st_mtime_ns)}"
            )
        payload = json.dumps(
            {
                "train_root": self.train_root,
                "siglip_model": self.siglip_model.config.name_or_path,
                "augment_epoch": int(self.augment_epoch),
                "rows": signature_rows,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"support_siglip_aug_{digest}.pt"

    def _build_proker_state(self) -> Dict[str, torch.Tensor | str]:
        solve_device = "cpu" if str(self.device) == "mps" else self.device
        vecs = self.gallery.support_features.to(solve_device)
        labels = self.gallery.labels.to(solve_device)
        text_weights = self.gallery.text_features.to(solve_device)
        n_classes = len(self.gallery.class_names)
        cache = F.one_hot(labels, num_classes=n_classes).float()
        logits_text_shots = torch.einsum("sd,cd->sc", vecs.float(), text_weights.float())
        kernel_ss = RBF_Kernel(vecs, vecs, beta=self.proker_beta)
        alpha = torch.linalg.solve(
            (1.0 / self.proker_lambda) * kernel_ss + torch.eye(vecs.size(0), device=solve_device),
            cache - logits_text_shots,
        )
        return {
            "device": solve_device,
            "vecs": vecs,
            "text_weights": text_weights,
            "alpha": alpha,
        }

    def _build_text_features(self, class_names: Sequence[str]) -> torch.Tensor:
        texts = []
        for class_name in class_names:
            prompt_name = class_name.replace("_", " ")
            texts.append(f"a photo of a {prompt_name}")
        with torch.no_grad():
            inputs = self.siglip_processor(text=texts, padding="max_length", return_tensors="pt").to(self.device)
            text_features = self.siglip_model.get_text_features(**inputs).float()
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def _proker_logits(self, test_features: torch.Tensor) -> torch.Tensor:
        solve_device = self._proker_state["device"]
        vecs = self._proker_state["vecs"]
        text_weights = self._proker_state["text_weights"]
        alpha = self._proker_state["alpha"]
        test_features = test_features.to(solve_device)
        kernel_xs = RBF_Kernel(test_features, vecs, beta=self.proker_beta)
        logits_text = torch.einsum("bd,cd->bc", test_features.float(), text_weights.float())
        return (logits_text + kernel_xs @ alpha).to(self.device)

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

    def _build_whitening_stats(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = features.detach().cpu().float()
        mean = feats.mean(dim=0)
        centered = feats - mean
        dim = centered.shape[-1]
        if centered.shape[0] <= 1:
            return mean, torch.eye(dim, dtype=torch.float32)
        cov = centered.T @ centered / max(1, centered.shape[0] - 1)
        cov = cov + 1e-5 * torch.eye(dim, dtype=torch.float32)
        evals, evecs = torch.linalg.eigh(cov)
        evals = torch.clamp(evals, min=1e-8)
        inv_sqrt = torch.diag(torch.rsqrt(evals))
        whiten_mat = evecs @ inv_sqrt
        return mean, whiten_mat.float()

    def _compute_whiten_log_like(
        self,
        feature: torch.Tensor,
        whiten_mean: Optional[torch.Tensor] = None,
        whiten_mat: Optional[torch.Tensor] = None,
    ) -> float:
        feats = feature.detach().cpu().float().unsqueeze(0)
        values = self._compute_whiten_log_like_batch(
            feats,
            whiten_mean=whiten_mean,
            whiten_mat=whiten_mat,
        )
        return float(values[0].item())

    def _compute_whiten_log_like_batch(
        self,
        features: torch.Tensor,
        whiten_mean: Optional[torch.Tensor] = None,
        whiten_mat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mean = (whiten_mean if whiten_mean is not None else self.whiten_mean).detach().cpu().float()
        w_mat = (whiten_mat if whiten_mat is not None else self.whiten_mat).detach().cpu().float()
        feats = features.detach().cpu().float()
        centered = feats - mean.unsqueeze(0)
        whitened = centered @ w_mat
        dim = whitened.shape[-1]
        log_two_pi = float(np.log(2.0 * np.pi))
        return -0.5 * (dim * log_two_pi + torch.sum(whitened ** 2, dim=-1))

    def _build_primary_detection_prompt(
        self,
        objects: Optional[Sequence[str] | str],
        detection_terms: Optional[Sequence[str]],
    ) -> str:
        terms: List[str] = []
        if isinstance(objects, str):
            terms.extend([p.strip() for p in objects.replace(",", ".").split(".") if p.strip()])
        elif objects:
            terms.extend([str(x).strip() for x in objects if str(x).strip()])
        if detection_terms:
            terms.extend([str(x).strip() for x in detection_terms if str(x).strip()])
        if not terms:
            terms = ["object"]
        dedup = []
        seen = set()
        for term in terms:
            norm = term.lower().strip().rstrip(".")
            if not norm or norm in seen:
                continue
            seen.add(norm)
            dedup.append(norm)
        return ". ".join(dedup) + "."

    def _run_detection_passes(
        self,
        image_path: str,
        prompt: str,
        box_threshold: Optional[float],
        text_threshold: Optional[float],
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        dino_result = self.dino_api.detect(
            image_path=image_path,
            text_prompt=prompt,
            output_path=None,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        boxes_np = np.array(dino_result["boxes"], dtype=np.float32) if dino_result["boxes"] else np.zeros((0, 4), dtype=np.float32)
        scores_np = np.array(dino_result["confidences"], dtype=np.float32) if dino_result["confidences"] else np.zeros((0,), dtype=np.float32)
        labels = [str(x).strip() for x in dino_result["labels"]]
        return boxes_np, scores_np, labels

    def _run_override_detections(
        self,
        image_path: str,
        prompts: Sequence[str],
        box_threshold: Optional[float],
        text_threshold: Optional[float],
        nms_iou: float,
    ) -> List[Dict]:
        all_boxes: List[List[float]] = []
        all_scores: List[float] = []
        all_labels: List[str] = []
        for prompt in prompts:
            boxes_np, scores_np, labels = self._run_detection_passes(
                image_path=image_path,
                prompt=prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if len(boxes_np) == 0:
                continue
            all_boxes.extend(boxes_np.tolist())
            all_scores.extend(scores_np.tolist())
            all_labels.extend(labels)
        if not all_boxes:
            return []
        boxes_np = np.array(all_boxes, dtype=np.float32)
        scores_np = np.array(all_scores, dtype=np.float32)
        keep = nms_xyxy_numpy(boxes_np, scores_np, iou_threshold=nms_iou)
        rows = []
        for idx in keep:
            rows.append({
                "bbox": [int(round(v)) for v in boxes_np[idx].tolist()],
                "predicted_class": str(all_labels[idx]).strip().lower(),
                "class_confidence": float(scores_np[idx]),
                "classification_confidence": float(scores_np[idx]),
                "whitening_log_like": None,
                "dino_label": str(all_labels[idx]).strip(),
                "dino_confidence": float(scores_np[idx]),
                "classification_bbox": None,
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

    def _filter_ambiguous_overlaps(
        self,
        detections: Sequence[Dict],
        containment_ratio: float,
        area_ratio: float,
    ) -> List[Dict]:
        kept = [dict(det) for det in detections]
        remove_ids = set()
        for i, det_i in enumerate(kept):
            if i in remove_ids:
                continue
            box_i = tuple(det_i["bbox"])
            area_i = self._bbox_area(box_i)
            for j, det_j in enumerate(kept):
                if i == j or j in remove_ids:
                    continue
                box_j = tuple(det_j["bbox"])
                area_j = self._bbox_area(box_j)
                if area_i <= area_j:
                    continue
                inter = self._bbox_intersection_area(box_i, box_j)
                if inter <= 0:
                    continue
                smaller_area = max(1.0, float(area_j))
                contains_smaller = inter / smaller_area >= containment_ratio
                too_large = float(area_i) / max(1.0, float(area_j)) >= area_ratio
                if not (contains_smaller and too_large):
                    continue

                prefer_j = (
                    float(det_j["class_confidence"]) >= float(det_i["class_confidence"]) or
                    float(det_j["dino_confidence"]) >= float(det_i["dino_confidence"])
                )
                if prefer_j:
                    remove_ids.add(i)
                    break

        return [det for idx, det in enumerate(kept) if idx not in remove_ids]

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

    def _encode_crop(self, image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            inputs = self.siglip_processor(images=image, return_tensors="pt").to(self.device)
            features = self.siglip_model.get_image_features(**inputs).squeeze(0).float()
        return (features / features.norm(dim=-1, keepdim=True)).detach()

    def _draw_results(self, image_bgr: np.ndarray, detections: Sequence[Dict]) -> np.ndarray:
        canvas = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (40, 200, 40), 2)
            if det.get("decision_source") == "override_detection":
                text = (
                    f"{det['predicted_class']} "
                    f"conf={det['classification_confidence']:.2f} "
                    f"override"
                )
            else:
                text = (
                    f"{det['predicted_class']} "
                    f"conf={det['classification_confidence']:.2f} "
                    f"ll={det['whitening_log_like']:.1f}"
                )
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            y0 = max(0, y1 - th - 6)
            cv2.rectangle(canvas, (x1, y0), (x1 + tw + 4, y1), (40, 200, 40), -1)
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

    parser = argparse.ArgumentParser(description="Closed-set object detector")
    parser.add_argument("--image", required=True, help="待检测图像")
    parser.add_argument("--train-root", required=True, help="按类别分目录的 support 图像目录")
    parser.add_argument("--objects", default=None, help="Grounding DINO 检测词，例如 'milk. cup. plate.'")
    parser.add_argument("--output", default=None, help="标注图输出路径")
    parser.add_argument("--json", default=None, help="结果 JSON 输出路径")
    parser.add_argument("--device", default=None, help="显式指定设备: cpu / mps / cuda")
    parser.add_argument("--siglip-model", default=DEFAULT_SIGLIP_MODEL, help="SigLIP 模型名")
    parser.add_argument("--support-cache-dir", default=str(DEFAULT_SUPPORT_CACHE_DIR), help="增强后的 support 特征缓存目录")
    parser.add_argument("--box-threshold", type=float, default=0.28)
    parser.add_argument("--text-threshold", type=float, default=0.2)
    parser.add_argument("--proker-beta", type=float, default=None)
    parser.add_argument("--proker-lambda", type=float, default=0.1)
    parser.add_argument("--augment-epoch", type=int, default=1, help="support 集随机增强轮数；首次生成后缓存复用")
    parser.add_argument("--min-similarity", type=float, default=None, help="SigLIP 分类最低保留概率；不传则不过滤")
    parser.add_argument("--whitening-log-like-threshold", type=float, default=-9000.0, help="白化 log-likelihood 低于该值的分类框会被删除")
    parser.add_argument("--classify-crop-shrink", type=float, default=0.15, help="分类时对检测框做中心收缩，减少相邻物体干扰")
    parser.add_argument("--overlap-containment-ratio", type=float, default=0.8, help="较大框覆盖较小框达到该比例时，认为大框可能是混合框")
    parser.add_argument("--overlap-area-ratio", type=float, default=1.35, help="较大框面积至少是较小框的该倍数时，才触发混合框过滤")
    parser.add_argument("--override-prompts", nargs="*", default=["plate.", "spoon."], help="最后覆盖阶段额外检测的类别词")
    parser.add_argument("--override-box-threshold", type=float, default=0.45, help="plate/spoon 覆盖检测的 box 阈值")
    parser.add_argument("--override-text-threshold", type=float, default=None, help="plate/spoon 覆盖检测的 text 阈值；不传则沿用主 text threshold")
    parser.add_argument("--override-iou", type=float, default=0.3, help="覆盖阶段删除重叠旧框的 IoU 阈值")
    parser.add_argument("--override-containment", type=float, default=0.6, help="覆盖阶段删除重叠旧框的包含率阈值")
    args = parser.parse_args()

    detector = ClosedSetObjectDetector(
        train_root=args.train_root,
        device=args.device,
        siglip_model_name=args.siglip_model,
        support_cache_dir=args.support_cache_dir,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        proker_beta=args.proker_beta,
        proker_lambda=args.proker_lambda,
        augment_epoch=args.augment_epoch,
        min_similarity=args.min_similarity,
        whitening_log_like_threshold=args.whitening_log_like_threshold,
    )
    result = detector.detect_and_classify(
        image_path=args.image,
        objects=args.objects,
        output_path=args.output,
        save_json_path=args.json,
        classify_crop_shrink=args.classify_crop_shrink,
        overlap_containment_ratio=args.overlap_containment_ratio,
        overlap_area_ratio=args.overlap_area_ratio,
        override_prompts=args.override_prompts,
        override_box_threshold=args.override_box_threshold,
        override_text_threshold=args.override_text_threshold,
        override_iou=args.override_iou,
        override_containment=args.override_containment,
    )
    print(json.dumps({
        "accepted": len(result["accepted_detections"]),
        "all_detections": len(result["detections"]),
        "siglip_model": result["siglip_model"],
        "min_similarity": result["min_similarity"],
        "whitening_log_like_threshold": result["whitening_log_like_threshold"],
        "augment_epoch": result["augment_epoch"],
        "elapsed_time_sec": result["elapsed_time_sec"],
        "classes": result["class_names"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

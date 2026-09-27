from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from .types import Detection


def rgb_to_ultralytics_source(image: np.ndarray) -> np.ndarray:
    """Convert our RGB raster array to the BGR ndarray expected by Ultralytics."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("L'image d'inférence doit avoir la forme (hauteur, largeur, 3)")
    return np.ascontiguousarray(image[..., ::-1])


def scale_box(
    box: tuple[float, float, float, float],
    factor: float,
) -> tuple[float, float, float, float]:
    """Resize a box around its centre.

    Models trained with free rotation predict boxes inflated by roughly 1.27,
    because Ultralytics re-encloses rotated corners in an axis-aligned box.
    """
    if factor == 1.0:
        return box
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    half_width = (x2 - x1) * factor / 2
    half_height = (y2 - y1) * factor / 2
    return (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def aspect_ratio(box: tuple[float, float, float, float]) -> float:
    """Ratio du grand côté sur le petit, toujours supérieur ou égal à 1."""
    x1, y1, x2, y2 = box
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    if width <= 0 or height <= 0:
        return math.inf
    return max(width, height) / min(width, height)


class YoloDetector:
    """Small adapter around Ultralytics, loaded only when inference starts."""

    def __init__(
        self,
        model_path: Path | str,
        device: str = "cpu",
        box_scale: float = 1.0,
        max_aspect_ratio: float = math.inf,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.device = device
        self.box_scale = box_scale
        self.max_aspect_ratio = max_aspect_ratio

    def predict(self, image: np.ndarray, confidence: float) -> list[Detection]:
        kwargs: dict[str, object] = {
            "source": rgb_to_ultralytics_source(image),
            "conf": confidence,
            "verbose": False,
        }
        if self.device and self.device.lower() != "auto":
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(int)
        for box, score, class_id in zip(boxes, confidences, classes, strict=True):
            class_name = names.get(int(class_id), str(class_id))
            x1, y1, x2, y2 = scale_box(
                (float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                self.box_scale,
            )
            if aspect_ratio((x1, y1, x2, y2)) > self.max_aspect_ratio:
                continue
            detections.append(
                Detection(
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(score),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )
        return detections


class CascadeYoloDetector:
    """Use a detector for proposals, then classify every crop."""

    CLASS_IDS = {
        "PlantingHoleEmpty": 0,
        "PlantingHoleOccupied": 1,
    }

    def __init__(
        self,
        detector_path: Path | str,
        classifier_path: Path | str,
        device: str = "cpu",
        classifier_confidence: float = 0.0,
        detector_nms_iou: float = 0.70,
        crop_context: float = 1.8,
        box_scale: float = 1.0,
        max_aspect_ratio: float = math.inf,
    ) -> None:
        from ultralytics import YOLO

        self.detector = YOLO(str(detector_path))
        self.classifier = YOLO(str(classifier_path))
        self.device = device
        self.classifier_confidence = classifier_confidence
        self.detector_nms_iou = detector_nms_iou
        self.crop_context = crop_context
        self.box_scale = box_scale
        self.max_aspect_ratio = max_aspect_ratio

    def _crop(
        self,
        image: Image.Image,
        box: tuple[float, float, float, float],
    ) -> Image.Image:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        size = max(24.0, max(x2 - x1, y2 - y1) * self.crop_context)
        return image.crop(
            (
                round(center_x - size / 2),
                round(center_y - size / 2),
                round(center_x + size / 2),
                round(center_y + size / 2),
            )
        ).resize((224, 224), Image.Resampling.LANCZOS)

    def predict(self, image: np.ndarray, confidence: float) -> list[Detection]:
        detector_kwargs: dict[str, object] = {
            "source": rgb_to_ultralytics_source(image),
            "conf": confidence,
            "iou": self.detector_nms_iou,
            "agnostic_nms": True,
            "verbose": False,
        }
        if self.device and self.device.lower() != "auto":
            detector_kwargs["device"] = self.device
        detector_result = self.detector.predict(**detector_kwargs)[0]
        if detector_result.boxes is None or len(detector_result.boxes) == 0:
            return []

        # Rescaling before cropping also restores the crop context the
        # classifier was trained on.
        boxes: list[tuple[float, float, float, float]] = []
        detector_confidences: list[float] = []
        for row, score in zip(
            detector_result.boxes.xyxy.detach().cpu().tolist(),
            detector_result.boxes.conf.detach().cpu().tolist(),
            strict=True,
        ):
            box = scale_box(tuple(float(value) for value in row), self.box_scale)
            # Écarter ici plutôt qu'après évite de classifier des vignettes inutiles.
            if aspect_ratio(box) > self.max_aspect_ratio:
                continue
            boxes.append(box)
            detector_confidences.append(float(score))
        if not boxes:
            return []

        pil_image = Image.fromarray(image, mode="RGB")
        crops = [self._crop(pil_image, box) for box in boxes]
        classifier_kwargs: dict[str, object] = {
            "source": crops,
            "imgsz": 224,
            "batch": 64,
            "verbose": False,
        }
        if self.device and self.device.lower() != "auto":
            classifier_kwargs["device"] = self.device
        classifier_results = self.classifier.predict(**classifier_kwargs)

        detections: list[Detection] = []
        for box, detector_score, classification in zip(
            boxes,
            detector_confidences,
            classifier_results,
            strict=True,
        ):
            class_index = int(classification.probs.top1)
            class_name = str(classification.names[class_index])
            class_score = float(classification.probs.top1conf)
            class_id = self.CLASS_IDS.get(class_name)
            if class_id is None or class_score < self.classifier_confidence:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=detector_score * class_score,
                    x1=box[0],
                    y1=box[1],
                    x2=box[2],
                    y2=box[3],
                )
            )
        return detections


def intersection_over_union(a: Detection, b: Detection) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def global_class_agnostic_nms(
    detections: Sequence[Detection],
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Keep the strongest box regardless of class for overlapping sites."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold doit être compris entre 0 et 1")
    if not detections:
        return []

    # Spatial buckets keep this exact greedy NMS practical for large plantations.
    # Any two boxes with a non-zero intersection share at least one grid cell.
    cell_size = 64.0
    ordered = sorted(
        enumerate(detections),
        key=lambda item: (-item[1].confidence, item[0]),
    )
    kept: list[Detection] = []
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)

    def occupied_cells(detection: Detection) -> list[tuple[int, int]]:
        min_column = math.floor(detection.x1 / cell_size)
        max_column = math.floor(max(detection.x1, detection.x2 - 1e-9) / cell_size)
        min_row = math.floor(detection.y1 / cell_size)
        max_row = math.floor(max(detection.y1, detection.y2 - 1e-9) / cell_size)
        return [
            (row, column)
            for row in range(min_row, max_row + 1)
            for column in range(min_column, max_column + 1)
        ]

    for _, candidate in ordered:
        cells = occupied_cells(candidate)
        possible_matches = {
            kept_index
            for cell in cells
            for kept_index in buckets.get(cell, ())
        }
        if any(
            intersection_over_union(candidate, kept[kept_index]) > iou_threshold
            for kept_index in possible_matches
        ):
            continue
        new_index = len(kept)
        kept.append(candidate)
        for cell in cells:
            buckets[cell].append(new_index)
    return kept

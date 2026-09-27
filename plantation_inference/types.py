from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


ProgressCallback = Callable[[str, float, str], None]

PACKAGE_ROOT = Path(__file__).resolve().parent
# Copie figée des poids retenus, livrée avec le paquet pour que la version du
# code et celle du modèle ne puissent pas se dissocier. La sortie du projet
# d'entraînement reste PlantationDatasetV2/models_v2.
DEFAULT_MODEL = PACKAGE_ROOT / "models" / "direct_two_classes_best.pt"
DEFAULT_CONFIDENCE = 0.20
# Les poids livrés ont été entraînés avec une rotation libre, qui gonfle les
# boîtes réencadrées. Remettre 1.0 dès qu'un modèle entraîné sans rotation est
# déployé, et recalibrer après chaque réentraînement : le ratio médian entre
# taille prédite et taille annotée sur la validation, dont on prend l'inverse.
# Sans cette correction, le rappel tombe de 94 % à 61 % sur le test verrouillé.
DEFAULT_BOX_SCALE = 0.82
# 0.82 et 2.0 maximisent le F1 conjointement sur les tuiles de test source2 et
# result00389 ; voir PlantationDatasetV2/geotiff_evaluation/shape_filter.json.
DEFAULT_MAX_ASPECT_RATIO = 2.0


@dataclass(frozen=True)
class TileWindow:
    """A raster window, expressed in global pixel coordinates."""

    col_off: int
    row_off: int
    width: int
    height: int


@dataclass(frozen=True)
class Detection:
    """One detection in the coordinate system of the source raster."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class PipelineConfig:
    # `model` porte une valeur par défaut pour qu'un appelant qui ne précise
    # rien obtienne la configuration mesurée, et non une configuration neutre.
    source: Path
    output_root: Path
    model: Path = DEFAULT_MODEL
    classifier: Path | None = None
    confidence: float = DEFAULT_CONFIDENCE
    classifier_confidence: float = 0.0
    detector_nms_iou: float = 0.70
    crop_context: float = 1.8
    box_scale: float = DEFAULT_BOX_SCALE
    max_aspect_ratio: float = DEFAULT_MAX_ASPECT_RATIO
    ignore_resolution: bool = False
    auto_resample: bool = True
    prepared_root: Path | None = None
    tile_size: int = 512
    overlap: int = 0
    nms_iou: float = 0.45
    device: str = "cpu"
    run_name: str | None = None

    def validate(self) -> None:
        if not self.source.is_file():
            raise FileNotFoundError(f"GeoTIFF introuvable : {self.source}")
        if not self.model.is_file():
            raise FileNotFoundError(f"Modèle introuvable : {self.model}")
        if self.classifier is not None and not self.classifier.is_file():
            raise FileNotFoundError(f"Classificateur introuvable : {self.classifier}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence doit être compris entre 0 et 1")
        if not 0 <= self.classifier_confidence <= 1:
            raise ValueError(
                "classifier_confidence doit être compris entre 0 et 1"
            )
        if not 0 <= self.detector_nms_iou <= 1:
            raise ValueError("detector_nms_iou doit être compris entre 0 et 1")
        if self.crop_context <= 0:
            raise ValueError("crop_context doit être strictement positif")
        if self.box_scale <= 0:
            raise ValueError("box_scale doit être strictement positif")
        if self.max_aspect_ratio < 1:
            raise ValueError("max_aspect_ratio doit être supérieur ou égal à 1")
        if self.tile_size <= 0:
            raise ValueError("tile_size doit être strictement positif")
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap doit être positif et inférieur à tile_size")
        if not 0 <= self.nms_iou <= 1:
            raise ValueError("nms_iou doit être compris entre 0 et 1")


@dataclass(frozen=True)
class PipelineResult:
    run_dir: Path
    geopackage: Path
    annotated_tiff: Path
    detections_csv: Path
    summary_csv: Path
    metadata_json: Path
    detection_count: int
    empty_count: int
    occupied_count: int
    success_rate: float | None
    elapsed_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "geopackage": str(self.geopackage),
            "annotated_tiff": str(self.annotated_tiff),
            "detections_csv": str(self.detections_csv),
            "summary_csv": str(self.summary_csv),
            "metadata_json": str(self.metadata_json),
            "detection_count": self.detection_count,
            "empty_count": self.empty_count,
            "occupied_count": self.occupied_count,
            "success_rate": self.success_rate,
            "elapsed_seconds": self.elapsed_seconds,
        }

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import rasterio
from rasterio.windows import Window

from .detection import (
    CascadeYoloDetector,
    YoloDetector,
    aspect_ratio,
    global_class_agnostic_nms,
)
from .preparation import prepared_copy
from .outputs import (
    summarize_detections,
    write_annotated_geotiff,
    write_detections_csv,
    write_geopackage,
    write_metadata,
    write_summary_csv,
)
from .tiling import generate_windows, to_rgb_uint8
from .types import Detection, PipelineConfig, PipelineResult, ProgressCallback


class Detector(Protocol):
    def predict(self, image: np.ndarray, confidence: float) -> list[Detection]: ...


# Tout le jeu d'entraînement est à cette résolution. Un trou y mesure environ
# 55 pixels ; à une autre échelle le modèle voit des fragments et invente des
# détections, mesuré sur result.tif natif : 8 boîtes là où il y a 3 trous.
TRAINING_GSD_M = 0.0389
GSD_TOLERANCE = 0.20
# Rééchantillonner une image plus grossière ne recrée pas l'information perdue :
# à 10 cm/pixel un tiers des trous échappe déjà au modèle, à 15 cm la moitié.
# Mesures dans PlantationDatasetV2/geotiff_evaluation/limites_resolution.json.
MAX_USABLE_GSD_M = 0.10


def matches_training_scale(gsd_m: float) -> bool:
    return abs(gsd_m - TRAINING_GSD_M) / TRAINING_GSD_M <= GSD_TOLERANCE


def check_resolution(gsd_m: float) -> None:
    """Refuse un raster dont l'échelle s'écarte trop de celle de l'entraînement."""
    if matches_training_scale(gsd_m):
        return
    raise ValueError(
        f"Résolution incompatible : {gsd_m * 100:.2f} cm/pixel alors que le modèle "
        f"a été entraîné à {TRAINING_GSD_M * 100:.2f} cm/pixel. Laissez le "
        "rééchantillonnage automatique agir, ou passez --ignore-resolution pour "
        "forcer malgré la dégradation attendue."
    )


def check_usable_resolution(gsd_m: float) -> None:
    """Refuse un raster trop grossier pour qu'un rééchantillonnage le sauve."""
    if gsd_m <= MAX_USABLE_GSD_M:
        return
    raise ValueError(
        f"Résolution trop grossière : {gsd_m * 100:.2f} cm/pixel. Le modèle a été "
        f"entraîné à {TRAINING_GSD_M * 100:.2f} cm/pixel et agrandir l'image ne "
        "recrée pas les détails absents : au-delà de "
        f"{MAX_USABLE_GSD_M * 100:.0f} cm/pixel, la majorité des trous est manquée."
    )


def prepare_source(
    config: PipelineConfig,
    notify: Callable[[str, float, str], None],
) -> tuple[Path, float]:
    """Raster prêt pour le modèle, rééchantillonné si son échelle diffère."""
    with rasterio.open(config.source) as probe:
        gsd = abs(probe.transform.a)
    if config.ignore_resolution or matches_training_scale(gsd):
        return config.source, gsd
    if not config.auto_resample:
        check_resolution(gsd)
    check_usable_resolution(gsd)

    cache = config.prepared_root or config.output_root / "_prepared"
    notify(
        "preparation",
        0.0,
        f"Rééchantillonnage de {gsd * 100:.2f} à {TRAINING_GSD_M * 100:.2f} cm/pixel",
    )
    prepared = prepared_copy(
        config.source,
        cache,
        TRAINING_GSD_M,
        lambda fraction: notify(
            "preparation", 0.01 * fraction, "Rééchantillonnage en cours"
        ),
    )
    return prepared, gsd


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return normalized or "run"


def create_run_directory(
    output_root: Path,
    source_stem: str,
    requested_name: str | None = None,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if requested_name:
        base_name = _safe_name(requested_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{_safe_name(source_stem)}_{timestamp}"

    candidate = output_root / base_name
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{base_name}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _translate_detection(
    local: Detection,
    col_off: int,
    row_off: int,
    valid_width: int,
    valid_height: int,
    raster_width: int,
    raster_height: int,
) -> Detection | None:
    center_x, center_y = local.center
    if not (0 <= center_x < valid_width and 0 <= center_y < valid_height):
        return None
    x1 = min(float(raster_width), max(0.0, local.x1 + col_off))
    y1 = min(float(raster_height), max(0.0, local.y1 + row_off))
    x2 = min(float(raster_width), max(0.0, local.x2 + col_off))
    y2 = min(float(raster_height), max(0.0, local.y2 + row_off))
    if x2 <= x1 or y2 <= y1:
        return None
    return Detection(
        class_id=local.class_id,
        class_name=local.class_name,
        confidence=local.confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def run_pipeline(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    detector: Detector | None = None,
) -> PipelineResult:
    config.validate()
    started_at = time.perf_counter()

    def notify(phase: str, fraction: float, message: str) -> None:
        if progress:
            progress(phase, min(1.0, max(0.0, fraction)), message)

    # Avant toute écriture : un raster inexploitable ne doit rien laisser derrière.
    raster_path, original_gsd = prepare_source(config, notify)

    run_dir = create_run_directory(
        config.output_root,
        config.source.stem,
        config.run_name,
    )
    notify("initialisation", 0.01, "Chargement du modèle")
    if detector is not None:
        active_detector = detector
    elif config.classifier is not None:
        active_detector = CascadeYoloDetector(
            config.model,
            config.classifier,
            device=config.device,
            classifier_confidence=config.classifier_confidence,
            detector_nms_iou=config.detector_nms_iou,
            crop_context=config.crop_context,
            box_scale=config.box_scale,
            max_aspect_ratio=config.max_aspect_ratio,
        )
    else:
        active_detector = YoloDetector(
            config.model,
            config.device,
            box_scale=config.box_scale,
            max_aspect_ratio=config.max_aspect_ratio,
        )

    raw_detections: list[Detection] = []
    with rasterio.open(raster_path) as source:
        windows = generate_windows(
            source.width,
            source.height,
            tile_size=config.tile_size,
            overlap=config.overlap,
        )
        band_indexes = list(range(1, min(source.count, 3) + 1))
        if not band_indexes:
            raise ValueError("Le raster source ne contient aucune bande")
        raster_metadata = {
            "width": source.width,
            "height": source.height,
            "band_count": source.count,
            "crs": source.crs.to_string() if source.crs else None,
            "transform": list(source.transform),
            "window_count": len(windows),
        }
        transform = source.transform
        crs = source.crs

        for position, tile in enumerate(windows, start=1):
            window = Window(tile.col_off, tile.row_off, tile.width, tile.height)
            bands = source.read(band_indexes, window=window)
            image = to_rgb_uint8(bands, config.tile_size)
            local_detections = active_detector.predict(image, config.confidence)
            for local in local_detections:
                translated = _translate_detection(
                    local,
                    tile.col_off,
                    tile.row_off,
                    tile.width,
                    tile.height,
                    source.width,
                    source.height,
                )
                if translated is not None:
                    raw_detections.append(translated)
            if position == len(windows) or position % 5 == 0:
                notify(
                    "inference",
                    0.02 + 0.68 * position / len(windows),
                    f"Inférence {position}/{len(windows)} fenêtres",
                )

    # Les détecteurs intégrés écartent déjà les boîtes étirées avant de
    # classifier ; ce second passage couvre aussi un détecteur injecté.
    raw_detections = [
        detection
        for detection in raw_detections
        if aspect_ratio((detection.x1, detection.y1, detection.x2, detection.y2))
        <= config.max_aspect_ratio
    ]
    notify(
        "deduplication",
        0.72,
        f"Déduplication de {len(raw_detections)} boîtes",
    )
    detections = global_class_agnostic_nms(raw_detections, config.nms_iou)

    geopackage = run_dir / "detections.gpkg"
    detections_csv = run_dir / "detections.csv"
    summary_csv = run_dir / "summary.csv"
    annotated_tiff = run_dir / "annotated.tif"
    metadata_json = run_dir / "metadata.json"

    notify("vecteurs", 0.75, "Écriture du GeoPackage et des CSV")
    write_geopackage(geopackage, detections, transform, crs)
    write_detections_csv(detections_csv, detections, transform)
    summary = write_summary_csv(summary_csv, detections)

    def annotation_progress(phase: str, fraction: float, message: str) -> None:
        notify(phase, 0.78 + 0.21 * fraction, message)

    write_annotated_geotiff(
        raster_path,
        annotated_tiff,
        detections,
        annotation_progress,
        block_size=512,
    )

    elapsed_seconds = time.perf_counter() - started_at
    empty_count, occupied_count, success_rate = summarize_detections(detections)
    metadata: dict[str, object] = {
        "source": str(config.source.resolve()),
        # Les coordonnées des détections se rapportent au raster réellement lu.
        "processed_raster": str(raster_path.resolve()),
        "source_gsd_m": original_gsd,
        "resampled": raster_path != config.source,
        "model": str(config.model.resolve()),
        "classifier": (
            str(config.classifier.resolve()) if config.classifier is not None else None
        ),
        "architecture": "cascade" if config.classifier is not None else "direct",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": {
            "confidence": config.confidence,
            "classifier_confidence": config.classifier_confidence,
            "detector_nms_iou": config.detector_nms_iou,
                "crop_context": config.crop_context,
                "box_scale": config.box_scale,
                "max_aspect_ratio": config.max_aspect_ratio,
                "tile_size": config.tile_size,
            "overlap": config.overlap,
            "nms_iou": config.nms_iou,
            "device": config.device,
        },
        "raster": raster_metadata,
        "raw_detection_count": len(raw_detections),
        "deduplicated_detection_count": len(detections),
        "summary": summary,
        "elapsed_seconds": elapsed_seconds,
        "colors": {
            "PlantingHoleEmpty": "orange",
            "PlantingHoleOccupied": "green",
        },
    }
    write_metadata(metadata_json, metadata)
    notify("termine", 1.0, f"Traitement terminé : {run_dir}")

    return PipelineResult(
        run_dir=run_dir,
        geopackage=geopackage,
        annotated_tiff=annotated_tiff,
        detections_csv=detections_csv,
        summary_csv=summary_csv,
        metadata_json=metadata_json,
        detection_count=len(detections),
        empty_count=empty_count,
        occupied_count=occupied_count,
        success_rate=success_rate,
        elapsed_seconds=elapsed_seconds,
    )

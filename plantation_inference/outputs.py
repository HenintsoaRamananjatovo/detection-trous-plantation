from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

import fiona
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.windows import Window

from .geospatial import (
    detection_geojson_geometry,
    detection_map_center,
    detection_map_ring,
)
from .tiling import generate_windows, to_rgb_uint8
from .types import Detection, ProgressCallback


CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 165, 0),  # PlantingHoleEmpty: orange
    1: (0, 200, 0),  # PlantingHoleOccupied: green
}


def calculate_success_rate(empty_count: int, occupied_count: int) -> float | None:
    total = empty_count + occupied_count
    return occupied_count / total if total else None


def summarize_detections(
    detections: Sequence[Detection],
) -> tuple[int, int, float | None]:
    counts = Counter(d.class_id for d in detections)
    empty_count = counts[0]
    occupied_count = counts[1]
    return empty_count, occupied_count, calculate_success_rate(empty_count, occupied_count)


def _ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"La sortie existe déjà et ne sera pas écrasée : {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_detections_csv(
    path: Path,
    detections: Sequence[Detection],
    transform: rasterio.Affine,
) -> None:
    _ensure_new(path)
    fieldnames = [
        "id",
        "class_id",
        "class_name",
        "confidence",
        "pixel_x1",
        "pixel_y1",
        "pixel_x2",
        "pixel_y2",
        "map_center_x",
        "map_center_y",
        "geometry_wkt",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, detection in enumerate(detections, start=1):
            center_x, center_y = detection_map_center(detection, transform)
            ring = detection_map_ring(detection, transform)
            coordinates = ", ".join(f"{x:.8f} {y:.8f}" for x, y in ring)
            writer.writerow(
                {
                    "id": index,
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "confidence": f"{detection.confidence:.6f}",
                    "pixel_x1": f"{detection.x1:.3f}",
                    "pixel_y1": f"{detection.y1:.3f}",
                    "pixel_x2": f"{detection.x2:.3f}",
                    "pixel_y2": f"{detection.y2:.3f}",
                    "map_center_x": f"{center_x:.8f}",
                    "map_center_y": f"{center_y:.8f}",
                    "geometry_wkt": f"POLYGON (({coordinates}))",
                }
            )


def write_summary_csv(path: Path, detections: Sequence[Detection]) -> dict[str, object]:
    _ensure_new(path)
    counts = Counter(d.class_id for d in detections)
    names = {
        0: "PlantingHoleEmpty",
        1: "PlantingHoleOccupied",
        **{d.class_id: d.class_name for d in detections},
    }
    empty_count, occupied_count, success_rate = summarize_detections(detections)
    total = len(detections)
    fieldnames = [
        "class_id",
        "class_name",
        "count",
        "share",
        "total_detections",
        "estimated_success_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for class_id in sorted(names):
            count = counts[class_id]
            writer.writerow(
                {
                    "class_id": class_id,
                    "class_name": names[class_id],
                    "count": count,
                    "share": f"{count / total:.6f}" if total else "",
                    "total_detections": total,
                    "estimated_success_rate": (
                        f"{success_rate:.6f}" if success_rate is not None else ""
                    ),
                }
            )
        writer.writerow(
            {
                "class_id": "",
                "class_name": "TOTAL",
                "count": total,
                "share": "1.000000" if total else "",
                "total_detections": total,
                "estimated_success_rate": (
                    f"{success_rate:.6f}" if success_rate is not None else ""
                ),
            }
        )
    return {
        "total_detections": total,
        "empty_count": empty_count,
        "occupied_count": occupied_count,
        "estimated_success_rate": success_rate,
    }


def write_geopackage(
    path: Path,
    detections: Sequence[Detection],
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS | None,
) -> None:
    _ensure_new(path)
    schema = {
        "geometry": "Polygon",
        "properties": {
            "id": "int",
            "class_id": "int",
            "class_name": "str:64",
            "confidence": "float",
            "px_minx": "float",
            "px_miny": "float",
            "px_maxx": "float",
            "px_maxy": "float",
            "center_x": "float",
            "center_y": "float",
        },
    }
    open_kwargs: dict[str, object] = {
        "driver": "GPKG",
        "schema": schema,
        "layer": "planting_holes",
    }
    if crs is not None:
        open_kwargs["crs_wkt"] = crs.to_wkt()

    with fiona.open(path, mode="w", **open_kwargs) as collection:
        for index, detection in enumerate(detections, start=1):
            center_x, center_y = detection_map_center(detection, transform)
            collection.write(
                {
                    "geometry": detection_geojson_geometry(detection, transform),
                    "properties": {
                        "id": index,
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "px_minx": detection.x1,
                        "px_miny": detection.y1,
                        "px_maxx": detection.x2,
                        "px_maxy": detection.y2,
                        "center_x": center_x,
                        "center_y": center_y,
                    },
                }
            )


def _detection_block_index(
    detections: Sequence[Detection],
    block_size: int,
    raster_width: int,
    raster_height: int,
) -> dict[tuple[int, int], list[Detection]]:
    block_columns = math.ceil(raster_width / block_size)
    block_rows = math.ceil(raster_height / block_size)
    index: dict[tuple[int, int], list[Detection]] = defaultdict(list)
    for detection in detections:
        start_column = max(0, int(detection.x1) // block_size)
        end_column = min(
            block_columns - 1,
            int(max(detection.x1, detection.x2 - 1e-6)) // block_size,
        )
        start_row = max(0, int(detection.y1) // block_size)
        end_row = min(
            block_rows - 1,
            int(max(detection.y1, detection.y2 - 1e-6)) // block_size,
        )
        for block_row in range(start_row, end_row + 1):
            for block_column in range(start_column, end_column + 1):
                index[(block_row, block_column)].append(detection)
    return index


def write_annotated_geotiff(
    source_path: Path,
    output_path: Path,
    detections: Sequence[Detection],
    progress: ProgressCallback | None = None,
    block_size: int = 512,
) -> None:
    _ensure_new(output_path)
    with rasterio.open(source_path) as source:
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            count=3,
            dtype="uint8",
            nodata=None,
            tiled=True,
            blockxsize=block_size,
            blockysize=block_size,
            compress="deflate",
            predictor=2,
            photometric="RGB",
            BIGTIFF="IF_SAFER",
        )
        block_index = _detection_block_index(
            detections,
            block_size,
            source.width,
            source.height,
        )
        windows = generate_windows(
            source.width,
            source.height,
            tile_size=block_size,
            overlap=0,
        )
        band_indexes = list(range(1, min(source.count, 3) + 1))

        with rasterio.open(output_path, "w", **profile) as destination:
            for position, tile in enumerate(windows, start=1):
                window = Window(tile.col_off, tile.row_off, tile.width, tile.height)
                bands = source.read(band_indexes, window=window)
                rgb = to_rgb_uint8(bands)
                image = Image.fromarray(rgb)
                draw = ImageDraw.Draw(image)
                block_row = tile.row_off // block_size
                block_column = tile.col_off // block_size
                for detection in block_index.get((block_row, block_column), []):
                    color = CLASS_COLORS.get(detection.class_id, (255, 0, 0))
                    draw.rectangle(
                        (
                            detection.x1 - tile.col_off,
                            detection.y1 - tile.row_off,
                            detection.x2 - tile.col_off,
                            detection.y2 - tile.row_off,
                        ),
                        outline=color,
                        width=3,
                    )
                destination.write(np.moveaxis(np.asarray(image), -1, 0), window=window)
                if progress and (position == len(windows) or position % 10 == 0):
                    progress(
                        "annotation",
                        position / len(windows),
                        f"Écriture GeoTIFF {position}/{len(windows)}",
                    )

            factors = [
                factor
                for factor in (2, 4, 8, 16)
                if source.width // factor >= 1 and source.height // factor >= 1
            ]
            if factors:
                destination.build_overviews(factors, Resampling.average)
                destination.update_tags(ns="rio_overview", resampling="average")
            destination.update_tags(
                software="PlantationInference",
                empty_color="orange",
                occupied_color="green",
            )


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    _ensure_new(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

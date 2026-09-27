from __future__ import annotations

from affine import Affine

from .types import Detection


def pixel_to_map(transform: Affine, column: float, row: float) -> tuple[float, float]:
    x, y = transform @ (column, row)
    return float(x), float(y)


def detection_map_ring(
    detection: Detection,
    transform: Affine,
) -> list[tuple[float, float]]:
    """Return a closed map-coordinate polygon for a pixel bounding box."""
    corners = [
        pixel_to_map(transform, detection.x1, detection.y1),
        pixel_to_map(transform, detection.x2, detection.y1),
        pixel_to_map(transform, detection.x2, detection.y2),
        pixel_to_map(transform, detection.x1, detection.y2),
    ]
    return [*corners, corners[0]]


def detection_map_center(
    detection: Detection,
    transform: Affine,
) -> tuple[float, float]:
    center_x, center_y = detection.center
    return pixel_to_map(transform, center_x, center_y)


def detection_geojson_geometry(
    detection: Detection,
    transform: Affine,
) -> dict[str, object]:
    return {
        "type": "Polygon",
        "coordinates": [detection_map_ring(detection, transform)],
    }

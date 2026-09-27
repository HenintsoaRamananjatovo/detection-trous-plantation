from __future__ import annotations

import numpy as np
import pytest
from affine import Affine

from plantation_inference.detection import (
    aspect_ratio,
    global_class_agnostic_nms,
    rgb_to_ultralytics_source,
    scale_box,
)
from plantation_inference.geospatial import detection_map_center, detection_map_ring
from plantation_inference.outputs import calculate_success_rate
from plantation_inference.pipeline import TRAINING_GSD_M, check_resolution
from plantation_inference.tiling import generate_windows, to_rgb_uint8
from plantation_inference.types import Detection


def make_detection(
    class_id: int,
    confidence: float,
    box: tuple[float, float, float, float],
) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=("PlantingHoleEmpty" if class_id == 0 else "PlantingHoleOccupied"),
        confidence=confidence,
        x1=box[0],
        y1=box[1],
        x2=box[2],
        y2=box[3],
    )


def test_windows_cover_raster_and_edges_are_partial() -> None:
    windows = generate_windows(width=1000, height=750, tile_size=512, overlap=64)
    coverage = np.zeros((750, 1000), dtype=np.uint8)
    for window in windows:
        coverage[
            window.row_off : window.row_off + window.height,
            window.col_off : window.col_off + window.width,
        ] = 1

    assert coverage.all()
    assert windows[-1].width == 104
    assert windows[-1].height == 302
    assert len(windows) == 6


@pytest.mark.parametrize(
    ("tile_size", "overlap"),
    [(0, 0), (512, -1), (512, 512), (512, 600)],
)
def test_windows_reject_invalid_parameters(tile_size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        generate_windows(100, 100, tile_size, overlap)


def test_rgb_conversion_and_border_padding() -> None:
    gray = np.full((1, 20, 30), 128, dtype=np.uint8)
    rgb = to_rgb_uint8(gray, tile_size=64)

    assert rgb.shape == (64, 64, 3)
    assert np.all(rgb[:20, :30] == 128)
    assert np.all(rgb[20:] == 0)
    assert np.all(rgb[:, 30:] == 0)


def test_rgb_is_converted_to_bgr_for_ultralytics_numpy_input() -> None:
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    bgr = rgb_to_ultralytics_source(rgb)

    assert bgr.tolist() == [[[30, 20, 10], [60, 50, 40]]]
    assert bgr.flags.c_contiguous


def test_scale_box_preserves_centre_and_undoes_rotation_inflation() -> None:
    inflated = scale_box((100.0, 200.0, 150.0, 250.0), 1.36)
    assert scale_box(inflated, 1 / 1.36) == pytest.approx(
        (100.0, 200.0, 150.0, 250.0)
    )
    assert ((inflated[0] + inflated[2]) / 2) == pytest.approx(125.0)
    assert ((inflated[1] + inflated[3]) / 2) == pytest.approx(225.0)
    assert (inflated[2] - inflated[0]) == pytest.approx(68.0)


def test_scale_box_is_a_noop_at_unit_factor() -> None:
    box = (10.0, 20.0, 30.0, 40.0)
    assert scale_box(box, 1.0) is box


def test_aspect_ratio_is_orientation_independent_and_never_below_one() -> None:
    assert aspect_ratio((0.0, 0.0, 30.0, 30.0)) == pytest.approx(1.0)
    assert aspect_ratio((0.0, 0.0, 60.0, 20.0)) == pytest.approx(3.0)
    assert aspect_ratio((0.0, 0.0, 20.0, 60.0)) == pytest.approx(3.0)


def test_aspect_ratio_rejects_degenerate_boxes() -> None:
    assert aspect_ratio((5.0, 5.0, 5.0, 40.0)) == float("inf")


def test_resolution_check_accepts_the_training_scale_and_nearby() -> None:
    check_resolution(TRAINING_GSD_M)
    check_resolution(TRAINING_GSD_M * 1.15)
    check_resolution(TRAINING_GSD_M * 0.85)


@pytest.mark.parametrize("gsd", [0.0085, 0.10, TRAINING_GSD_M * 1.5])
def test_resolution_check_rejects_other_scales(gsd: float) -> None:
    with pytest.raises(ValueError, match="Résolution incompatible"):
        check_resolution(gsd)


def test_pixel_to_map_for_bbox_and_center() -> None:
    transform = Affine(2, 0, 100, 0, -3, 500)
    detection = make_detection(0, 0.9, (10, 20, 30, 40))

    assert detection_map_center(detection, transform) == (140.0, 410.0)
    assert detection_map_ring(detection, transform) == [
        (120.0, 440.0),
        (160.0, 440.0),
        (160.0, 380.0),
        (120.0, 380.0),
        (120.0, 440.0),
    ]


def test_global_nms_is_class_agnostic_and_keeps_best_box() -> None:
    lower_empty = make_detection(0, 0.7, (10, 10, 30, 30))
    higher_occupied = make_detection(1, 0.95, (11, 11, 31, 31))
    separate = make_detection(0, 0.8, (100, 100, 120, 120))

    kept = global_class_agnostic_nms(
        [lower_empty, higher_occupied, separate],
        iou_threshold=0.45,
    )

    assert kept == [higher_occupied, separate]


def test_success_rate() -> None:
    assert calculate_success_rate(25, 75) == pytest.approx(0.75)
    assert calculate_success_rate(0, 4) == pytest.approx(1.0)
    assert calculate_success_rate(0, 0) is None

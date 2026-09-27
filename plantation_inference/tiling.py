from __future__ import annotations

import math

import numpy as np

from .types import TileWindow


def generate_windows(
    width: int,
    height: int,
    tile_size: int = 512,
    overlap: int = 64,
) -> list[TileWindow]:
    """Cover a raster without gaps; edge windows may be smaller."""
    if width <= 0 or height <= 0:
        raise ValueError("Les dimensions du raster doivent être positives")
    if tile_size <= 0:
        raise ValueError("tile_size doit être positif")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap doit être positif et inférieur à tile_size")

    step = tile_size - overlap
    windows: list[TileWindow] = []
    for row_off in range(0, height, step):
        for col_off in range(0, width, step):
            windows.append(
                TileWindow(
                    col_off=col_off,
                    row_off=row_off,
                    width=min(tile_size, width - col_off),
                    height=min(tile_size, height - row_off),
                )
            )
    return windows


def window_count(width: int, height: int, tile_size: int, overlap: int) -> int:
    step = tile_size - overlap
    return math.ceil(width / step) * math.ceil(height / step)


def to_rgb_uint8(bands: np.ndarray, tile_size: int | None = None) -> np.ndarray:
    """Convert band-first raster data to padded RGB uint8."""
    if bands.ndim != 3 or bands.shape[0] == 0:
        raise ValueError("Le tableau attendu doit avoir la forme (bandes, hauteur, largeur)")

    selected = bands[:3]
    if selected.shape[0] == 1:
        selected = np.repeat(selected, 3, axis=0)
    elif selected.shape[0] == 2:
        selected = np.concatenate([selected, selected[1:2]], axis=0)

    if selected.dtype == np.uint8:
        converted = selected
    elif np.issubdtype(selected.dtype, np.integer):
        info = np.iinfo(selected.dtype)
        converted = np.clip(
            (selected.astype(np.float32) - info.min) * (255.0 / (info.max - info.min)),
            0,
            255,
        ).astype(np.uint8)
    else:
        data = np.nan_to_num(selected.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        finite_min = float(data.min())
        finite_max = float(data.max())
        if finite_min >= 0 and finite_max <= 1:
            data *= 255.0
        elif finite_max > finite_min:
            data = (data - finite_min) * (255.0 / (finite_max - finite_min))
        converted = np.clip(data, 0, 255).astype(np.uint8)

    rgb = np.moveaxis(converted, 0, -1)
    if tile_size is None:
        return np.ascontiguousarray(rgb)
    if tile_size < rgb.shape[0] or tile_size < rgb.shape[1]:
        raise ValueError("tile_size ne peut pas être inférieur à la fenêtre")
    padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    padded[: rgb.shape[0], : rgb.shape[1]] = rgb
    return padded

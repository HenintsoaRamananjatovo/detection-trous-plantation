"""Mise d'un raster à la résolution attendue par le modèle."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject


def cache_name(source: Path, resolution: float) -> str:
    """Nom stable et lisible, la résolution étant exprimée en dixièmes de millimètre."""
    return f"{source.stem}_gsd{round(resolution * 10_000)}.tif"


def resample(
    source_path: Path,
    destination_path: Path,
    resolution: float,
    progress: Callable[[float], None] | None = None,
) -> None:
    """Écrit une copie du raster à la résolution demandée, en mètres par pixel."""
    if resolution <= 0:
        raise ValueError("La résolution cible doit être strictement positive")
    temporary = destination_path.with_suffix(".partiel.tif")
    with rasterio.open(source_path) as source:
        if source.crs is None:
            raise ValueError(
                f"{source_path.name} n'a pas de système de coordonnées, "
                "son échelle ne peut pas être corrigée"
            )
        width = math.ceil(source.width * abs(source.res[0]) / resolution)
        height = math.ceil(source.height * abs(source.res[1]) / resolution)
        if width < 1 or height < 1:
            raise ValueError("La résolution demandée réduirait le raster à néant")
        transform = source.transform * Affine.scale(
            source.width / width,
            source.height / height,
        )
        profile = source.profile.copy()
        profile.update(
            driver="GTiff",
            width=width,
            height=height,
            transform=transform,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="deflate",
            predictor=2,
            BIGTIFF="IF_SAFER",
        )
        try:
            with rasterio.open(temporary, "w", **profile) as destination:
                for index in range(1, source.count + 1):
                    band = np.zeros((height, width), dtype=source.dtypes[index - 1])
                    reproject(
                        source=rasterio.band(source, index),
                        destination=band,
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=transform,
                        dst_crs=source.crs,
                        dst_nodata=source.nodata,
                        # Un canal alpha ne se moyenne pas : il vaut tout ou rien.
                        resampling=(
                            Resampling.nearest if index == 4 else Resampling.bilinear
                        ),
                        num_threads=4,
                    )
                    destination.write(band, index)
                    if progress is not None:
                        progress(index / source.count)
                factors = [
                    factor for factor in (2, 4, 8, 16) if min(width, height) // factor
                ]
                if factors:
                    destination.build_overviews(factors, Resampling.average)
        except BaseException:
            # Un fichier tronqué serait pris pour un cache valide au prochain appel.
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(destination_path)


def prepared_copy(
    source_path: Path,
    cache_directory: Path,
    resolution: float,
    progress: Callable[[float], None] | None = None,
) -> Path:
    """Version du raster à la bonne résolution, fabriquée seulement si nécessaire."""
    cache_directory.mkdir(parents=True, exist_ok=True)
    target = cache_directory / cache_name(source_path, resolution)
    # Une copie antérieure à la source proviendrait d'un fichier remplacé depuis.
    if target.is_file() and target.stat().st_mtime >= source_path.stat().st_mtime:
        return target
    resample(source_path, target, resolution, progress)
    return target

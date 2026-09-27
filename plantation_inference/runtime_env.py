from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def configure_geospatial_data() -> None:
    """Avoid incompatible global PostGIS GDAL/PROJ paths on Windows."""
    spec = importlib.util.find_spec("rasterio")
    if spec is None or spec.origin is None:
        return
    rasterio_directory = Path(spec.origin).parent
    proj_directory = rasterio_directory / "proj_data"
    gdal_directory = rasterio_directory / "gdal_data"
    if (proj_directory / "proj.db").is_file():
        os.environ["PROJ_LIB"] = str(proj_directory)
        os.environ["PROJ_DATA"] = str(proj_directory)
    if gdal_directory.is_dir():
        os.environ["GDAL_DATA"] = str(gdal_directory)

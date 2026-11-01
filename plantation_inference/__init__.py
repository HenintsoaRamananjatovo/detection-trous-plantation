"""Windowed geospatial inference for plantation monitoring."""

from .runtime_env import configure_geospatial_data

configure_geospatial_data()

from .pipeline import run_pipeline
from .types import (
    DEFAULT_BOX_SCALE,
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_MODEL,
    DEFAULT_NMS_CENTER_DISTANCE_PX,
    Detection,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    "DEFAULT_BOX_SCALE",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MAX_ASPECT_RATIO",
    "DEFAULT_MODEL",
    "DEFAULT_NMS_CENTER_DISTANCE_PX",
    "Detection",
    "PipelineConfig",
    "PipelineResult",
    "run_pipeline",
]

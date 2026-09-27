"""Windowed geospatial inference for plantation monitoring."""

from .runtime_env import configure_geospatial_data

configure_geospatial_data()

from .pipeline import run_pipeline
from .types import Detection, PipelineConfig, PipelineResult

__all__ = ["Detection", "PipelineConfig", "PipelineResult", "run_pipeline"]

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import fiona
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app import app
from plantation_inference.pipeline import TRAINING_GSD_M, run_pipeline
from plantation_inference.types import Detection, PipelineConfig


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: np.ndarray, confidence: float) -> list[Detection]:
        assert image.shape == (512, 512, 3)
        self.calls += 1
        if self.calls == 1:
            return [
                Detection(
                    1,
                    "PlantingHoleOccupied",
                    0.95,
                    430,
                    100,
                    480,
                    150,
                )
            ]
        return [
            Detection(
                0,
                "PlantingHoleEmpty",
                0.70,
                0,
                100,
                32,
                150,
            )
        ]


def create_source(path: Path, gsd: float = 0.04) -> None:
    rows, columns = np.indices((300, 700))
    data = np.stack(
        [
            columns % 256,
            rows % 256,
            (rows + columns) % 256,
            np.full_like(rows, 255),
        ]
    ).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=700,
        height=300,
        count=4,
        dtype="uint8",
        crs="EPSG:32739",
        transform=from_origin(500000, 9000000, gsd, gsd),
    ) as destination:
        destination.write(data)


def test_complete_pipeline_outputs(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    create_source(source)
    config = PipelineConfig(
        source=source,
        model=model,
        output_root=tmp_path / "outputs",
        confidence=0.25,
        tile_size=512,
        overlap=64,
        run_name="integration",
    )

    result = run_pipeline(config, detector=FakeDetector())

    assert result.detection_count == 1
    assert result.occupied_count == 1
    assert result.empty_count == 0
    assert result.success_rate == 1.0
    assert all(
        path.is_file()
        for path in (
            result.geopackage,
            result.annotated_tiff,
            result.detections_csv,
            result.summary_csv,
            result.metadata_json,
        )
    )

    with fiona.open(result.geopackage, layer="planting_holes") as features:
        assert len(features) == 1
        assert features.crs.to_epsg() == 32739
        feature = next(iter(features))
        assert feature["properties"]["class_name"] == "PlantingHoleOccupied"

    with rasterio.open(result.annotated_tiff) as annotated:
        assert (annotated.width, annotated.height, annotated.count) == (700, 300, 3)
        assert annotated.crs.to_epsg() == 32739
        assert annotated.overviews(1) == [2, 4, 8, 16]

    with result.detections_csv.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 1

    second = run_pipeline(config, detector=FakeDetector())
    assert second.run_dir != result.run_dir
    assert second.run_dir.name == "integration_2"


class StretchedBoxDetector:
    """Renvoie un trou plausible et une boîte étirée typique d'un faux positif."""

    def predict(self, image: np.ndarray, confidence: float) -> list[Detection]:
        return [
            Detection(1, "PlantingHoleOccupied", 0.95, 100, 100, 150, 150),
            Detection(0, "PlantingHoleEmpty", 0.90, 200, 100, 340, 130),
        ]


def test_stretched_boxes_are_rejected_unless_the_limit_is_lifted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tif"
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    create_source(source)
    common = {
        "source": source,
        "model": model,
        "output_root": tmp_path / "outputs",
        "confidence": 0.25,
        "tile_size": 1024,  # une seule fenêtre couvre le raster de test
        "overlap": 0,
    }

    filtered = run_pipeline(
        PipelineConfig(**common, max_aspect_ratio=2.0, run_name="filtre"),
        detector=StretchedBoxDetector(),
    )
    assert filtered.detection_count == 1
    assert filtered.occupied_count == 1

    kept = run_pipeline(
        PipelineConfig(**common, max_aspect_ratio=99.0, run_name="sans_filtre"),
        detector=StretchedBoxDetector(),
    )
    assert kept.detection_count == 2


class AnySizeDetector:
    """Détecteur minimal, indifférent à la taille de la fenêtre reçue."""

    def predict(self, image: np.ndarray, confidence: float) -> list[Detection]:
        return [Detection(1, "PlantingHoleOccupied", 0.95, 10, 10, 40, 40)]


def test_a_finer_raster_is_resampled_to_the_training_scale(tmp_path: Path) -> None:
    source = tmp_path / "trop_fin.tif"
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    create_source(source, gsd=0.0085)
    common = {
        "source": source,
        "model": model,
        "output_root": tmp_path / "outputs",
        "tile_size": 1024,
        "overlap": 0,
    }

    result = run_pipeline(
        PipelineConfig(**common, run_name="auto"), detector=AnySizeDetector()
    )

    metadata = json.loads(result.metadata_json.read_text(encoding="utf-8"))
    assert metadata["resampled"] is True
    assert metadata["source"] == str(source.resolve())
    prepared = Path(metadata["processed_raster"])
    assert prepared.parent == tmp_path / "outputs" / "_prepared"
    with rasterio.open(prepared) as raster:
        assert abs(raster.transform.a) == pytest.approx(TRAINING_GSD_M, rel=1e-3)
        # 700 pixels de 0.85 cm couvrent 5.95 m, soit 153 pixels de 3.89 cm.
        assert raster.width == 153

    # Le second passage doit réutiliser la copie au lieu de la refabriquer.
    horodatage = prepared.stat().st_mtime_ns
    run_pipeline(PipelineConfig(**common, run_name="reprise"), detector=AnySizeDetector())
    assert prepared.stat().st_mtime_ns == horodatage


def test_resampling_can_be_refused_or_bypassed(tmp_path: Path) -> None:
    source = tmp_path / "trop_fin.tif"
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    create_source(source, gsd=0.0085)
    common = {
        "source": source,
        "model": model,
        "output_root": tmp_path / "outputs",
        "tile_size": 1024,
        "overlap": 0,
    }

    with pytest.raises(ValueError, match="Résolution incompatible"):
        run_pipeline(
            PipelineConfig(**common, auto_resample=False), detector=FakeDetector()
        )
    assert not (tmp_path / "outputs").exists()

    forced = run_pipeline(
        PipelineConfig(**common, ignore_resolution=True, run_name="force"),
        detector=StretchedBoxDetector(),
    )
    assert forced.detection_count == 1
    metadata = json.loads(forced.metadata_json.read_text(encoding="utf-8"))
    assert metadata["resampled"] is False


def test_a_raster_too_coarse_to_rescue_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "trop_grossier.tif"
    model = tmp_path / "model.pt"
    model.write_bytes(b"fake")
    create_source(source, gsd=0.30)

    with pytest.raises(ValueError, match="trop grossière"):
        run_pipeline(
            PipelineConfig(
                source=source,
                model=model,
                output_root=tmp_path / "outputs",
                tile_size=1024,
            ),
            detector=FakeDetector(),
        )
    assert not (tmp_path / "outputs").exists()


@pytest.fixture
def client(tmp_path: Path):
    """Client web dont les téléversements atterrissent hors du dossier du projet."""
    app.config["UPLOAD_DIRECTORY"] = tmp_path / "uploads"
    return app.test_client()


def uploads(tmp_path: Path) -> list[Path]:
    directory = tmp_path / "uploads"
    return sorted(directory.iterdir()) if directory.is_dir() else []


def test_the_page_is_a_bare_upload_form(client) -> None:
    page = client.get("/")
    assert page.status_code == 200
    body = page.data.decode("utf-8")
    assert 'type="file"' in body
    assert "GeoTIFF à analyser" in body
    # Les réglages ne sont plus exposés : ils dérouteraient sans servir.
    for absent in ("Options avancées", "box_scale", "tile_size", "source_override"):
        assert absent not in body


def test_a_request_without_any_source_is_refused(client) -> None:
    response = client.post("/jobs", data={})
    assert response.status_code == 400
    assert "Choisissez un GeoTIFF" in response.get_json()["error"]


def test_an_upload_that_is_not_a_raster_leaves_nothing_behind(
    client, tmp_path: Path
) -> None:
    response = client.post(
        "/jobs",
        data={"upload": (io.BytesIO(b"pas du tout un raster"), "faux.tif")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "n'est pas un GeoTIFF lisible" in response.get_json()["error"]
    assert uploads(tmp_path) == []


def test_an_upload_with_another_extension_is_refused(client, tmp_path: Path) -> None:
    response = client.post(
        "/jobs",
        data={"upload": (io.BytesIO(b"\x89PNG"), "photo.png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert ".tif" in response.get_json()["error"]
    assert uploads(tmp_path) == []


def test_a_local_path_takes_precedence_over_an_upload(client, tmp_path: Path) -> None:
    response = client.post(
        "/jobs",
        data={
            "source_override": "  Z:/typed.tif  ",
            "upload": (io.BytesIO(b"contenu ignore"), "ignore.tif"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "typed.tif" in response.get_json()["error"]
    # Rien ne doit être copié quand l'utilisateur désigne un fichier local.
    assert uploads(tmp_path) == []


def test_an_uploaded_raster_too_coarse_is_refused_right_away(
    client, tmp_path: Path
) -> None:
    original = tmp_path / "grossier.tif"
    create_source(original, gsd=0.30)

    response = client.post(
        "/jobs",
        data={"upload": (io.BytesIO(original.read_bytes()), "grossier.tif")},
        content_type="multipart/form-data",
    )

    # Refusé sans lancer de traitement : aucun rééchantillonnage ne le sauverait.
    assert response.status_code == 400
    assert "trop grossière" in response.get_json()["error"]
    assert [path.name for path in uploads(tmp_path)] == ["grossier.tif"]

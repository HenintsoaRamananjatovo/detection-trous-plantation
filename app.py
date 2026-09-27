from __future__ import annotations

from pathlib import Path

import rasterio
from flask import Flask, abort, jsonify, render_template, request, send_from_directory
from rasterio.errors import RasterioIOError
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

from cli import (
    DEFAULT_BOX_SCALE,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_MODEL,
    DEFAULT_OUTPUTS,
)
from plantation_inference.job_manager import JobManager
from plantation_inference.pipeline import check_usable_resolution
from plantation_inference.types import PipelineConfig


PROJECT_ROOT = Path(__file__).resolve().parent
# Werkzeug écrit déjà le corps de la requête sur disque au-delà de quelques
# centaines de kilooctets : cette limite protège l'espace disque, pas la mémoire.
MAX_UPLOAD_BYTES = 8 * 1024**3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.config["UPLOAD_DIRECTORY"] = PROJECT_ROOT / "uploads"
jobs = JobManager()


def store_upload(upload: FileStorage) -> Path:
    """Écrit le raster reçu sous un nom libre et renvoie son chemin."""
    name = secure_filename(upload.filename or "")
    if not name.lower().endswith((".tif", ".tiff")):
        raise ValueError("Seuls les fichiers .tif et .tiff sont acceptés")
    directory = Path(app.config["UPLOAD_DIRECTORY"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    index = 2
    while target.exists():
        target = directory / f"{Path(name).stem}_{index}{Path(name).suffix}"
        index += 1
    upload.save(target)
    return target


def read_resolution(path: Path) -> float:
    """Taille du pixel en mètres, ou une erreur lisible si le raster est invalide."""
    try:
        with rasterio.open(path) as raster:
            if raster.count == 0:
                raise ValueError(f"{path.name} ne contient aucune bande")
            return abs(raster.transform.a)
    except RasterioIOError as error:
        raise ValueError(f"{path.name} n'est pas un GeoTIFF lisible : {error}") from error


def resolve_source(form, files) -> tuple[Path, Path | None]:
    """Chemin du raster à traiter, et sa copie téléversée s'il y en a une.

    `source_override` n'apparaît pas dans le formulaire, volontairement dépouillé.
    Il reste accepté pour traiter un gros raster local sans le copier, ce que
    fait aussi cli.py.
    """
    override = form.get("source_override", "").strip()
    if override:
        return Path(override).expanduser().resolve(), None
    upload = files.get("upload")
    if upload is not None and upload.filename:
        stored = store_upload(upload)
        return stored, stored
    raise ValueError("Choisissez un GeoTIFF à analyser")


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(_error):
    return jsonify(
        {
            "error": (
                f"Fichier trop volumineux, limite {MAX_UPLOAD_BYTES / 1024**3:.0f} Go. "
                "Traitez-le en ligne de commande avec cli.py, qui lit le raster "
                "sur place sans le copier."
            )
        }
    ), 413


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/jobs")
def create_job():
    if jobs.has_active_job():
        return jsonify({"error": "Un traitement est déjà en cours."}), 409

    stored: Path | None = None
    try:
        source, stored = resolve_source(request.form, request.files)
        gsd = read_resolution(source)
    except ValueError as error:
        # Un fichier inexploitable n'a aucune raison d'occuper le disque.
        if stored is not None:
            stored.unlink(missing_ok=True)
        return jsonify({"error": str(error)}), 400

    try:
        # Une échelle différente est corrigée par le pipeline ; seul un raster
        # trop grossier est sans espoir, et autant le dire tout de suite.
        if not request.form.get("ignore_resolution"):
            check_usable_resolution(gsd)
        config = PipelineConfig(
            source=source,
            model=Path(request.form.get("model", str(DEFAULT_MODEL)))
            .expanduser()
            .resolve(),
            output_root=Path(request.form.get("output_dir", str(DEFAULT_OUTPUTS)))
            .expanduser()
            .resolve(),
            classifier=(
                Path(request.form["classifier"]).expanduser().resolve()
                if request.form.get("classifier", "").strip()
                else None
            ),
            confidence=float(request.form.get("confidence", 0.20)),
            classifier_confidence=float(
                request.form.get("classifier_confidence", 0.0)
            ),
            detector_nms_iou=float(request.form.get("detector_nms_iou", 0.70)),
            crop_context=float(request.form.get("crop_context", 1.8)),
            box_scale=float(request.form.get("box_scale", DEFAULT_BOX_SCALE)),
            max_aspect_ratio=float(
                request.form.get("max_aspect_ratio", DEFAULT_MAX_ASPECT_RATIO)
            ),
            ignore_resolution=bool(request.form.get("ignore_resolution")),
            tile_size=int(request.form.get("tile_size", 512)),
            overlap=int(request.form.get("overlap", 0)),
            nms_iou=float(request.form.get("nms_iou", 0.45)),
            device=request.form.get("device", "cpu"),
            run_name=request.form.get("run_name") or None,
        )
        config.validate()
    except (KeyError, ValueError, FileNotFoundError) as error:
        return jsonify({"error": str(error)}), 400

    job = jobs.create(config)
    return jsonify(job.public_dict()), 202


@app.get("/jobs/<job_id>")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        abort(404)
    payload = job.public_dict()
    if job.result:
        payload["files"] = {
            key: f"/jobs/{job_id}/files/{Path(value).name}"
            for key, value in job.result.items()
            if key
            in {
                "geopackage",
                "annotated_tiff",
                "detections_csv",
                "summary_csv",
                "metadata_json",
            }
        }
    return jsonify(payload)


@app.get("/jobs/<job_id>/files/<filename>")
def job_file(job_id: str, filename: str):
    job = jobs.get(job_id)
    if job is None or not job.result:
        abort(404)
    allowed = {
        Path(job.result[key]).name: Path(job.result[key])
        for key in (
            "geopackage",
            "annotated_tiff",
            "detections_csv",
            "summary_csv",
            "metadata_json",
        )
    }
    target = allowed.get(filename)
    if target is None or not target.is_file():
        abort(404)
    return send_from_directory(target.parent, target.name, as_attachment=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

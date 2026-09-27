from __future__ import annotations

import argparse
import json
from pathlib import Path

# Les réglages calibrés appartiennent au paquet : les redéfinir ici ferait deux
# sources pour une même valeur mesurée, et un autre appelant en obtiendrait une
# troisième. Ils sont réexportés parce que app.py les importe de ce module.
from plantation_inference import (
    DEFAULT_BOX_SCALE,
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_MODEL,
    PipelineConfig,
    run_pipeline,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = PROJECT_ROOT / "outputs"
# Le détecteur prédit déjà vide ou occupé. Le classificateur en cascade dégrade
# ce classement sur les deux rasters annotés, de 0.935 à 0.908 sur source2 et de
# 0.887 à 0.827 sur result00389, tout en doublant la durée. Il reste accessible
# par --classifier pour comparaison. Il ne voyage pas avec le paquet : cette
# architecture est abandonnée, ces poids ne servent plus qu'à refaire la mesure.
OPTIONAL_CLASSIFIER = PROJECT_ROOT / "models" / "occupancy_classifier_best.pt"
DEFAULT_CLASSIFIER: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Détecte les trous de plantation vides/occupés dans un GeoTIFF, "
            "sans créer de tuiles intermédiaires."
        )
    )
    parser.add_argument("source", type=Path, help="GeoTIFF à analyser")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Modèle direct, ou détecteur/proposeur si --classifier est fourni",
    )
    parser.add_argument(
        "--classifier",
        type=Path,
        default=DEFAULT_CLASSIFIER,
        help=(
            "Active la cascade avec un classificateur d'occupation. Déconseillé : "
            f"il dégrade le classement. Poids disponibles : {OPTIONAL_CLASSIFIER}"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--classifier-confidence", type=float, default=0.0)
    parser.add_argument("--detector-nms-iou", type=float, default=0.70)
    parser.add_argument("--crop-context", type=float, default=1.8)
    parser.add_argument(
        "--box-scale",
        type=float,
        default=DEFAULT_BOX_SCALE,
        help="Recalibrage de la taille des boîtes prédites (1.0 = aucun)",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=DEFAULT_MAX_ASPECT_RATIO,
        help=(
            "Rejette les boîtes plus allongées que ce rapport, presque toutes "
            "des faux positifs sur le sol nu entre les rangées"
        ),
    )
    parser.add_argument(
        "--ignore-resolution",
        action="store_true",
        help="Traiter malgré une échelle différente de celle de l'entraînement",
    )
    parser.add_argument(
        "--no-resample",
        dest="auto_resample",
        action="store_false",
        help=(
            "Refuser un raster à une autre échelle au lieu de le rééchantillonner "
            "automatiquement à 3.89 cm/pixel"
        ),
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        help="Où garder les rasters rééchantillonnés (défaut : <sorties>/_prepared)",
    )
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--nms-iou", type=float, default=0.45)
    parser.add_argument(
        "--device",
        default="cpu",
        help='Périphérique Ultralytics : "cpu", "0" pour le premier GPU, ou "auto"',
    )
    parser.add_argument("--run-name", help="Nom de dossier souhaité (jamais écrasé)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = PipelineConfig(
        source=args.source.expanduser().resolve(),
        model=args.model.expanduser().resolve(),
        output_root=args.output_dir.expanduser().resolve(),
        classifier=(
            args.classifier.expanduser().resolve() if args.classifier else None
        ),
        confidence=args.confidence,
        classifier_confidence=args.classifier_confidence,
        detector_nms_iou=args.detector_nms_iou,
        crop_context=args.crop_context,
        box_scale=args.box_scale,
        max_aspect_ratio=args.max_aspect_ratio,
        ignore_resolution=args.ignore_resolution,
        auto_resample=args.auto_resample,
        prepared_root=(
            args.prepared_dir.expanduser().resolve() if args.prepared_dir else None
        ),
        tile_size=args.tile_size,
        overlap=args.overlap,
        nms_iou=args.nms_iou,
        device=args.device,
        run_name=args.run_name,
    )

    last_message = ""

    def show_progress(phase: str, fraction: float, message: str) -> None:
        nonlocal last_message
        rendered = f"[{fraction * 100:6.2f}%] {phase}: {message}"
        if rendered != last_message:
            print(rendered, flush=True)
            last_message = rendered

    try:
        result = run_pipeline(config, show_progress)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Erreur : {error}")
        return 1

    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

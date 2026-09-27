"""Le paquet doit livrer les poids retenus et leurs réglages calibrés.

Un appelant qui construit une configuration sans rien préciser doit obtenir la
configuration mesurée, pas une configuration neutre : box_scale à 1.0 fait
tomber le rappel de 94 % à 61 % sur le lot de test verrouillé, sans qu'aucune
erreur ne le signale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from plantation_inference import (
    DEFAULT_BOX_SCALE,
    DEFAULT_CONFIDENCE,
    DEFAULT_MAX_ASPECT_RATIO,
    DEFAULT_MODEL,
    PipelineConfig,
)


# Poids validés le 22 septembre 2026 : 95 % de rappel, 85 % de précision et
# 84 % de justesse de bout en bout sur les 8 tuiles de test de result_00389m.
EXPECTED_WEIGHTS_SHA256 = (
    "6c296b3eafc1b9b4982bb93c59a1133eb865826c4bf4bfa5c57a3cd1a4784305"
)

RECALIBRATION_REMINDER = (
    "Les poids livrés ne sont plus ceux qui ont été mesurés. Après un "
    "réentraînement, recalibrer box_scale sur la validation — ratio médian "
    "entre taille prédite et taille annotée, dont on prend l'inverse — puis "
    "mettre à jour cette empreinte et la version du paquet."
)


def test_the_retained_weights_travel_inside_the_package() -> None:
    assert DEFAULT_MODEL.is_file()
    assert DEFAULT_MODEL.parent.parent.name == "plantation_inference"


def test_the_shipped_weights_are_the_measured_ones() -> None:
    digest = hashlib.sha256(DEFAULT_MODEL.read_bytes()).hexdigest()
    assert digest == EXPECTED_WEIGHTS_SHA256, RECALIBRATION_REMINDER


def test_a_configuration_without_options_is_the_calibrated_one(tmp_path: Path) -> None:
    config = PipelineConfig(source=tmp_path / "parcelle.tif", output_root=tmp_path)

    assert config.model == DEFAULT_MODEL
    assert config.box_scale == DEFAULT_BOX_SCALE == 0.82
    assert config.max_aspect_ratio == DEFAULT_MAX_ASPECT_RATIO == 2.0
    assert config.confidence == DEFAULT_CONFIDENCE == 0.20

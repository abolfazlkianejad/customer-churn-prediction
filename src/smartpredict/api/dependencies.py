"""Lazy loading and injection of saved model artifacts for the API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import DatasetSpec
from smartpredict.config.settings import get_settings
from smartpredict.core.exceptions import ArtifactError
from smartpredict.core.logging import get_logger
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.metadata import read_decision_threshold

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelArtifacts:
    """All saved objects required to serve predictions."""

    spec: DatasetSpec
    preprocessor: ColumnTransformer
    model: ClassifierMixin
    metadata: dict[str, Any]
    feature_names: tuple[str, ...]
    selected_model: str
    threshold: float = 0.5


@lru_cache(maxsize=4)
def _cached_artifacts(models_dir: str) -> ModelArtifacts:
    """Load and cache artifacts for one model directory."""
    return load_model_artifacts(models_dir=Path(models_dir))


def _validate_metadata_contract(
    metadata: Mapping[str, Any],
    feature_names: tuple[str, ...],
    model: ClassifierMixin,
    preprocessor: ColumnTransformer,
) -> str:
    """Verify that metadata, transformer, and estimator agree.

    Args:
        metadata: Deserialized model metadata.
        feature_names: Feature names produced by the saved preprocessor.
        model: Saved estimator.
        preprocessor: Saved transformer.

    Returns:
        The selected model name from metadata.

    Raises:
        ArtifactError: If the bundle contract is incomplete or inconsistent.

    """
    model_section = metadata.get("model")
    if not isinstance(model_section, Mapping):
        msg = "Saved metadata is missing the 'model' section."
        raise ArtifactError(msg)
    selected_model = model_section.get("selected")
    if not isinstance(selected_model, str) or not selected_model.strip():
        msg = "Saved metadata is missing a valid selected model name."
        raise ArtifactError(msg)

    features_section = metadata.get("features")
    if not isinstance(features_section, Mapping):
        msg = "Saved metadata is missing the 'features' contract."
        raise ArtifactError(msg)

    saved_names = features_section.get("names")
    saved_count = features_section.get("count")
    if not isinstance(saved_names, list) or not all(
        isinstance(name, str) for name in saved_names
    ):
        msg = "Saved metadata feature names must be a list of strings."
        raise ArtifactError(msg)
    if isinstance(saved_count, bool) or not isinstance(saved_count, int):
        msg = "Saved metadata feature count must be an integer."
        raise ArtifactError(msg)
    if saved_count != len(feature_names) or tuple(saved_names) != feature_names:
        msg = "Saved feature metadata does not match the preprocessing pipeline."
        raise ArtifactError(msg)

    if not callable(getattr(preprocessor, "transform", None)) or not callable(
        getattr(preprocessor, "get_feature_names_out", None)
    ):
        msg = "Saved preprocessing artifact is missing the transform contract."
        raise ArtifactError(msg)
    if not callable(getattr(model, "predict", None)) or not callable(
        getattr(model, "predict_proba", None)
    ):
        msg = "Saved model artifact is missing the prediction contract."
        raise ArtifactError(msg)

    model_count = getattr(model, "n_features_in_", None)
    if model_count is not None:
        try:
            normalized_count = int(model_count)
        except (TypeError, ValueError) as exc:
            msg = "Saved model reports an invalid feature count."
            raise ArtifactError(msg) from exc
        if normalized_count != len(feature_names):
            msg = "Saved model feature count does not match preprocessing output."
            raise ArtifactError(msg)

    return selected_model


def load_model_artifacts(
    *,
    data_config: str | Path | None = None,
    train_config: str | Path | None = None,
    models_dir: Path | None = None,
) -> ModelArtifacts:
    """Load the selected model, preprocessor, and metadata from disk."""
    spec = load_dataset_spec(data_config)
    config = load_training_config(data_config, train_config)
    paths = build_artifact_paths(config.artifacts, models_dir=models_dir)
    for label in ("best_model", "preprocessor", "metadata"):
        if not paths[label].is_file():
            msg = (
                f"API artifact '{label}' is missing from the configured model "
                "directory. Train the model before starting the API."
            )
            raise ArtifactError(msg)

    try:
        model = joblib.load(paths["best_model"])
        preprocessor = joblib.load(paths["preprocessor"])
        metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        feature_names = tuple(
            str(name) for name in preprocessor.get_feature_names_out()
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("API artifact load failed: %s: %s", type(exc).__name__, exc)
        msg = (
            f"Could not load saved API artifacts ({type(exc).__name__}). "
            "Verify that the model bundle is complete and compatible."
        )
        raise ArtifactError(msg) from exc
    if not isinstance(metadata, dict):
        msg = "Saved model_metadata.json must contain a JSON object."
        raise ArtifactError(msg)

    selected_model = _validate_metadata_contract(
        metadata, feature_names, model, preprocessor
    )
    return ModelArtifacts(
        spec=spec,
        preprocessor=preprocessor,
        model=model,
        metadata=metadata,
        feature_names=feature_names,
        selected_model=selected_model,
        threshold=read_decision_threshold(metadata),
    )


def get_artifacts() -> ModelArtifacts:
    """FastAPI dependency returning the process-wide saved artifacts."""
    settings = get_settings()
    return _cached_artifacts(str(settings.models_dir))

"""Inference helpers that reuse saved preprocessing and model objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer

from smartpredict.config.schema import DatasetSpec
from smartpredict.core.exceptions import PredictionError
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.validation import validate_dataset
from smartpredict.features.preprocessor import prepare_features_frame
from smartpredict.models.prediction import ModelPredictions, generate_predictions


@dataclass(frozen=True)
class InferenceResult:
    """Predictions and audit details for one batch of raw records."""

    labels: np.ndarray
    probabilities: np.ndarray | None
    feature_names: tuple[str, ...]
    n_records: int
    threshold: float = 0.5


def predict_from_raw(
    records: pd.DataFrame | list[dict[str, object]],
    spec: DatasetSpec,
    preprocessor: ColumnTransformer,
    model: ClassifierMixin,
    threshold: float = 0.5,
) -> InferenceResult:
    """Validate, transform, and predict without fitting anything.

    The preprocessor and model must already be fitted. ``customerID`` is
    retained in the input for traceability but is dropped by the configured
    ``ColumnTransformer`` and never reaches the estimator. Positive-class
    probabilities are produced independently of ``threshold``; only the final
    binary labels are derived from the supplied cutoff.

    Args:
        records: Raw customer rows with the configured schema columns.
        spec: Dataset contract used during training.
        preprocessor: Fitted transformer loaded from the saved artifact.
        model: Fitted classifier loaded from the saved artifact.
        threshold: Positive-class decision threshold for binary labels.

    Returns:
        Threshold-dependent binary labels, threshold-independent probabilities,
        feature names, the record count, and the applied threshold.

    Raises:
        PredictionError: If validation, transformation, or prediction fails,
            if the threshold is invalid, or if the identifier leaked into the
            transformed features.

    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        msg = "Decision threshold must be a number between 0 and 1."
        raise PredictionError(msg)
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        msg = "Decision threshold must be between 0 and 1."
        raise PredictionError(msg)

    frame = (
        records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    )
    coerced, report = validate_dataset(
        frame, DatasetSchema.from_spec(spec), require_target=False
    )
    try:
        report.raise_if_invalid()
    except Exception as exc:
        msg = f"Inference records failed validation: {exc}"
        raise PredictionError(msg) from exc

    try:
        prepared = prepare_features_frame(coerced, spec)
        features = preprocessor.transform(prepared)
        predictions: ModelPredictions = generate_predictions(model, features)
    except Exception as exc:
        msg = f"Inference could not be completed: {exc}"
        raise PredictionError(msg) from exc

    feature_names = tuple(
        str(name) for name in preprocessor.get_feature_names_out()
    )
    if any(
        name == spec.id_column or name.startswith(f"{spec.id_column}_")
        for name in feature_names
    ):
        msg = f"Identifier column '{spec.id_column}' leaked into model features."
        raise PredictionError(msg)

    labels = (
        (np.asarray(predictions.probabilities, dtype=float) >= threshold).astype(int)
        if predictions.probabilities is not None
        else np.asarray(predictions.labels)
    )
    if labels.shape != (frame.shape[0],):
        msg = (
            f"Prediction shape {labels.shape} does not match "
            f"{frame.shape[0]} input records."
        )
        raise PredictionError(msg)

    return InferenceResult(
        labels=labels,
        probabilities=predictions.probabilities,
        feature_names=feature_names,
        n_records=int(frame.shape[0]),
        threshold=threshold,
    )

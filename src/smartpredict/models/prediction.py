"""Shared helpers for converting model outputs to binary evaluation arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from smartpredict.core.exceptions import EvaluationError

POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0


@dataclass(frozen=True)
class ModelPredictions:
    """Integer labels and optional positive-class probabilities."""

    labels: np.ndarray
    probabilities: np.ndarray | None


def binary_targets(
    values: object, positive_label: str
) -> np.ndarray:
    """Convert raw target labels to ``0``/``1`` integers.

    String labels are compared with the configured positive label. Already
    encoded ``0``/``1`` targets are also accepted.

    Args:
        values: Series or array-like of target labels.
        positive_label: Configured label representing the positive class.

    Returns:
        Integer array where ``1`` denotes the positive class.

    Raises:
        EvaluationError: If labels cannot be mapped to a binary target.

    """
    series = pd.Series(values).astype("string").str.strip()
    unique = {value for value in series.dropna().unique() if value}
    if positive_label in unique:
        return np.asarray(series.eq(positive_label).astype(int), dtype=int)

    if unique and unique.issubset({"0", "1"}):
        return np.asarray(series.astype(int), dtype=int)

    msg = (
        "Target values must contain the configured positive label "
        f"'{positive_label}' or already-encoded 0/1 values; got "
        f"{sorted(unique)}."
    )
    raise EvaluationError(msg)


def positive_probabilities(
    model: ClassifierMixin, features: np.ndarray
) -> np.ndarray | None:
    """Return positive-class probabilities when the model supports them.

    Args:
        model: Fitted classifier.
        features: Preprocessed feature matrix.

    Returns:
        One-dimensional probability array, or ``None`` when probabilities are
        unavailable or the positive class is absent from the fitted model.

    """
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None
    try:
        probabilities = np.asarray(predict_proba(features))
    except (TypeError, ValueError):
        return None
    if probabilities.ndim != 2:  # noqa: PLR2004
        return None

    classes = getattr(model, "classes_", None)
    if classes is None:
        return None
    class_values = [str(value) for value in np.asarray(classes)]
    if str(POSITIVE_CLASS) not in class_values:
        return None
    positive_index = class_values.index(str(POSITIVE_CLASS))
    result = np.asarray(probabilities[:, positive_index], dtype=float)
    if not np.isfinite(result).all():
        return None
    return result


def generate_predictions(
    model: ClassifierMixin, features: np.ndarray
) -> ModelPredictions:
    """Generate labels and positive probabilities for one feature matrix."""
    raw_labels = np.asarray(model.predict(features))
    labels = raw_labels.astype(int)
    return ModelPredictions(
        labels=labels,
        probabilities=positive_probabilities(model, features),
    )

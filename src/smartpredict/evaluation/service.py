"""Evaluation orchestration for fitted classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin

from smartpredict.evaluation.metrics import EvaluationOutput, evaluate_predictions
from smartpredict.models.prediction import binary_targets, generate_predictions


def evaluate_model(
    model: ClassifierMixin,
    features: np.ndarray,
    raw_target: object,
    *,
    model_name: str,
    split: str,
    positive_label: str,
) -> EvaluationOutput:
    """Evaluate a fitted model on one split without modifying the model.

    Args:
        model: Fitted classifier.
        features: Preprocessed feature matrix for the split.
        raw_target: Original target labels for the split.
        model_name: Model name recorded in the report.
        split: Split label, used to distinguish validation from test results.
        positive_label: Configured positive-class label.

    Returns:
        Evaluation result and the arrays used for plotting.

    """
    y_true = binary_targets(raw_target, positive_label)
    predictions = generate_predictions(model, features)
    result = evaluate_predictions(
        y_true,
        predictions.labels,
        predictions.probabilities,
        model_name=model_name,
        split=split,
    )
    return EvaluationOutput(
        result=result,
        y_true=y_true,
        y_pred=predictions.labels,
        y_prob=predictions.probabilities,
    )

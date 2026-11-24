"""Tests for evaluation metrics and edge cases."""

from __future__ import annotations

from typing import Any

import numpy as np

from smartpredict.config.schema import MetricName
from smartpredict.evaluation.metrics import evaluate_predictions
from smartpredict.evaluation.service import evaluate_model
from smartpredict.models.prediction import binary_targets


def test_evaluate_predictions_computes_all_core_metrics() -> None:
    """Core metrics and matrix use the positive class consistently."""
    truth = np.array([0, 0, 1, 1, 0, 1])
    predicted = np.array([0, 1, 1, 1, 0, 0])
    probabilities = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4])

    result = evaluate_predictions(
        truth,
        predicted,
        probabilities,
        model_name="logistic_regression",
        split="validation",
    )
    assert result.metrics[MetricName.ACCURACY.value] == 4 / 6
    assert result.metrics[MetricName.PRECISION.value] == 2 / 3
    assert result.metrics[MetricName.RECALL.value] == 2 / 3
    assert result.metrics[MetricName.ROC_AUC.value] is not None
    assert result.confusion == ((2, 1), (1, 2))
    assert result.classification_report["positive"]["f1-score"] == 2 / 3
    assert result.probability_available is True


def test_roc_auc_is_skipped_without_probabilities() -> None:
    """Models without predict_proba yield a clear warning, not a crash."""
    result = evaluate_predictions(
        [0, 1, 1, 0], [0, 1, 0, 0], model_name="tree", split="test"
    )
    assert result.metrics[MetricName.ROC_AUC.value] is None
    assert result.probability_available is False
    assert any("Probability predictions" in item for item in result.warnings)


def test_roc_auc_is_skipped_for_single_class() -> None:
    """ROC-AUC is undefined when only one true class is present."""
    result = evaluate_predictions(
        [1, 1, 1], [1, 1, 0], [0.6, 0.7, 0.4], model_name="m", split="test"
    )
    assert result.metrics[MetricName.ROC_AUC.value] is None
    assert any("one class" in item for item in result.warnings)


def test_invalid_probabilities_are_ignored() -> None:
    """NaN probability arrays do not break metric computation."""
    result = evaluate_predictions(
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0.1, np.nan, 0.2, 0.8],
        model_name="m",
        split="test",
    )
    assert result.metrics[MetricName.ROC_AUC.value] is None
    assert any("invalid" in item for item in result.warnings)


def test_binary_targets_supports_raw_labels_and_encoded_values() -> None:
    """Target conversion handles configured labels and encoded integers."""
    np.testing.assert_array_equal(
        binary_targets(["Yes", "No", "Yes"], "Yes"), np.array([1, 0, 1])
    )
    np.testing.assert_array_equal(
        binary_targets([1, 0, 1], "Yes"), np.array([1, 0, 1])
    )


def test_evaluate_model_without_predict_proba() -> None:
    """A model exposing only predict can still be evaluated."""

    class LabelOnlyModel:
        """Minimal classifier without probability predictions."""

        def predict(self, features: Any) -> np.ndarray:
            """Return fixed labels."""
            return np.asarray(features).reshape(-1)

    output = evaluate_model(
        LabelOnlyModel(),
        np.array([[0], [1], [1], [0]]),
        ["No", "Yes", "Yes", "No"],
        model_name="label_only",
        split="test",
        positive_label="Yes",
    )
    assert output.y_prob is None
    assert output.result.metrics[MetricName.ROC_AUC.value] is None


def test_evaluation_result_is_json_serializable() -> None:
    """Reports can be written directly as JSON payloads."""
    result = evaluate_predictions(
        [0, 1, 0, 1], [0, 1, 1, 1], [0.2, 0.7, 0.6, 0.9],
        model_name="m", split="validation"
    )
    payload = result.to_dict()
    assert isinstance(payload["confusion_matrix"], list)
    assert isinstance(payload["classification_report"]["positive"], dict)

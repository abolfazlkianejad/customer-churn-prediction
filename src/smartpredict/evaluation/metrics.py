"""Binary classification metrics with explicit edge-case handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from smartpredict.config.schema import MetricName

LABEL_NAMES = ("negative", "positive")


def _json_value(value: Any) -> Any:
    """Convert numpy scalars and nested containers to JSON-safe values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class EvaluationResult:
    """Structured metrics for one model on one dataset split."""

    model_name: str
    split: str
    metrics: dict[str, float | None]
    confusion: tuple[tuple[int, int], tuple[int, int]]
    classification_report: dict[str, Any]
    n_samples: int
    positive_class: int
    probability_available: bool
    warnings: tuple[str, ...]
    evaluated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the evaluation."""
        return {
            "model_name": self.model_name,
            "split": self.split,
            "metrics": dict(self.metrics),
            "confusion_matrix": [list(row) for row in self.confusion],
            "classification_report": _json_value(self.classification_report),
            "n_samples": self.n_samples,
            "positive_class": self.positive_class,
            "probability_available": self.probability_available,
            "warnings": list(self.warnings),
            "evaluated_at": self.evaluated_at,
        }


@dataclass(frozen=True)
class EvaluationOutput:
    """Evaluation result plus the arrays required by plotting functions."""

    result: EvaluationResult
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray | None


def evaluate_predictions(
    y_true: object,
    y_pred: object,
    y_prob: object | None = None,
    *,
    model_name: str,
    split: str,
) -> EvaluationResult:
    """Compute binary classification metrics from prediction arrays.

    ROC-AUC and probability-based metrics are omitted with an explanatory
    warning when probabilities are unavailable or the true labels contain only
    one class.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted positive-class probabilities, if available.
        model_name: Name recorded in the result.
        split: Dataset split recorded in the result (``validation`` or ``test``).

    Returns:
        Structured evaluation result.

    """
    truth = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    warnings: list[str] = []

    probabilities: np.ndarray | None = None
    if y_prob is not None:
        candidate = np.asarray(y_prob, dtype=float)
        if candidate.ndim == 1 and np.isfinite(candidate).all():
            probabilities = candidate
        else:
            warnings.append("Probabilities were invalid and were ignored.")

    two_classes = len(np.unique(truth)) == 2
    if probabilities is None:
        warnings.append(
            "Probability predictions are unavailable; probability-based metrics "
            "were skipped."
        )
    elif not two_classes:
        warnings.append(
            "Only one class is present in the true labels; probability-based "
            "ranking metrics were skipped."
        )

    metrics: dict[str, float | None] = {
        MetricName.ACCURACY.value: float(accuracy_score(truth, predictions)),
        MetricName.PRECISION.value: float(
            precision_score(truth, predictions, pos_label=1, zero_division=0)
        ),
        MetricName.RECALL.value: float(
            recall_score(truth, predictions, pos_label=1, zero_division=0)
        ),
        MetricName.F1.value: float(
            f1_score(truth, predictions, pos_label=1, zero_division=0)
        ),
        MetricName.ROC_AUC.value: None,
        MetricName.AVERAGE_PRECISION.value: None,
        MetricName.BRIER_SCORE.value: None,
        MetricName.LOG_LOSS.value: None,
    }

    if probabilities is not None and two_classes:
        metrics[MetricName.ROC_AUC.value] = float(
            roc_auc_score(truth, probabilities)
        )
        metrics[MetricName.AVERAGE_PRECISION.value] = float(
            average_precision_score(truth, probabilities)
        )
        metrics[MetricName.BRIER_SCORE.value] = float(
            brier_score_loss(truth, np.clip(probabilities, 0.0, 1.0))
        )
        clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
        metrics[MetricName.LOG_LOSS.value] = float(
            log_loss(truth, np.column_stack([1.0 - clipped, clipped]), labels=[0, 1])
        )

    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    report = classification_report(
        truth,
        predictions,
        labels=[0, 1],
        target_names=list(LABEL_NAMES),
        output_dict=True,
        zero_division=0,
    )
    return EvaluationResult(
        model_name=model_name,
        split=split,
        metrics=metrics,
        confusion=(
            (int(matrix[0, 0]), int(matrix[0, 1])),
            (int(matrix[1, 0]), int(matrix[1, 1])),
        ),
        classification_report=report,
        n_samples=int(truth.size),
        positive_class=1,
        probability_available=probabilities is not None,
        warnings=tuple(warnings),
        evaluated_at=datetime.now(tz=UTC).isoformat(),
    )


def roc_points(
    y_true: object, y_prob: object
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ROC curve coordinates when the inputs are informative."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_prob, dtype=float)
    if len(np.unique(truth)) != 2:
        return None
    false_positive, true_positive, _ = roc_curve(truth, scores)
    return false_positive, true_positive


def precision_recall_points(
    y_true: object, y_prob: object
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return precision-recall curve coordinates when inputs are informative."""
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_prob, dtype=float)
    if len(np.unique(truth)) != 2:
        return None
    precision, recall, _ = precision_recall_curve(truth, scores)
    return precision, recall

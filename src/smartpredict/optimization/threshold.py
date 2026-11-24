"""Validation-only decision-threshold analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from smartpredict.config.schema import ThresholdConfig, ThresholdCriterion
from smartpredict.core.exceptions import OptimizationError
from smartpredict.core.logging import get_logger

logger = get_logger(__name__)
BASELINE_THRESHOLD = 0.5


@dataclass(frozen=True)
class ThresholdMetrics:
    """All requested metrics for one candidate threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    accuracy: float
    specificity: float
    selection_score: float
    eligible: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe metric row."""
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "specificity": self.specificity,
            "selection_score": self.selection_score,
            "eligible": self.eligible,
        }


@dataclass(frozen=True)
class ThresholdAnalysisResult:
    """Complete threshold grid, baseline, and configured selection outcome."""

    source_split: str
    criterion: str
    weights: dict[str, float]
    constraints: dict[str, float | None]
    baseline_threshold: float
    rows: tuple[ThresholdMetrics, ...]
    selected: ThresholdMetrics
    baseline: ThresholdMetrics
    eligible_count: int
    fallback_used: bool
    generated_at: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe threshold analysis report."""
        return {
            "schema_version": "1.0",
            "source_split": self.source_split,
            "criterion": self.criterion,
            "weights": dict(self.weights),
            "constraints": dict(self.constraints),
            "baseline_threshold": self.baseline_threshold,
            "selected_threshold": self.selected.threshold,
            "selected_metrics": self.selected.to_dict(),
            "baseline_metrics": self.baseline.to_dict(),
            "eligible_thresholds": self.eligible_count,
            "fallback_used": self.fallback_used,
            "thresholds": [row.to_dict() for row in self.rows],
            "generated_at": self.generated_at,
            "note": self.note,
        }


def _threshold_grid(config: ThresholdConfig) -> list[float]:
    """Build an ordered grid that always includes the 0.5 baseline."""
    values: list[float] = []
    current = float(config.minimum)
    tolerance = 1e-9
    while current <= float(config.maximum) + tolerance:
        values.append(round(current, 10))
        current += float(config.step)
    if not any(abs(value - BASELINE_THRESHOLD) < tolerance for value in values):
        values.append(BASELINE_THRESHOLD)
    return sorted(set(values))


def _selection_score(
    metrics: dict[str, float], config: ThresholdConfig
) -> float:
    """Apply the configured business objective to one metric row."""
    if config.criterion is ThresholdCriterion.WEIGHTED_SCORE:
        return float(
            sum(
                float(weight) * metrics[name]
                for name, weight in config.weights.items()
                if name in metrics
            )
        )
    return float(metrics[config.criterion.value])


def _metrics_at_threshold(
    truth: np.ndarray, scores: np.ndarray, threshold: float, config: ThresholdConfig
) -> ThresholdMetrics:
    """Compute all threshold metrics for one cutoff."""
    labels = (scores >= threshold).astype(int)
    true_negative = int(np.sum((truth == 0) & (labels == 0)))
    false_positive = int(np.sum((truth == 0) & (labels == 1)))
    values = {
        "precision": float(
            precision_score(truth, labels, pos_label=1, zero_division=0)
        ),
        "recall": float(recall_score(truth, labels, pos_label=1, zero_division=0)),
        "f1": float(f1_score(truth, labels, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(truth, labels)),
        "specificity": (
            true_negative / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        ),
    }
    eligible = True
    for key, minimum in (
        ("precision", config.min_precision),
        ("recall", config.min_recall),
        ("specificity", config.min_specificity),
    ):
        if minimum is not None and values[key] < float(minimum):
            eligible = False
            break
    return ThresholdMetrics(
        threshold=round(float(threshold), 10),
        precision=values["precision"],
        recall=values["recall"],
        f1=values["f1"],
        accuracy=values["accuracy"],
        specificity=values["specificity"],
        selection_score=_selection_score(values, config),
        eligible=eligible,
    )


def analyze_thresholds(
    y_true: object,
    probabilities: object,
    config: ThresholdConfig,
    *,
    source_split: str = "validation",
) -> ThresholdAnalysisResult:
    """Analyze thresholds using validation probabilities only.

    Args:
        y_true: Ground-truth binary validation labels.
        probabilities: Positive-class validation probabilities.
        config: Configurable threshold grid and business objective.
        source_split: Label recorded in the report; always validation in the
            production workflow.

    Returns:
        Full threshold grid with baseline and selected rows.

    Raises:
        OptimizationError: If probabilities are invalid or the grid is unusable.

    """
    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if truth.ndim != 1 or scores.ndim != 1 or truth.shape != scores.shape:
        msg = "Threshold inputs must be equal-length one-dimensional arrays."
        raise OptimizationError(msg)
    if scores.size == 0:
        msg = "Threshold analysis requires at least one validation row."
        raise OptimizationError(msg)
    if not np.isfinite(scores).all() or np.any(scores < 0) or np.any(scores > 1):
        msg = "Threshold probabilities must be finite values between 0 and 1."
        raise OptimizationError(msg)

    rows = tuple(
        _metrics_at_threshold(truth, scores, threshold, config)
        for threshold in _threshold_grid(config)
    )
    baseline = next(
        row for row in rows if abs(row.threshold - BASELINE_THRESHOLD) < 1e-9
    )
    if not config.enabled:
        eligible = [baseline]
    else:
        eligible = [row for row in rows if row.eligible]
    fallback_used = False
    if not eligible:
        eligible = list(rows)
        fallback_used = True
        logger.warning(
            "No threshold satisfied the configured business constraints; "
            "falling back to the full validation grid."
        )
    selected = max(
        eligible,
        key=lambda row: (round(row.selection_score, 12), row.threshold),
    )

    if not config.enabled:
        note = "Threshold analysis is disabled; the 0.5 baseline is retained."
    elif fallback_used:
        note = (
            "No threshold satisfied all configured constraints, so selection "
            "used the full validation grid."
        )
    else:
        note = (
            "The selected threshold maximizes the configured business "
            "criterion among thresholds satisfying all constraints."
        )
    return ThresholdAnalysisResult(
        source_split=source_split,
        criterion=config.criterion.value,
        weights=dict(config.weights),
        constraints={
            "min_precision": config.min_precision,
            "min_recall": config.min_recall,
            "min_specificity": config.min_specificity,
        },
        baseline_threshold=BASELINE_THRESHOLD,
        rows=rows,
        selected=selected,
        baseline=baseline,
        eligible_count=len(eligible),
        fallback_used=fallback_used,
        generated_at=datetime.now(tz=UTC).isoformat(),
        note=note,
    )


def save_threshold_plots(
    result: ThresholdAnalysisResult, output_dir: Path
) -> dict[str, str]:
    """Save precision/recall and F1 plots over the threshold grid.

    Args:
        result: Completed threshold analysis.
        output_dir: Destination directory.

    Returns:
        Mapping from plot label to written path.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [row.threshold for row in result.rows]
    paths = {
        "threshold_precision_recall": output_dir / "threshold_precision_recall.png",
        "threshold_f1": output_dir / "threshold_f1.png",
    }

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(thresholds, [row.precision for row in result.rows], label="precision")
    axes.plot(thresholds, [row.recall for row in result.rows], label="recall")
    axes.axvline(
        result.baseline.threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="baseline 0.5",
    )
    axes.axvline(
        result.selected.threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"selected {result.selected.threshold:.2f}",
    )
    axes.set_xlabel("Decision threshold")
    axes.set_ylabel("Score")
    axes.set_title("Precision and recall by validation threshold")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(paths["threshold_precision_recall"], dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(figsize=(8, 5))
    axes.plot(thresholds, [row.f1 for row in result.rows], color="green")
    axes.axvline(
        result.baseline.threshold,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="baseline 0.5",
    )
    axes.axvline(
        result.selected.threshold,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"selected {result.selected.threshold:.2f}",
    )
    axes.set_xlabel("Decision threshold")
    axes.set_ylabel("F1")
    axes.set_title("F1 by validation threshold")
    axes.legend()
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(paths["threshold_f1"], dpi=150)
    plt.close(figure)

    return {name: str(path) for name, path in paths.items()}

"""Synthetic tests for validation-only threshold metrics and selection."""

from __future__ import annotations

import numpy as np
import pytest

from smartpredict.config.schema import ThresholdCriterion
from smartpredict.core.exceptions import OptimizationError
from smartpredict.optimization.threshold import (
    BASELINE_THRESHOLD,
    ThresholdConfig,
    analyze_thresholds,
)


def test_threshold_metrics_are_calculated_at_baseline() -> None:
    """All requested metrics are computed for the preserved 0.5 cutoff."""
    config = ThresholdConfig(minimum=0.1, maximum=0.9, step=0.1)
    result = analyze_thresholds(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.4, 0.6, 0.9]),
        config,
    )

    assert result.baseline.threshold == BASELINE_THRESHOLD
    assert result.source_split == "validation"
    assert result.baseline.precision == 1.0
    assert result.baseline.recall == 1.0
    assert result.baseline.f1 == 1.0
    assert result.baseline.accuracy == 1.0
    assert result.baseline.specificity == 1.0
    assert result.to_dict()["baseline_metrics"]["threshold"] == 0.5


def test_selection_respects_configured_business_criterion() -> None:
    """Recall and F1 criteria can select different synthetic thresholds."""
    truth = np.asarray([0, 0, 0, 0, 0, 0, 1, 1])
    scores = np.asarray([0.2, 0.3, 0.4, 0.79, 0.6, 0.7, 0.5, 0.9])

    recall_result = analyze_thresholds(
        truth,
        scores,
        ThresholdConfig(criterion=ThresholdCriterion.RECALL),
    )
    f1_result = analyze_thresholds(
        truth,
        scores,
        ThresholdConfig(criterion=ThresholdCriterion.F1),
    )

    assert recall_result.selected.threshold == 0.5
    assert recall_result.selected.recall == 1.0
    assert f1_result.selected.threshold == 0.9
    assert f1_result.selected.recall == 0.5
    assert recall_result.selected.threshold != f1_result.selected.threshold


def test_unattainable_constraint_falls_back_explicitly() -> None:
    """Impossible business constraints are reported rather than silently ignored."""
    result = analyze_thresholds(
        np.asarray([0, 1]),
        np.asarray([0.9, 0.1]),
        ThresholdConfig(min_precision=0.8),
    )

    assert result.fallback_used is True
    assert result.eligible_count == len(result.rows)
    assert "constraints" in result.note


def test_invalid_probabilities_are_rejected() -> None:
    """Non-probability scores never enter threshold selection."""
    with pytest.raises(OptimizationError, match="finite values between 0 and 1"):
        analyze_thresholds(
            np.asarray([0, 1]),
            np.asarray([0.2, 1.4]),
            ThresholdConfig(),
        )

    with pytest.raises(OptimizationError, match="equal-length"):
        analyze_thresholds(
            np.asarray([0, 1]),
            np.asarray([0.2]),
            ThresholdConfig(),
        )

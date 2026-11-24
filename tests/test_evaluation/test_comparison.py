"""Tests for validation-only model comparison and selection."""

from __future__ import annotations

import numpy as np
import pytest

from smartpredict.config.schema import MetricName, OptimizationDirection
from smartpredict.core.exceptions import EvaluationError
from smartpredict.evaluation.comparison import compare_models, select_best_model
from smartpredict.evaluation.metrics import evaluate_predictions
from smartpredict.models.prediction import ModelPredictions
from smartpredict.models.training import TrainingResult


def _training(name: str) -> TrainingResult:
    """Create a lightweight training result for comparison tests."""
    return TrainingResult(
        name=name,
        model=object(),
        validation_predictions=ModelPredictions(
            labels=np.array([0, 1]), probabilities=None
        ),
        metadata={"model_name": name},
    )


def _result(
    name: str, probabilities: list[float] | None = None
) -> object:
    """Create a validation evaluation result with controlled metrics."""
    return evaluate_predictions(
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        probabilities,
        model_name=name,
        split="validation",
    )


def test_compare_models_selects_highest_roc_auc() -> None:
    """The default direction selects the strongest validation score."""
    results = [_training("logistic_regression"), _training("decision_tree")]
    evaluations = {
        "logistic_regression": _result(
            "logistic_regression", [0.1, 0.4, 0.6, 0.9]
        ),
        "decision_tree": _result("decision_tree", [0.4, 0.6, 0.3, 0.7]),
    }
    report = compare_models(
        results,
        evaluations,
        primary_metric=MetricName.ROC_AUC,
        direction=OptimizationDirection.MAXIMIZE,
    )
    assert report.selected_model == "logistic_regression"
    assert [item.model_name for item in report.rankings][0] == "logistic_regression"
    best = select_best_model(report, results)
    assert best.name == "logistic_regression"


def test_compare_models_supports_minimization() -> None:
    """Ascending optimization direction is honored for loss metrics."""
    results = [_training("a"), _training("b")]
    evaluations = {
        "a": _result("a", [0.9, 0.8, 0.2, 0.1]),
        "b": _result("b", [0.1, 0.2, 0.8, 0.9]),
    }
    report = compare_models(
        results,
        evaluations,
        primary_metric=MetricName.LOG_LOSS,
        direction=OptimizationDirection.MINIMIZE,
    )
    assert report.optimization_direction == "minimize"
    assert report.selected_model == "b"


def test_failed_models_are_retained_without_losing_successes() -> None:
    """Training failures appear in the report but do not stop selection."""
    results = [_training("a")]
    evaluations = {"a": _result("a", [0.1, 0.2, 0.8, 0.9])}
    report = compare_models(
        results,
        evaluations,
        primary_metric=MetricName.ROC_AUC,
        direction=OptimizationDirection.MAXIMIZE,
        failure_messages={"xgboost": "Missing dependency"},
    )
    assert report.selected_model == "a"
    assert report.failures[0].model_name == "xgboost"
    assert report.to_dict()["failures"][0]["message"] == "Missing dependency"


def test_models_without_primary_metric_rank_last() -> None:
    """A missing score is placed last even when maximizing."""
    results = [_training("no_probability"), _training("with_probability")]
    evaluations = {
        "no_probability": _result("no_probability"),
        "with_probability": _result(
            "with_probability", [0.1, 0.2, 0.8, 0.9]
        ),
    }
    report = compare_models(
        results,
        evaluations,
        primary_metric=MetricName.ROC_AUC,
        direction=OptimizationDirection.MAXIMIZE,
    )
    assert [item.model_name for item in report.rankings] == [
        "with_probability",
        "no_probability",
    ]


def test_select_best_model_requires_success() -> None:
    """An empty comparison cannot produce a selected model."""
    report = compare_models(
        [],
        {},
        primary_metric=MetricName.ROC_AUC,
        direction=OptimizationDirection.MAXIMIZE,
    )
    with pytest.raises(EvaluationError, match="No model"):
        select_best_model(report, [])

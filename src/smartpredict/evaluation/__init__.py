"""Metrics, model comparison, and visualization layer."""

from smartpredict.evaluation.comparison import (
    ModelComparisonReport,
    ModelFailure,
    ModelRanking,
    compare_models,
    select_best_model,
)
from smartpredict.evaluation.metrics import (
    EvaluationOutput,
    EvaluationResult,
    evaluate_predictions,
)
from smartpredict.evaluation.plots import generate_evaluation_plots
from smartpredict.evaluation.service import evaluate_model

__all__ = [
    "EvaluationOutput",
    "EvaluationResult",
    "ModelComparisonReport",
    "ModelFailure",
    "ModelRanking",
    "compare_models",
    "evaluate_model",
    "evaluate_predictions",
    "generate_evaluation_plots",
    "select_best_model",
]

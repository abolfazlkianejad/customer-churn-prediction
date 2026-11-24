"""Validation-only model comparison and best-model selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from smartpredict.config.schema import MetricName, OptimizationDirection
from smartpredict.core.exceptions import EvaluationError
from smartpredict.evaluation.metrics import EvaluationResult

if TYPE_CHECKING:
    from smartpredict.models.training import TrainingResult


@dataclass(frozen=True)
class ModelRanking:
    """Validation ranking entry for one successfully trained model."""

    model_name: str
    primary_score: float | None
    metrics: dict[str, float | None]
    training_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ranking entry."""
        return {
            "model_name": self.model_name,
            "primary_score": self.primary_score,
            "metrics": dict(self.metrics),
            "training_metadata": dict(self.training_metadata),
        }


@dataclass(frozen=True)
class ModelFailure:
    """Failure details for one model that could not be evaluated."""

    model_name: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable failure entry."""
        return {
            "model_name": self.model_name,
            "error_type": self.error_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class ModelComparisonReport:
    """Structured comparison of all candidate models on validation data."""

    primary_metric: str
    optimization_direction: str
    rankings: tuple[ModelRanking, ...]
    failures: tuple[ModelFailure, ...]
    selected_model: str | None
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable comparison report."""
        return {
            "selection_split": "validation",
            "primary_metric": self.primary_metric,
            "optimization_direction": self.optimization_direction,
            "rankings": [entry.to_dict() for entry in self.rankings],
            "failures": [failure.to_dict() for failure in self.failures],
            "selected_model": self.selected_model,
            "created_at": self.created_at,
        }


def compare_models(
    training_results: list[TrainingResult],
    validation_results: dict[str, EvaluationResult],
    *,
    primary_metric: MetricName | str,
    direction: OptimizationDirection | str,
    failure_messages: dict[str, str] | None = None,
) -> ModelComparisonReport:
    """Rank successful models by one validation metric.

    Missing primary-metric values are always ranked last, regardless of the
    optimization direction. Failed models are retained in the report without
    discarding successful results.

    Args:
        training_results: fitted models with metadata.
        validation_results: validation evaluations keyed by model name.
        primary_metric: Metric used to rank candidates.
        direction: ``maximize`` or ``minimize``.
        failure_messages: Optional failure explanations keyed by model name.

    Returns:
        Structured comparison report.

    Raises:
        EvaluationError: If the primary metric is not reported.

    """
    metric = MetricName(primary_metric).value
    optimization = OptimizationDirection(direction)
    rankings: list[ModelRanking] = []

    for training in training_results:
        evaluation = validation_results.get(training.name)
        if evaluation is None or evaluation.split != "validation":
            continue
        if metric not in evaluation.metrics:
            msg = f"Primary metric '{metric}' is not available for comparison."
            raise EvaluationError(msg)
        rankings.append(
            ModelRanking(
                model_name=training.name,
                primary_score=evaluation.metrics[metric],
                metrics=dict(evaluation.metrics),
                training_metadata=dict(training.metadata),
            )
        )

    descending = optimization is OptimizationDirection.MAXIMIZE

    def ranking_key(item: ModelRanking) -> tuple[bool, float, str]:
        """Sort valid scores first, then by optimization direction and name."""
        score = item.primary_score
        if score is None:
            return True, 0.0, item.model_name
        return False, -score if descending else score, item.model_name

    rankings.sort(key=ranking_key)

    evaluated_names = {entry.model_name for entry in rankings}
    failures = tuple(
        ModelFailure(
            model_name=name,
            error_type="ModelTrainingError",
            message=message,
        )
        for name, message in sorted((failure_messages or {}).items())
        if name not in evaluated_names
    )

    return ModelComparisonReport(
        primary_metric=metric,
        optimization_direction=optimization.value,
        rankings=tuple(rankings),
        failures=failures,
        selected_model=rankings[0].model_name if rankings else None,
        created_at=datetime.now(tz=UTC).isoformat(),
    )


def select_best_model(
    report: ModelComparisonReport, training_results: list[TrainingResult]
) -> TrainingResult:
    """Return the training result chosen by a comparison report.

    Args:
        report: Validation-only comparison report.
        training_results: Results produced during training.

    Returns:
        Best-ranked training result.

    Raises:
        EvaluationError: If no model succeeded or the selection is inconsistent.

    """
    if report.selected_model is None:
        msg = "No model was trained successfully; cannot select a best model."
        raise EvaluationError(msg)
    for result in training_results:
        if result.name == report.selected_model:
            return result
    msg = f"Selected model '{report.selected_model}' has no training result."
    raise EvaluationError(msg)

"""Leakage-safe cross-validation for Optuna trials."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from smartpredict.config.schema import (
    DatasetSpec,
    FeatureConfig,
    MetricName,
    OptimizationDirection,
    SupportedModel,
)
from smartpredict.core.exceptions import OptimizationError
from smartpredict.core.logging import get_logger
from smartpredict.evaluation.metrics import evaluate_predictions
from smartpredict.features.preprocessor import (
    build_preprocessor,
    prepare_features_frame,
)
from smartpredict.models.factory import create_model
from smartpredict.models.prediction import binary_targets, generate_predictions

logger = get_logger(__name__)

PreprocessorFactory = Callable[[DatasetSpec, FeatureConfig], ColumnTransformer]
ModelFactory = Callable[[str, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class FoldResult:
    """Objective score and audit indices for one stratified fold."""

    fold: int
    fit_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    objective_value: float
    raw_metric: float
    metrics: dict[str, float | None]
    duration_seconds: float


@dataclass(frozen=True)
class CrossValidationResult:
    """Aggregated fold scores for one trial's hyperparameters."""

    objective_mean: float
    objective_std: float
    raw_metric: str
    raw_metric_mean: float
    mean_metrics: dict[str, float | None]
    folds: tuple[FoldResult, ...]
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the cross-validation result."""
        return {
            "objective_mean": self.objective_mean,
            "objective_std": self.objective_std,
            "metric": self.raw_metric,
            "metric_mean": self.raw_metric_mean,
            "mean_metrics": dict(self.mean_metrics),
            "n_samples": self.n_samples,
            "folds": [
                {
                    "fold": fold.fold,
                    "n_fit_rows": len(fold.fit_indices),
                    "n_validation_rows": len(fold.validation_indices),
                    "fit_indices": list(fold.fit_indices),
                    "validation_indices": list(fold.validation_indices),
                    "objective_value": fold.objective_value,
                    "raw_metric": fold.raw_metric,
                    "metrics": dict(fold.metrics),
                    "duration_seconds": fold.duration_seconds,
                }
                for fold in self.folds
            ],
        }


def _objective_value(value: float, direction: OptimizationDirection) -> float:
    """Convert a raw metric into Optuna's maximization objective."""
    return value if direction is OptimizationDirection.MAXIMIZE else -value


def cross_validate_params(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    feature_config: FeatureConfig,
    model_name: SupportedModel,
    params: Mapping[str, Any],
    *,
    cv_folds: int,
    metric: MetricName,
    direction: OptimizationDirection,
    seed: int,
    positive_label: str,
    preprocessor_factory: PreprocessorFactory = build_preprocessor,
    model_factory: ModelFactory | None = None,
) -> CrossValidationResult:
    """Score parameters with stratified CV using only the supplied train rows.

    A fresh preprocessor is built and fitted inside every fold, so imputation,
    scaling, and categorical categories are learned from fold-training rows
    only. Validation rows are used solely for scoring. Callers must supply the
    training split; validation and test splits are never accessed here.

    Args:
        frame: Raw training rows including the configured target column.
        spec: Dataset contract used for validation and preprocessing.
        feature_config: Preprocessing choices from ``train.yaml``.
        model_name: Model type to fit in each fold.
        params: Candidate hyperparameters.
        cv_folds: Number of stratified folds.
        metric: Metric reported for each fold.
        direction: Optimization direction for the metric.
        seed: Seed for shuffling and model construction.
        positive_label: Configured positive-class label.
        preprocessor_factory: Injectable unfitted transformer factory.
        model_factory: Injectable unfitted classifier factory.

    Returns:
        Aggregate and per-fold objective results.

    Raises:
        OptimizationError: If folding, preprocessing, fitting, or scoring fails.

    """
    target_column = spec.target_column
    if target_column not in frame.columns:
        msg = f"Training frame is missing target column '{target_column}'."
        raise OptimizationError(msg)

    try:
        y = binary_targets(frame[target_column], positive_label)
        prepared = prepare_features_frame(frame, spec)
        splitter = StratifiedKFold(
            n_splits=cv_folds, shuffle=True, random_state=seed
        )
        folds = list(splitter.split(prepared, y))
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not create {cv_folds} stratified folds: {exc}"
        raise OptimizationError(msg) from exc

    def default_model_factory(name: str, candidate: Mapping[str, Any]) -> Any:
        """Create a model with the configured experiment seed."""
        return create_model(name, candidate, seed=seed)

    active_model_factory = model_factory or default_model_factory
    fold_results: list[FoldResult] = []

    for fold_number, (fit_indices, validation_indices) in enumerate(folds, start=1):
        started = perf_counter()
        try:
            preprocessor = preprocessor_factory(spec, feature_config)
            model = active_model_factory(model_name.value, params)
            pipeline = Pipeline(
                [("preprocessor", preprocessor), ("model", model)]
            )
            pipeline.fit(prepared.iloc[fit_indices], y[fit_indices])
            predictions = generate_predictions(
                pipeline, prepared.iloc[validation_indices]
            )
            evaluation = evaluate_predictions(
                y[validation_indices],
                predictions.labels,
                predictions.probabilities,
                model_name=model_name.value,
                split="cv_fold",
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Fold {fold_number} failed during cross-validation: {exc}"
            logger.warning(msg)
            raise OptimizationError(msg) from exc

        raw_value = evaluation.metrics.get(metric.value)
        if raw_value is None:
            msg = (
                f"Metric '{metric.value}' is unavailable in fold {fold_number}; "
                "cross-validation cannot optimize it."
            )
            raise OptimizationError(msg)
        fold_results.append(
            FoldResult(
                fold=fold_number,
                fit_indices=tuple(int(index) for index in fit_indices),
                validation_indices=tuple(
                    int(index) for index in validation_indices
                ),
                objective_value=_objective_value(raw_value, direction),
                raw_metric=float(raw_value),
                metrics=dict(evaluation.metrics),
                duration_seconds=round(perf_counter() - started, 6),
            )
        )

    objective_values = np.asarray(
        [fold.objective_value for fold in fold_results], dtype=float
    )
    raw_values = np.asarray([fold.raw_metric for fold in fold_results], dtype=float)
    mean_metrics: dict[str, float | None] = {}
    for key in fold_results[0].metrics:
        values = [
            fold.metrics[key]
            for fold in fold_results
            if fold.metrics.get(key) is not None
        ]
        mean_metrics[key] = (
            float(np.mean(np.asarray(values, dtype=float))) if values else None
        )

    logger.debug(
        "Cross-validation completed: folds=%s metric=%s mean=%.6f",
        cv_folds,
        metric.value,
        float(raw_values.mean()),
    )
    return CrossValidationResult(
        objective_mean=float(objective_values.mean()),
        objective_std=float(objective_values.std()),
        raw_metric=metric.value,
        raw_metric_mean=float(raw_values.mean()),
        mean_metrics=mean_metrics,
        folds=tuple(fold_results),
        n_samples=int(y.size),
    )

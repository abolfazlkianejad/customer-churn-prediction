"""Synthetic tests for leakage-safe cross-validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

from smartpredict.config.schema import (
    DatasetSpec,
    FeatureConfig,
    MetricName,
    OptimizationDirection,
    SupportedModel,
)
from smartpredict.optimization.cross_validation import cross_validate_params


class TrackingPreprocessor(BaseEstimator, TransformerMixin):
    """Synthetic transformer that records every fit and transform call."""

    def __init__(
        self,
        events: list[tuple[str, tuple[int, ...]]],
        numeric_columns: tuple[str, ...],
    ) -> None:
        """Store shared event log and numeric column names."""
        self.events = events
        self.numeric_columns = numeric_columns

    def fit(self, features: Any, target: Any = None) -> TrackingPreprocessor:
        """Record the exact rows used to fit this transformer."""
        self.events.append(("fit", tuple(int(index) for index in features.index)))
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Record transformed rows and return a simple numeric matrix."""
        self.events.append(
            ("transform", tuple(int(index) for index in features.index))
        )
        values = features[list(self.numeric_columns)].to_numpy(dtype=float)
        return np.nan_to_num(values, nan=0.0)


def _model_factory(name: str, params: Mapping[str, Any]) -> LogisticRegression:
    """Build a fast classifier for synthetic cross-validation tests."""
    return LogisticRegression(max_iter=200, random_state=42)


def _run_cv(
    train: Any,
    spec: DatasetSpec,
    features: FeatureConfig,
    direction: OptimizationDirection,
    events: list[tuple[str, tuple[int, ...]]],
    created: list[int],
) -> Any:
    """Run synthetic CV while capturing fold-local preprocessing events."""

    def preprocessor_factory(
        active_spec: DatasetSpec, active_features: FeatureConfig
    ) -> TrackingPreprocessor:
        """Create a fresh tracking transformer for each fold."""
        transformer = TrackingPreprocessor(events, tuple(active_spec.numeric_columns))
        created.append(id(transformer))
        return transformer

    return cross_validate_params(
        train,
        spec,
        features,
        SupportedModel.XGBOOST,
        {},
        cv_folds=3,
        metric=MetricName.ROC_AUC,
        direction=direction,
        seed=42,
        positive_label=spec.positive_label,
        preprocessor_factory=preprocessor_factory,
        model_factory=_model_factory,
    )


def test_cv_fits_preprocessing_only_on_fold_training_rows(
    prepared_training_data: dict[str, Any],
) -> None:
    """Each fold fits a fresh transformer on training rows only."""
    spec = prepared_training_data["spec"]
    train = prepared_training_data["split"].train.reset_index(drop=True)
    events: list[tuple[str, tuple[int, ...]]] = []
    created: list[int] = []

    result = _run_cv(
        train,
        spec,
        prepared_training_data["config"].features,
        OptimizationDirection.MAXIMIZE,
        events,
        created,
    )

    fit_events = [indices for kind, indices in events if kind == "fit"]
    transform_events = [indices for kind, indices in events if kind == "transform"]
    assert len(fit_events) == 3
    # Pipeline.fit_transform, predict, and predict_proba each transform rows.
    assert len(transform_events) == 9
    assert len(set(created)) == 3
    assert result.n_samples == len(train)
    assert result.objective_mean == result.raw_metric_mean

    for fold, fitted in zip(result.folds, fit_events, strict=True):
        assert set(fitted) == set(fold.fit_indices)
        assert not set(fitted).intersection(fold.validation_indices)
        assert len(fitted) + len(fold.validation_indices) == len(train)


def test_cv_normalizes_minimization_for_optuna(
    prepared_training_data: dict[str, Any],
) -> None:
    """Raw metric and maximization objective remain distinct for minimization."""
    spec = prepared_training_data["spec"]
    train = prepared_training_data["split"].train
    features = prepared_training_data["config"].features

    result = _run_cv(
        train,
        spec,
        features,
        OptimizationDirection.MINIMIZE,
        [],
        [],
    )

    assert result.raw_metric == "roc_auc"
    assert result.objective_mean == -result.raw_metric_mean
    assert all(
        fold.objective_value == -fold.raw_metric for fold in result.folds
    )

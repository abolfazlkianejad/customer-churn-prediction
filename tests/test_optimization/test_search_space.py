"""Synthetic tests for YAML-driven Optuna search spaces."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from smartpredict.config.schema import (
    SearchSpaceParameter,
    SearchSpaceType,
    SupportedModel,
    TrainingConfig,
)
from smartpredict.core.exceptions import OptimizationError
from smartpredict.optimization.search_space import (
    describe_search_space,
    merge_parameters,
    suggest_parameters,
    validate_search_space,
)


def test_configured_xgboost_space_is_valid(training_config: TrainingConfig) -> None:
    """The shipped XGBoost space contains only approved parameters."""
    space = training_config.optimization.search_space[SupportedModel.XGBOOST.value]
    validate_search_space(SupportedModel.XGBOOST, space)

    assert set(space) == {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    }
    assert describe_search_space(space)["learning_rate"]["log"] is True


def test_unknown_parameter_is_rejected(training_config: TrainingConfig) -> None:
    """Unsupported parameters never reach an Optuna trial."""
    space = dict(
        training_config.optimization.search_space[SupportedModel.XGBOOST.value]
    )
    space["objective"] = SearchSpaceParameter(
        type=SearchSpaceType.CATEGORICAL, choices=("binary:logistic",)
    )

    with pytest.raises(OptimizationError, match="Unsupported optimization parameters"):
        validate_search_space(SupportedModel.XGBOOST, space)


def test_suggest_parameters_uses_configured_distributions() -> None:
    """Suggestion types and bounds come from typed YAML definitions."""
    space = {
        "n_estimators": SearchSpaceParameter(
            type=SearchSpaceType.INTEGER, low=10, high=100, step=10
        ),
        "learning_rate": SearchSpaceParameter(
            type=SearchSpaceType.FLOAT, low=0.01, high=0.2, log=True
        ),
        "booster": SearchSpaceParameter(
            type=SearchSpaceType.CATEGORICAL, choices=("gbtree", "dart")
        ),
    }
    trial = Mock()
    trial.suggest_int.return_value = 50
    trial.suggest_float.return_value = 0.05
    trial.suggest_categorical.return_value = "gbtree"

    suggested = suggest_parameters(trial, space)

    assert suggested == {
        "booster": "gbtree",
        "learning_rate": 0.05,
        "n_estimators": 50,
    }
    trial.suggest_int.assert_called_once_with("n_estimators", 10, 100, step=10)
    trial.suggest_float.assert_called_once_with(
        "learning_rate", low=0.01, high=0.2, log=True
    )
    trial.suggest_categorical.assert_called_once_with("booster", ["gbtree", "dart"])


def test_merge_parameters_keeps_fixed_baseline_settings() -> None:
    """Only suggested parameters override the deployed baseline settings."""
    merged = merge_parameters(
        {"objective": "binary:logistic", "max_depth": 4},
        {"max_depth": 6, "learning_rate": 0.05},
    )

    assert merged == {
        "objective": "binary:logistic",
        "max_depth": 6,
        "learning_rate": 0.05,
    }

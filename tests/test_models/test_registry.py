"""Tests for the model registry and factory."""

from __future__ import annotations

from typing import Any

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from smartpredict.config.schema import SupportedModel
from smartpredict.core.exceptions import ModelError, ModelNotFoundError
from smartpredict.models.factory import create_candidate_model, create_model
from smartpredict.models.registry import ModelRegistry, get_default_registry


def test_registry_exposes_all_supported_models() -> None:
    """All four required classifiers are registered by default."""
    registry = ModelRegistry()
    assert registry.available_models() == tuple(
        sorted(model.value for model in SupportedModel)
    )


def test_registry_prevents_duplicate_registration() -> None:
    """Duplicate registrations require an explicit replacement."""
    registry = ModelRegistry()
    with pytest.raises(ModelError, match="already registered"):
        registry.register("random_forest", lambda params, seed: None)
    registry.register("random_forest", lambda params, seed: None, replace=True)


def test_registry_rejects_unknown_model() -> None:
    """Unknown model names fail before construction."""
    registry = ModelRegistry()
    with pytest.raises(ModelError, match="Unsupported model name"):
        registry.create("mystery_model")


def test_registry_missing_model_raises() -> None:
    """Removing a constructor produces a not-found error."""
    registry = ModelRegistry()
    registry._constructors.clear()
    with pytest.raises(ModelNotFoundError, match="not registered"):
        registry.create("logistic_regression")


def test_factory_builds_sklearn_models_with_seed() -> None:
    """Configured parameters and the experiment seed reach constructors."""
    registry = ModelRegistry()
    logistic = registry.create(
        "logistic_regression", {"max_iter": 50}, seed=7
    )
    tree = registry.create("decision_tree", {"max_depth": 3}, seed=7)
    forest = registry.create("random_forest", {"n_estimators": 5}, seed=7)

    assert isinstance(logistic, LogisticRegression)
    assert logistic.max_iter == 50
    assert logistic.random_state == 7
    assert isinstance(tree, DecisionTreeClassifier)
    assert tree.random_state == 7
    assert isinstance(forest, RandomForestClassifier)
    assert forest.n_estimators == 5
    assert forest.random_state == 7


def test_configured_random_state_overrides_experiment_seed() -> None:
    """An explicit YAML random_state is honored."""
    model = create_model(
        "random_forest", {"n_estimators": 2, "random_state": 99}, seed=7
    )
    assert model.random_state == 99


def test_factory_rejects_invalid_parameters() -> None:
    """Invalid hyperparameters are wrapped as domain errors."""
    with pytest.raises(ModelError, match="Invalid parameters"):
        create_model("decision_tree", {"not_a_parameter": 1})


def test_default_registry_is_process_wide() -> None:
    """The convenience accessor returns a stable registry instance."""
    assert get_default_registry() is get_default_registry()


def test_xgboost_missing_dependency_is_graceful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional XGBoost dependency becomes a domain error."""

    def missing_import(name: str) -> Any:
        """Simulate an environment without XGBoost."""
        if name == "xgboost":
            msg = "No module named 'xgboost'"
            raise ImportError(msg)
        msg = f"Unexpected import: {name}"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "smartpredict.models.registry.import_module", missing_import
    )
    with pytest.raises(ModelError, match="not installed"):
        create_model("xgboost")


def test_candidate_factory_passes_parameters() -> None:
    """Typed candidate configuration is accepted by the factory."""
    from smartpredict.config.schema import CandidateModelConfig

    candidate = CandidateModelConfig(params={"max_iter": 25})
    model = create_candidate_model("logistic_regression", candidate, seed=3)
    assert model.max_iter == 25
    assert model.random_state == 3


def test_xgboost_constructs_when_available() -> None:
    """XGBoost is constructed with configured hyperparameters when installed."""
    pytest.importorskip("xgboost")
    model = create_model("xgboost", {"n_estimators": 5, "max_depth": 2}, seed=11)
    assert model.get_params()["n_estimators"] == 5
    assert model.get_params()["random_state"] == 11

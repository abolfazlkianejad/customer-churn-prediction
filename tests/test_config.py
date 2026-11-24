"""Tests for settings and typed configuration objects."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smartpredict.config.schema import (
    DataConfig,
    ExplainabilityConfig,
    ExplainabilityMethod,
    ModelConfig,
    SplitConfig,
    TrainingConfig,
)
from smartpredict.config.settings import Environment, Settings, get_settings
from smartpredict.core.exceptions import ConfigurationError


def test_default_settings_are_valid() -> None:
    """Defaults match the Telco churn project contract."""
    settings = Settings(_env_file=None)
    assert settings.random_seed == 42
    assert settings.target_column == "Churn"
    assert settings.id_column == "customerID"
    assert settings.api_prefix == "/api/v1"


def test_environment_variables_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefixed environment variables take precedence over defaults."""
    monkeypatch.setenv("SMARTPREDICT_RANDOM_SEED", "123")
    monkeypatch.setenv("SMARTPREDICT_ENVIRONMENT", "testing")
    settings = Settings(_env_file=None)
    assert settings.random_seed == 123
    assert settings.environment is Environment.TESTING


def test_invalid_log_level_is_rejected() -> None:
    """Unknown logging levels fail validation."""
    with pytest.raises(ValueError, match="Invalid log level"):
        Settings(_env_file=None, log_level="verbose")


def test_api_prefix_is_normalized() -> None:
    """API prefixes always have one leading and no trailing slash."""
    settings = Settings(_env_file=None, api_prefix="api/v2/")
    assert settings.api_prefix == "/api/v2"


def test_settings_are_cached() -> None:
    """``get_settings`` returns one process-wide instance."""
    get_settings.cache_clear()
    first = get_settings()
    assert first is get_settings()
    get_settings.cache_clear()


def test_split_ratios_must_sum_to_one() -> None:
    """Invalid split ratios are rejected at construction time."""
    with pytest.raises(ConfigurationError, match="sum to 1.0"):
        SplitConfig(train=0.8, validation=0.1, test=0.2)


def test_model_config_rejects_empty_name() -> None:
    """A model cannot be registered without a name."""
    with pytest.raises(ConfigurationError, match="name"):
        ModelConfig(name="   ")


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_size": 0},
        {"local_examples": -1},
        {"max_background_samples": True},
        {"method": "unknown"},
    ],
)
def test_explainability_config_rejects_invalid_settings(
    overrides: dict[str, object],
) -> None:
    """Explainability settings fail loudly instead of producing empty reports."""
    with pytest.raises(ConfigurationError, match="explainability|Unsupported"):
        ExplainabilityConfig(**overrides)


def test_explainability_defaults_use_shap_tree() -> None:
    """Default explanations use SHAP tree explanations for the saved model."""
    config = ExplainabilityConfig()
    assert config.enabled is True
    assert config.method is ExplainabilityMethod.SHAP_TREE
    assert config.sample_size == 100


def test_training_config_rejects_invalid_cv_folds() -> None:
    """Cross-validation requires at least two folds."""
    with pytest.raises(ConfigurationError, match="cv_folds"):
        TrainingConfig(
            experiment_name="baseline",
            data=DataConfig(raw_path=Path("data/raw/telco_customer_churn.csv")),
            cv_folds=1,
        )


@pytest.mark.parametrize("filename", ["data.yaml", "train.yaml"])
def test_yaml_configs_are_valid(project_root: Path, filename: str) -> None:
    """Declarative configuration files exist and parse as mappings."""
    config_path = project_root / "configs" / filename
    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(content, dict)
    assert content


def test_data_yaml_describes_expected_telco_columns(project_root: Path) -> None:
    """The data config covers all 21 IBM Telco dataset columns."""
    config_path = project_root / "configs" / "data.yaml"
    content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    columns = content["expected_columns"]
    assert len(columns) == 21
    assert "customerID" in columns
    assert "TotalCharges" in columns
    assert "Churn" in columns

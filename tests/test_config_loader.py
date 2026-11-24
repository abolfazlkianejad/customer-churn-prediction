"""Tests for declarative YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import DatasetSpec
from smartpredict.core.exceptions import ConfigurationError


def _write_config(tmp_path: Path, content: dict[str, Any], name: str) -> Path:
    """Write a YAML mapping to a temporary file."""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_load_dataset_spec_from_repository_config(dataset_spec: DatasetSpec) -> None:
    """The real data.yaml produces a complete 21-column contract."""
    assert len(dataset_spec.expected_columns) == 21
    assert dataset_spec.target_column == "Churn"
    assert dataset_spec.id_column == "customerID"
    assert dataset_spec.split.train == 0.70
    assert dataset_spec.split.validation == 0.15
    assert dataset_spec.split.test == 0.15
    assert dataset_spec.raw_path.name == "telco_customer_churn.csv"
    assert dataset_spec.numeric_ranges["tenure"] == (0.0, 72.0)
    assert "Fiber optic" in dataset_spec.allowed_categories["InternetService"]


def test_load_training_config_from_repository_config() -> None:
    """Training configuration combines both YAML files."""
    config = load_training_config()
    assert config.experiment_name == "churn-xgboost-baseline"
    assert config.model.name == "xgboost"
    assert config.features.scale_numeric is True
    assert config.features.handle_unknown == "ignore"
    assert config.cv_folds == 5
    assert set(config.models) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
    assert all(candidate.enabled for candidate in config.models.values())
    assert config.evaluation.primary_metric.value == "roc_auc"
    assert config.evaluation.optimization_direction.value == "maximize"
    assert config.artifacts.best_model_file == "best_model.joblib"
    assert config.explainability.enabled is True
    assert config.explainability.method.value == "shap_tree"
    assert config.explainability.sample_size == 100
    assert config.explainability.local_examples == 3
    assert config.explainability.output_dir.name == "explainability"


def test_missing_config_file_raises(tmp_path: Path) -> None:
    """A missing YAML file is reported as a configuration error."""
    with pytest.raises(ConfigurationError, match="not found"):
        load_dataset_spec(tmp_path / "missing.yaml")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    """Invalid YAML is reported as a configuration error."""
    path = tmp_path / "broken.yaml"
    path.write_text("dataset: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Could not parse"):
        load_dataset_spec(path)


def test_inconsistent_columns_raise(tmp_path: Path, project_root: Path) -> None:
    """A schema that omits the target column cannot be loaded."""
    source = yaml.safe_load(
        (project_root / "configs" / "data.yaml").read_text(encoding="utf-8")
    )
    source["expected_columns"].remove("Churn")
    path = _write_config(tmp_path, source, "data.yaml")
    with pytest.raises(ConfigurationError, match="missing from expected_columns"):
        load_dataset_spec(path)


def test_unknown_split_strategy_raises(tmp_path: Path, project_root: Path) -> None:
    """Unsupported split strategies fail at load time."""
    source = yaml.safe_load(
        (project_root / "configs" / "data.yaml").read_text(encoding="utf-8")
    )
    source["split"]["strategy"] = "unknown"
    path = _write_config(tmp_path, source, "data.yaml")
    with pytest.raises(ConfigurationError, match="Unsupported split strategy"):
        load_dataset_spec(path)


def _train_yaml(project_root: Path) -> dict[str, Any]:
    """Load the repository training YAML as a mutable dictionary."""
    path = project_root / "configs" / "train.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_unknown_model_name_raises(tmp_path: Path, project_root: Path) -> None:
    """Unsupported model names fail configuration validation."""
    source = _train_yaml(project_root)
    source["models"] = {"mystery_model": {"enabled": True, "params": {}}}
    path = _write_config(tmp_path, source, "train.yaml")
    with pytest.raises(ConfigurationError, match="Unsupported model name"):
        load_training_config(train_config=path)


def test_all_models_disabled_raises(tmp_path: Path, project_root: Path) -> None:
    """A training run needs at least one enabled model."""
    source = _train_yaml(project_root)
    for candidate in source["models"].values():
        candidate["enabled"] = False
    path = _write_config(tmp_path, source, "train.yaml")
    with pytest.raises(ConfigurationError, match="must be enabled"):
        load_training_config(train_config=path)


def test_invalid_optimization_direction_raises(
    tmp_path: Path, project_root: Path
) -> None:
    """Only maximize and minimize are valid ranking directions."""
    source = _train_yaml(project_root)
    source["evaluation"]["optimization_direction"] = "sideways"
    path = _write_config(tmp_path, source, "train.yaml")
    with pytest.raises(ConfigurationError, match="OptimizationDirection"):
        load_training_config(train_config=path)


def test_unknown_primary_metric_raises(tmp_path: Path, project_root: Path) -> None:
    """Selection metrics are restricted to computed metrics."""
    source = _train_yaml(project_root)
    source["evaluation"]["primary_metric"] = "unknown_metric"
    path = _write_config(tmp_path, source, "train.yaml")
    with pytest.raises(ConfigurationError, match="Invalid evaluation"):
        load_training_config(train_config=path)


def test_legacy_single_model_config_still_loads(
    tmp_path: Path, project_root: Path
) -> None:
    """Phase 2 train.yaml files without a models section remain usable."""
    source = _train_yaml(project_root)
    source.pop("models")
    source["model"]["params"] = {"max_iter": 100}
    path = _write_config(tmp_path, source, "train.yaml")
    config = load_training_config(train_config=path)
    assert list(config.models) == ["xgboost"]
    assert config.models["xgboost"].params == {"max_iter": 100}


_AFFECTED_CATEGORIES = (
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
)


def test_repository_config_preserves_explicit_yes_no(
    dataset_spec: DatasetSpec,
) -> None:
    """Quoted Yes/No domains load as the strings present in the CSV."""
    for column in _AFFECTED_CATEGORIES:
        values = dataset_spec.allowed_categories[column]
        assert "Yes" in values
        assert "No" in values
        assert "True" not in values
        assert "False" not in values
    assert dataset_spec.allowed_categories["MultipleLines"] == (
        "Yes",
        "No",
        "No phone service",
    )
    assert dataset_spec.allowed_categories["OnlineSecurity"] == (
        "Yes",
        "No",
        "No internet service",
    )


def test_numeric_category_integers_remain_supported(
    dataset_spec: DatasetSpec,
) -> None:
    """SeniorCitizen integers are intentionally stringified."""
    assert dataset_spec.allowed_categories["SeniorCitizen"] == ("0", "1")


def _data_yaml_text(project_root: Path) -> str:
    """Return the repository data.yaml as raw text for mutation tests."""
    path = project_root / "configs" / "data.yaml"
    return path.read_text(encoding="utf-8")


def test_unquoted_boolean_like_values_raise(
    tmp_path: Path, project_root: Path
) -> None:
    """YAML booleans must fail loudly instead of becoming True/False."""
    text = _data_yaml_text(project_root).replace(
        'Partner: ["Yes", "No"]', "Partner: [Yes, No]"
    )
    assert "Partner: [Yes, No]" in text
    path = tmp_path / "data.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigurationError, match="YAML boolean"):
        load_dataset_spec(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("SeniorCitizen: [0, 1]", "SeniorCitizen: [0, 1.5]", "must be a string"),
        ("SeniorCitizen: [0, 1]", "SeniorCitizen: [0, null]", "must be a string"),
        (
            'Partner: ["Yes", "No"]',
            "Partner: {Yes: 1, No: 0}",
            "must be a list",
        ),
    ],
)
def test_invalid_configuration_types_raise(
    tmp_path: Path,
    project_root: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    """Unsupported YAML scalar and container types are rejected at load time."""
    text = _data_yaml_text(project_root)
    assert old in text
    path = tmp_path / "data.yaml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ConfigurationError, match=message):
        load_dataset_spec(path)

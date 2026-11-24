"""Shared pytest fixtures.

Synthetic records here exist only for tests; no fabricated dataset is written
into the project's data directory.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import DatasetSpec, TrainingConfig
from smartpredict.config.settings import Settings
from smartpredict.core.paths import PROJECT_ROOT
from smartpredict.core.seed import set_global_seed
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.splitter import stratified_split
from smartpredict.data.validation import validate_dataset
from smartpredict.features.preprocessor import fit_transform_features
from smartpredict.models.prediction import binary_targets


@pytest.fixture()
def settings() -> Settings:
    """Return default settings without loading a local ``.env`` file."""
    return Settings(_env_file=None)


@pytest.fixture()
def project_root() -> Path:
    """Return the absolute project root."""
    return PROJECT_ROOT


@pytest.fixture(autouse=True)
def fixed_seed() -> Iterator[None]:
    """Keep every test run deterministic."""
    set_global_seed(42)
    yield


@pytest.fixture()
def dataset_spec() -> DatasetSpec:
    """Load the real schema declared in ``configs/data.yaml``."""
    return load_dataset_spec()


@pytest.fixture()
def dataset_schema(dataset_spec: DatasetSpec) -> DatasetSchema:
    """Build the runtime schema used by validation."""
    return DatasetSchema.from_spec(dataset_spec)


@pytest.fixture()
def synthetic_frame(dataset_spec: DatasetSpec) -> pd.DataFrame:
    """Build an in-memory test frame matching the declared Telco schema."""
    rng = np.random.default_rng(42)
    row_count = 240
    data: dict[str, object] = {
        dataset_spec.id_column: [f"ID-{index:05d}" for index in range(row_count)],
        "SeniorCitizen": rng.integers(0, 2, row_count),
        "tenure": rng.integers(0, 73, row_count),
        "MonthlyCharges": np.round(rng.uniform(20, 100, row_count), 2),
    }
    data["TotalCharges"] = np.round(
        np.asarray(data["tenure"]) * np.asarray(data["MonthlyCharges"]) / 12, 2
    )
    for column in dataset_spec.categorical_columns:
        allowed = dataset_spec.allowed_categories[column]
        selected = rng.integers(0, len(allowed), row_count)
        data[column] = [allowed[index] for index in selected]
    churn = rng.random(row_count) < 0.35
    data[dataset_spec.target_column] = [
        dataset_spec.positive_label if flag else dataset_spec.negative_label
        for flag in churn
    ]

    frame = pd.DataFrame(data)[list(dataset_spec.expected_columns)]
    frame["TotalCharges"] = frame["TotalCharges"].astype(object)
    frame.loc[[3, 17], "TotalCharges"] = ""
    return frame


@pytest.fixture()
def synthetic_csv(tmp_path: Path, synthetic_frame: pd.DataFrame) -> Path:
    """Write the synthetic frame to a temporary CSV file."""
    path = tmp_path / "telco_synthetic.csv"
    synthetic_frame.to_csv(path, index=False)
    return path


@pytest.fixture()
def training_config() -> TrainingConfig:
    """Load the repository training configuration."""
    return load_training_config()


@pytest.fixture()
def prepared_training_data(
    synthetic_frame: pd.DataFrame,
    dataset_schema: DatasetSchema,
    dataset_spec: DatasetSpec,
    training_config: TrainingConfig,
) -> dict[str, Any]:
    """Return validated, split, and preprocessed synthetic data for model tests."""
    coerced, report = validate_dataset(synthetic_frame, dataset_schema)
    report.raise_if_invalid()
    split = stratified_split(coerced, dataset_spec)
    features, preprocessor = fit_transform_features(
        split.train,
        split.validation,
        split.test,
        dataset_spec,
        training_config.features,
    )
    return {
        "spec": dataset_spec,
        "config": training_config,
        "split": split,
        "features": features,
        "preprocessor": preprocessor,
        "y_train": binary_targets(
            split.train[dataset_spec.target_column], dataset_spec.positive_label
        ),
        "y_validation": binary_targets(
            split.validation[dataset_spec.target_column], dataset_spec.positive_label
        ),
        "y_test": binary_targets(
            split.test[dataset_spec.target_column], dataset_spec.positive_label
        ),
    }


@pytest.fixture()
def fast_training_configs(
    tmp_path: Path,
    project_root: Path,
    synthetic_csv: Path,
) -> dict[str, Path]:
    """Create fast, synthetic-only configs for full workflow tests."""
    data_source = yaml.safe_load(
        (project_root / "configs" / "data.yaml").read_text(encoding="utf-8")
    )
    data_source["dataset"]["raw_path"] = str(synthetic_csv)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(yaml.safe_dump(data_source), encoding="utf-8")

    train_source = yaml.safe_load(
        (project_root / "configs" / "train.yaml").read_text(encoding="utf-8")
    )
    train_source["models"] = {
        "logistic_regression": {"enabled": True, "params": {"max_iter": 100}},
        "decision_tree": {"enabled": True, "params": {"max_depth": 3}},
    }
    train_config = tmp_path / "train.yaml"
    train_config.write_text(yaml.safe_dump(train_source), encoding="utf-8")

    return {
        "data_config": data_config,
        "train_config": train_config,
        "models_dir": tmp_path / "artifacts" / "models",
        "reports_dir": tmp_path / "artifacts" / "reports",
        "plots_dir": tmp_path / "artifacts" / "plots",
    }

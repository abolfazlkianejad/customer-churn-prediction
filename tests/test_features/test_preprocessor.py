"""Tests for the reusable sklearn preprocessing pipeline."""

from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from smartpredict.config.loader import load_training_config
from smartpredict.config.schema import (
    DatasetSpec,
    FeatureConfig,
    ImputationStrategy,
)
from smartpredict.core.exceptions import FeatureEngineeringError
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.splitter import SplitResult, stratified_split
from smartpredict.data.validation import validate_dataset
from smartpredict.features.preprocessor import (
    build_preprocessor,
    fit_transform_features,
)


@dataclass(frozen=True)
class PreparedData:
    """Validated and split data plus the configured feature settings."""

    split: SplitResult
    spec: DatasetSpec
    config: FeatureConfig


@pytest.fixture()
def prepared_data(
    synthetic_frame: pd.DataFrame,
    dataset_schema: DatasetSchema,
    dataset_spec: DatasetSpec,
) -> PreparedData:
    """Validate and split the synthetic dataset for preprocessing tests."""
    coerced, report = validate_dataset(synthetic_frame, dataset_schema)
    report.raise_if_invalid()
    return PreparedData(
        split=stratified_split(coerced, dataset_spec),
        spec=dataset_spec,
        config=load_training_config().features,
    )


def test_build_preprocessor_is_unfitted(prepared_data: PreparedData) -> None:
    """The builder returns a reusable, unfitted transformer."""
    preprocessor = build_preprocessor(prepared_data.spec, prepared_data.config)
    assert {name for name, _, _ in preprocessor.transformers} == {
        "numeric",
        "categorical",
    }
    assert not hasattr(preprocessor, "transformers_")


def test_fit_transform_returns_aligned_arrays(prepared_data: PreparedData) -> None:
    """Fitted features align with their input splits and contain no NaNs."""
    features, _ = fit_transform_features(
        prepared_data.split.train,
        prepared_data.split.validation,
        prepared_data.split.test,
        prepared_data.spec,
        prepared_data.config,
    )
    assert features.train.shape[0] == prepared_data.split.train.shape[0]
    assert features.validation.shape[0] == prepared_data.split.validation.shape[0]
    assert features.test.shape[0] == prepared_data.split.test.shape[0]
    assert features.train.shape[1] == len(features.feature_names)
    assert not np.isnan(features.train).any()
    assert not np.isnan(features.validation).any()
    assert not np.isnan(features.test).any()


def test_one_hot_encoder_creates_named_columns(prepared_data: PreparedData) -> None:
    """Categorical values become one-hot columns."""
    features, _ = fit_transform_features(
        prepared_data.split.train,
        prepared_data.split.validation,
        prepared_data.split.test,
        prepared_data.spec,
        prepared_data.config,
    )
    assert any(name.startswith("gender_") for name in features.feature_names)
    assert "tenure" in features.feature_names


def test_imputation_statistics_come_from_training_only(
    prepared_data: PreparedData,
) -> None:
    """Validation values cannot influence fitted imputation statistics."""
    _, preprocessor = fit_transform_features(
        prepared_data.split.train,
        prepared_data.split.validation,
        prepared_data.split.test,
        prepared_data.spec,
        prepared_data.config,
    )
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    tenure_index = prepared_data.spec.numeric_columns.index("tenure")
    train_median = prepared_data.split.train["tenure"].median()
    assert imputer.statistics_[tenure_index] == pytest.approx(train_median)

    altered_validation = prepared_data.split.validation.copy()
    altered_validation["tenure"] = 1000
    assert np.median(altered_validation["tenure"]) != pytest.approx(train_median)


def test_unknown_categories_are_handled(prepared_data: PreparedData) -> None:
    """Unseen categories do not break validation-time transformation."""
    altered_validation = prepared_data.split.validation.copy()
    altered_validation["InternetService"] = "Satellite"
    features, _ = fit_transform_features(
        prepared_data.split.train,
        altered_validation,
        prepared_data.split.test,
        prepared_data.spec,
        prepared_data.config,
    )
    assert features.validation.shape[1] == features.train.shape[1]
    assert not np.isnan(features.validation).any()


def test_scaling_can_be_disabled(prepared_data: PreparedData) -> None:
    """The scaler step follows configuration."""
    config = FeatureConfig(scale_numeric=False)
    preprocessor = build_preprocessor(prepared_data.spec, config)
    numeric_pipeline = next(
        transformer
        for name, transformer, _ in preprocessor.transformers
        if name == "numeric"
    )
    assert "scaler" not in numeric_pipeline.named_steps

    features, _ = fit_transform_features(
        prepared_data.split.train,
        prepared_data.split.validation,
        prepared_data.split.test,
        prepared_data.spec,
        config,
    )
    assert not np.isnan(features.train).any()


def test_fitted_preprocessor_can_be_serialized(prepared_data: PreparedData) -> None:
    """A fitted preprocessor can be pickled for inference reuse."""
    features, preprocessor = fit_transform_features(
        prepared_data.split.train,
        prepared_data.split.validation,
        prepared_data.split.test,
        prepared_data.spec,
        prepared_data.config,
    )
    restored = pickle.loads(pickle.dumps(preprocessor))
    transformed = restored.transform(prepared_data.split.test)
    np.testing.assert_allclose(transformed, features.test)


def test_unsupported_imputation_raises(prepared_data: PreparedData) -> None:
    """Unsupported strategies fail explicitly."""
    config = FeatureConfig(numeric_imputation=ImputationStrategy.DROP)
    with pytest.raises(FeatureEngineeringError, match="Unsupported numeric"):
        build_preprocessor(prepared_data.spec, config)

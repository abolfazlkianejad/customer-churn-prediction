"""Reusable scikit-learn preprocessing built from declarative configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from smartpredict.config.schema import DatasetSpec, FeatureConfig
from smartpredict.core.exceptions import FeatureEngineeringError

_SUPPORTED_IMPUTERS = {"median", "most_frequent"}


@dataclass(frozen=True)
class FeatureSet:
    """Transformed arrays for all three splits plus the fitted feature names."""

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    feature_names: tuple[str, ...]


def build_preprocessor(
    spec: DatasetSpec, config: FeatureConfig
) -> ColumnTransformer:
    """Create an unfitted preprocessing pipeline.

    Args:
        spec: Dataset specification declaring numeric and categorical columns.
        config: Preprocessing choices loaded from ``train.yaml``.

    Returns:
        Unfitted column transformer, ready to be fitted on training rows only.

    Raises:
        FeatureEngineeringError: If the configuration requests an unsupported
            imputation or encoding strategy.

    """
    if config.numeric_imputation.value not in _SUPPORTED_IMPUTERS:
        msg = f"Unsupported numeric imputation: {config.numeric_imputation!r}"
        raise FeatureEngineeringError(msg)
    if config.categorical_imputation.value not in _SUPPORTED_IMPUTERS:
        msg = f"Unsupported categorical imputation: {config.categorical_imputation!r}"
        raise FeatureEngineeringError(msg)

    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config.numeric_imputation.value))
    ]
    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    if config.categorical_encoding.value == "one_hot":
        encoder: object = OneHotEncoder(
            handle_unknown=config.handle_unknown, sparse_output=False
        )
    elif config.categorical_encoding.value == "ordinal":
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:
        msg = f"Unsupported categorical encoding: {config.categorical_encoding!r}"
        raise FeatureEngineeringError(msg)

    categorical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config.categorical_imputation.value)),
        ("encoder", encoder),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), list(spec.numeric_columns)),
            (
                "categorical",
                Pipeline(categorical_steps),
                list(spec.categorical_columns),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features_frame(
    frame: pd.DataFrame, spec: DatasetSpec
) -> pd.DataFrame:
    """Convert declared columns into sklearn-compatible numpy dtypes.

    Args:
        frame: Coerced dataframe from the validation layer.
        spec: Dataset specification declaring column roles.

    Returns:
        Copy with numeric columns as float64 and categorical columns as object.

    """
    prepared = frame.copy()
    for column in spec.numeric_columns:
        if column not in prepared.columns:
            continue
        values = prepared[column].astype(object).where(
            prepared[column].notna(), np.nan
        )
        prepared[column] = pd.to_numeric(values, errors="coerce").astype("float64")
    for column in spec.categorical_columns:
        if column not in prepared.columns:
            continue
        values = prepared[column].astype(object)
        prepared[column] = values.where(values.notna(), np.nan)
    return prepared


def fit_transform_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    spec: DatasetSpec,
    config: FeatureConfig,
) -> tuple[FeatureSet, ColumnTransformer]:
    """Fit preprocessing on training rows only and transform all three splits.

    Args:
        train: Training dataframe from the splitter.
        validation: Validation dataframe from the splitter.
        test: Test dataframe from the splitter.
        spec: Dataset specification declaring column roles.
        config: Preprocessing choices.

    Returns:
        Transformed feature sets and the fitted preprocessor for reuse.

    Raises:
        FeatureEngineeringError: If fitting or transformation fails, or if the
            transformed matrices contain missing values.

    """
    preprocessor = build_preprocessor(spec, config)
    train_prepared = prepare_features_frame(train, spec)
    validation_prepared = prepare_features_frame(validation, spec)
    test_prepared = prepare_features_frame(test, spec)

    try:
        train_array = preprocessor.fit_transform(train_prepared)
        validation_array = preprocessor.transform(validation_prepared)
        test_array = preprocessor.transform(test_prepared)
    except (ValueError, TypeError, KeyError) as exc:
        msg = f"Feature preprocessing failed: {exc}"
        raise FeatureEngineeringError(msg) from exc

    arrays = (train_array, validation_array, test_array)
    if any(np.isnan(array).any() for array in arrays):
        msg = "Preprocessing produced missing values in the transformed features."
        raise FeatureEngineeringError(msg)

    return (
        FeatureSet(
            train=train_array,
            validation=validation_array,
            test=test_array,
            feature_names=tuple(
                str(name) for name in preprocessor.get_feature_names_out()
            ),
        ),
        preprocessor,
    )

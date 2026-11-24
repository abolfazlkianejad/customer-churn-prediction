"""Regression tests for artifact-based inference on raw records."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import CandidateModelConfig
from smartpredict.core.exceptions import PredictionError
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.inference import predict_from_raw
from smartpredict.models.training import TrainingService


def _fitted_bundle(
    prepared_training_data: dict[str, Any], tmp_path: Any
) -> tuple[Any, Any]:
    """Train a small model, save it, and reload both artifacts."""
    spec = prepared_training_data["spec"]
    split = prepared_training_data["split"]
    trained = TrainingService().train(
        "logistic_regression",
        CandidateModelConfig(params={"max_iter": 100}),
        prepared_training_data["features"],
        split.train[spec.target_column],
        split.validation[spec.target_column],
        seed=42,
        positive_label=spec.positive_label,
    )
    model_path = tmp_path / "best_model.joblib"
    preprocessor_path = tmp_path / "preprocessing_pipeline.joblib"
    joblib.dump(trained.model, model_path)
    joblib.dump(prepared_training_data["preprocessor"], preprocessor_path)
    return joblib.load(model_path), joblib.load(preprocessor_path)


def test_predict_from_raw_returns_valid_shapes(
    prepared_training_data: dict[str, Any],
    synthetic_frame: pd.DataFrame,
    tmp_path: Any,
) -> None:
    """Saved artifacts predict raw rows with valid label and probability arrays."""
    spec = prepared_training_data["spec"]
    model, preprocessor = _fitted_bundle(prepared_training_data, tmp_path)
    raw = synthetic_frame[list(spec.expected_columns)].head(5)

    result = predict_from_raw(raw, spec, preprocessor, model)

    assert result.n_records == 5
    assert result.labels.shape == (5,)
    assert set(np.unique(result.labels)).issubset({0, 1})
    assert result.probabilities is not None
    assert result.probabilities.shape == (5,)
    assert np.isfinite(result.probabilities).all()
    assert ((result.probabilities >= 0) & (result.probabilities <= 1)).all()
    assert spec.id_column not in result.feature_names
    assert not any(name.startswith(spec.id_column) for name in result.feature_names)


def test_inference_never_refits_transformers_or_model(
    prepared_training_data: dict[str, Any],
    synthetic_frame: pd.DataFrame,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction only transforms data; no estimator is refitted."""
    spec = prepared_training_data["spec"]
    model, preprocessor = _fitted_bundle(prepared_training_data, tmp_path)
    numeric = preprocessor.named_transformers_["numeric"]
    imputer = numeric.named_steps["imputer"]
    scaler = numeric.named_steps["scaler"]
    encoder = preprocessor.named_transformers_["categorical"].named_steps["encoder"]

    original_statistics = imputer.statistics_.copy()
    original_mean = scaler.mean_.copy()
    original_categories = [np.asarray(c).copy() for c in encoder.categories_]
    original_coef = model.coef_.copy()

    def fail_fit(*args: object, **kwargs: object) -> None:
        """Fail immediately if inference attempts to refit an estimator."""
        raise AssertionError("Inference must not call fit().")

    monkeypatch.setattr(SimpleImputer, "fit", fail_fit)
    monkeypatch.setattr(StandardScaler, "fit", fail_fit)
    monkeypatch.setattr(OneHotEncoder, "fit", fail_fit)
    monkeypatch.setattr(LogisticRegression, "fit", fail_fit)

    result = predict_from_raw(
        synthetic_frame[list(spec.expected_columns)].head(5),
        spec,
        preprocessor,
        model,
    )

    assert result.labels.shape == (5,)
    np.testing.assert_allclose(imputer.statistics_, original_statistics)
    np.testing.assert_allclose(scaler.mean_, original_mean)
    for actual, expected in zip(encoder.categories_, original_categories, strict=True):
        np.testing.assert_array_equal(np.asarray(actual), expected)
    np.testing.assert_allclose(model.coef_, original_coef)


def test_inference_rejects_invalid_categories(
    prepared_training_data: dict[str, Any],
    synthetic_frame: pd.DataFrame,
    tmp_path: Any,
) -> None:
    """Invalid raw domains fail prediction instead of being silently accepted."""
    spec = prepared_training_data["spec"]
    model, preprocessor = _fitted_bundle(prepared_training_data, tmp_path)
    raw = synthetic_frame[list(spec.expected_columns)].head(2).copy()
    raw.loc[raw.index[0], "InternetService"] = "Satellite"

    with pytest.raises(PredictionError, match="failed validation"):
        predict_from_raw(raw, spec, preprocessor, model)


@pytest.mark.parametrize("drop_target", [False, True])
def test_inference_accepts_single_class_or_missing_target(
    prepared_training_data: dict[str, Any],
    synthetic_frame: pd.DataFrame,
    tmp_path: Any,
    drop_target: bool,
) -> None:
    """Prediction batches need feature validity, not a balanced target."""
    spec = prepared_training_data["spec"]
    model, preprocessor = _fitted_bundle(prepared_training_data, tmp_path)
    raw = synthetic_frame[list(spec.expected_columns)].head(3).copy()
    raw[spec.target_column] = spec.negative_label
    if drop_target:
        raw = raw.drop(columns=[spec.target_column])

    result = predict_from_raw(raw, spec, preprocessor, model)

    assert result.labels.shape == (3,)
    assert result.feature_names == tuple(preprocessor.get_feature_names_out())


def test_real_artifacts_inference_when_available() -> None:
    """If a real training run has produced artifacts, they remain loadable."""
    spec = load_dataset_spec()
    config = load_training_config()
    paths = build_artifact_paths(config.artifacts)
    if not paths["best_model"].exists() or not paths["preprocessor"].exists():
        pytest.skip("Real training artifacts have not been generated.")

    model = joblib.load(paths["best_model"])
    preprocessor = joblib.load(paths["preprocessor"])
    raw = pd.read_csv(spec.raw_path, dtype=str, keep_default_na=False).head(3)

    result = predict_from_raw(raw, spec, preprocessor, model)

    assert result.n_records == 3
    assert result.labels.shape == (3,)
    assert result.probabilities is not None
    assert result.probabilities.shape == (3,)
    assert spec.id_column not in result.feature_names

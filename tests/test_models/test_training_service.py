"""Tests for the reusable training service."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from smartpredict.config.schema import CandidateModelConfig
from smartpredict.core.exceptions import ModelTrainingError
from smartpredict.models.training import TrainingService


def test_training_service_records_metadata_and_validation_predictions(
    prepared_training_data: dict[str, Any],
) -> None:
    """Training returns fitted model metadata and validation predictions."""
    split = prepared_training_data["split"]
    result = TrainingService().train(
        "logistic_regression",
        CandidateModelConfig(params={"max_iter": 100}),
        prepared_training_data["features"],
        split.train["Churn"],
        split.validation["Churn"],
        seed=42,
        positive_label="Yes",
    )

    assert result.name == "logistic_regression"
    assert result.validation_predictions.labels.shape == (
        split.validation.shape[0],
    )
    assert result.validation_predictions.probabilities is not None
    assert result.metadata["n_train_rows"] == split.train.shape[0]
    assert result.metadata["n_validation_rows"] == split.validation.shape[0]
    assert result.metadata["random_seed"] == 42
    assert result.metadata["probability_available"] is True
    assert result.to_dict()["name"] == "logistic_regression"


def test_training_service_never_reads_test_features(
    prepared_training_data: dict[str, Any],
) -> None:
    """The training service only accesses train and validation matrices."""

    class TrainOnlyFeatures:
        """Feature container that fails if the test split is accessed."""

        def __init__(self, features: Any) -> None:
            """Wrap a prepared feature set."""
            self.train = features.train
            self.validation = features.validation

        @property
        def test(self) -> np.ndarray:
            """Fail because test data must not be visible during training."""
            msg = "Test features were accessed during training."
            raise AssertionError(msg)

    split = prepared_training_data["split"]
    result = TrainingService().train(
        "decision_tree",
        CandidateModelConfig(params={"max_depth": 3}),
        TrainOnlyFeatures(prepared_training_data["features"]),
        split.train["Churn"],
        split.validation["Churn"],
        seed=42,
        positive_label="Yes",
    )
    assert result.metadata["n_features"] == prepared_training_data[
        "features"
    ].train.shape[1]


def test_training_service_wraps_model_errors(
    prepared_training_data: dict[str, Any],
) -> None:
    """Invalid model configuration becomes a ModelTrainingError."""
    split = prepared_training_data["split"]
    with pytest.raises(ModelTrainingError, match="Training failed"):
        TrainingService().train(
            "decision_tree",
            CandidateModelConfig(params={"unknown_option": 1}),
            prepared_training_data["features"],
            split.train["Churn"],
            split.validation["Churn"],
            seed=42,
            positive_label="Yes",
        )

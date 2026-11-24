"""Reusable training service for configured classifiers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from sklearn.base import ClassifierMixin

from smartpredict.config.schema import CandidateModelConfig
from smartpredict.core.exceptions import ModelTrainingError
from smartpredict.core.logging import get_logger
from smartpredict.features.preprocessor import FeatureSet
from smartpredict.models.factory import create_candidate_model
from smartpredict.models.prediction import (
    ModelPredictions,
    binary_targets,
    generate_predictions,
)
from smartpredict.models.registry import ModelRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """One fitted model, its validation predictions, and audit metadata."""

    name: str
    model: ClassifierMixin
    validation_predictions: ModelPredictions
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view without the fitted model object."""
        return {
            "name": self.name,
            **self.metadata,
        }


class TrainingService:
    """Train classifiers on an already-preprocessed training matrix."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        """Initialize the service with an optional model registry."""
        self.registry = registry

    def train(
        self,
        name: str,
        candidate: CandidateModelConfig,
        features: FeatureSet,
        train_target: object,
        validation_target: object,
        *,
        seed: int,
        positive_label: str,
    ) -> TrainingResult:
        """Fit one candidate and generate validation predictions.

        The preprocessing represented by ``features`` was fitted before this
        method is called and is never refitted here. Test features are not
        accessed.

        Args:
            name: Canonical model name.
            candidate: Enabled candidate configuration.
            features: Already-preprocessed train/validation/test arrays.
            train_target: Raw training target labels.
            validation_target: Raw validation target labels.
            seed: Experiment seed.
            positive_label: Configured positive-class label.

        Returns:
            Fitted model with validation predictions and training metadata.

        Raises:
            ModelTrainingError: If fitting or validation prediction fails.

        """
        logger.info("Training model '%s'", name)
        y_train = binary_targets(train_target, positive_label)
        y_validation = binary_targets(validation_target, positive_label)

        try:
            model = create_candidate_model(
                name, candidate, seed=seed, registry=self.registry
            )
            started_at = datetime.now(tz=UTC).isoformat()
            started = perf_counter()
            model.fit(features.train, y_train)
            duration_seconds = perf_counter() - started
            validation_predictions = generate_predictions(
                model, features.validation
            )
        except Exception as exc:
            msg = f"Training failed for model '{name}': {exc}"
            logger.exception(msg)
            raise ModelTrainingError(msg) from exc

        metadata: dict[str, Any] = {
            "model_name": name,
            "model_type": type(model).__name__,
            "parameters": dict(candidate.params),
            "random_seed": seed,
            "started_at": started_at,
            "duration_seconds": round(duration_seconds, 6),
            "n_train_rows": int(features.train.shape[0]),
            "n_validation_rows": int(features.validation.shape[0]),
            "n_features": int(features.train.shape[1]),
            "n_positive_train": int(y_train.sum()),
            "n_negative_train": int(y_train.size - y_train.sum()),
            "n_positive_validation": int(y_validation.sum()),
            "n_negative_validation": int(y_validation.size - y_validation.sum()),
            "probability_available": validation_predictions.probabilities is not None,
        }
        logger.info(
            "Model '%s' trained in %.3fs on %s rows",
            name,
            duration_seconds,
            metadata["n_train_rows"],
        )
        return TrainingResult(
            name=name,
            model=model,
            validation_predictions=validation_predictions,
            metadata=metadata,
        )

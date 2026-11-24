"""Exception hierarchy for SmartPredict."""

from __future__ import annotations


class SmartPredictError(Exception):
    """Base class for all project-specific exceptions."""


class ConfigurationError(SmartPredictError):
    """Raised when configuration is missing, invalid, or inconsistent."""


class DataError(SmartPredictError):
    """Raised when data cannot be loaded or processed."""


class DataValidationError(DataError):
    """Raised when data fails schema or quality checks."""


class FeatureEngineeringError(SmartPredictError):
    """Raised when feature construction or transformation fails."""


class ModelError(SmartPredictError):
    """Base class for model-related failures."""


class ModelNotFoundError(ModelError):
    """Raised when a requested model version is not available."""


class ModelTrainingError(ModelError):
    """Raised when model fitting fails."""


class ArtifactError(SmartPredictError):
    """Raised when a model artifact cannot be saved, loaded, or verified."""


class EvaluationError(SmartPredictError):
    """Raised when metrics or model comparisons cannot be computed."""


class PredictionError(SmartPredictError):
    """Raised when inference cannot be completed."""


class ExplainabilityError(SmartPredictError):
    """Raised when model explanations cannot be generated or persisted."""


class OptimizationError(SmartPredictError):
    """Raised when optimization studies or threshold analysis fail."""

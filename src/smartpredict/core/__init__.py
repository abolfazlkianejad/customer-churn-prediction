"""Cross-cutting infrastructure shared by every layer."""

from smartpredict.core.exceptions import (
    ArtifactError,
    ConfigurationError,
    DataError,
    DataValidationError,
    FeatureEngineeringError,
    ModelError,
    ModelNotFoundError,
    ModelTrainingError,
    PredictionError,
    SmartPredictError,
)
from smartpredict.core.logging import configure_logging, get_logger
from smartpredict.core.paths import (
    ARTIFACTS_DIR,
    PROJECT_ROOT,
    get_project_root,
    resolve_path,
)
from smartpredict.core.seed import set_global_seed

__all__ = [
    "ARTIFACTS_DIR",
    "ArtifactError",
    "ConfigurationError",
    "DataError",
    "DataValidationError",
    "FeatureEngineeringError",
    "ModelError",
    "ModelNotFoundError",
    "ModelTrainingError",
    "PredictionError",
    "PROJECT_ROOT",
    "SmartPredictError",
    "configure_logging",
    "get_logger",
    "get_project_root",
    "resolve_path",
    "set_global_seed",
]

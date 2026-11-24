"""FastAPI serving layer."""

from smartpredict.api.app import app, create_app
from smartpredict.api.dependencies import ModelArtifacts, load_model_artifacts

__all__ = ["ModelArtifacts", "app", "create_app", "load_model_artifacts"]

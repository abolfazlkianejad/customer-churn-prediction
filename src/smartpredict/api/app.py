"""FastAPI application factory for the SmartPredict inference service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from smartpredict.api.dependencies import (
    ModelArtifacts,
    _cached_artifacts,
    get_artifacts,
)
from smartpredict.api.routes import router
from smartpredict.config.settings import Settings, get_settings
from smartpredict.core.exceptions import (
    ArtifactError,
    ConfigurationError,
    DataValidationError,
    ExplainabilityError,
    PredictionError,
    SmartPredictError,
)
from smartpredict.core.logging import get_logger

logger = get_logger(__name__)


def _error(status_code: int, detail: object) -> JSONResponse:
    """Build the uniform JSON error response."""
    return JSONResponse(status_code=status_code, content={"detail": detail})


def _register_error_handlers(app: FastAPI) -> None:
    """Map project exceptions to deterministic HTTP status codes."""

    def artifact_error(request: Request, exc: Exception) -> JSONResponse:
        """Report missing or unreadable artifacts as unavailable."""
        logger.warning("Artifact failure at %s: %s", request.url.path, exc)
        return _error(503, str(exc))

    def bad_request(request: Request, exc: Exception) -> JSONResponse:
        """Report invalid requests and runtime prediction failures."""
        logger.info("Request rejected at %s: %s", request.url.path, exc)
        return _error(400, str(exc))

    def server_error(request: Request, exc: Exception) -> JSONResponse:
        """Report unexpected project errors without exposing internals."""
        logger.error(
            "Unhandled application error at %s: %s: %s",
            request.url.path,
            type(exc).__name__,
            exc,
        )
        return _error(500, "Internal application error.")

    app.add_exception_handler(ArtifactError, artifact_error)
    for error_type in (
        PredictionError,
        DataValidationError,
        ConfigurationError,
        ExplainabilityError,
    ):
        app.add_exception_handler(error_type, bad_request)
    app.add_exception_handler(SmartPredictError, server_error)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the API application without loading artifacts at import time."""
    effective = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Warm the saved artifact cache and log lifecycle transitions."""
        logger.info(
            "SmartPredict API starting (environment=%s, prefix=%s)",
            effective.environment.value,
            effective.api_prefix,
        )
        try:
            artifacts = _cached_artifacts(str(effective.models_dir))
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "SmartPredict API startup failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            raise
        app.state.artifacts = artifacts
        logger.info(
            "SmartPredict API ready (model=%s, features=%d, threshold=%s)",
            artifacts.selected_model,
            len(artifacts.feature_names),
            artifacts.threshold,
        )
        try:
            yield
        finally:
            logger.info("SmartPredict API shutdown complete")

    app = FastAPI(
        title="SmartPredict Customer Churn API",
        description=(
            "Inference API backed by saved XGBoost and preprocessing artifacts."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = effective

    def artifacts_dependency() -> ModelArtifacts:
        """Load artifacts from the settings supplied to this application."""
        return _cached_artifacts(str(effective.models_dir))

    app.dependency_overrides[get_artifacts] = artifacts_dependency
    prefix = "" if effective.api_prefix == "/" else effective.api_prefix
    app.include_router(router, prefix=prefix)
    _register_error_handlers(app)
    return app


app = create_app()

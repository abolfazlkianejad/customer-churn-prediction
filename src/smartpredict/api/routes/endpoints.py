"""Health, prediction, and model-information endpoints."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from smartpredict.api.dependencies import ModelArtifacts, get_artifacts
from smartpredict.api.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from smartpredict.models.inference import InferenceResult, predict_from_raw

router = APIRouter()

_PATH_KEY_PARTS = ("path", "dir", "root", "file")
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_PATH_SEPARATORS = re.compile(r"[\\/]+")


def _redact_path(value: str) -> str:
    """Return a basename for values that look like filesystem paths."""
    if (
        _WINDOWS_PATH.match(value)
        or value.startswith(("/", "\\"))
        or "\\" in value
    ):
        parts = [part for part in _PATH_SEPARATORS.split(value.rstrip("\\/")) if part]
        return parts[-1] if parts else "[redacted]"
    return value


def _sanitize_metadata(value: Any, *, key: str = "") -> Any:
    """Remove internal filesystem locations from metadata responses."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for child_key, child_value in value.items():
            normalized_key = str(child_key)
            if any(part in normalized_key.lower() for part in _PATH_KEY_PARTS):
                if isinstance(child_value, str):
                    result[child_key] = _redact_path(child_value)
                else:
                    result[child_key] = _sanitize_metadata(
                        child_value, key=normalized_key
                    )
            else:
                result[child_key] = _sanitize_metadata(
                    child_value, key=normalized_key
                )
        return result
    if isinstance(value, list):
        return [_sanitize_metadata(item, key=key) for item in value]
    if isinstance(value, str):
        return _redact_path(value)
    return value


def _response(
    result: InferenceResult, customer_ids: list[str]
) -> PredictionResponse:
    """Convert an inference result into the JSON response model."""
    return PredictionResponse(
        labels=[int(label) for label in result.labels],
        probabilities=(
            [float(value) for value in result.probabilities]
            if result.probabilities is not None
            else None
        ),
        feature_names=list(result.feature_names),
        customer_ids=customer_ids,
        n_records=result.n_records,
        threshold=result.threshold,
    )


@router.get("/health", response_model=HealthResponse)
def health(
    artifacts: Annotated[ModelArtifacts, Depends(get_artifacts)],
) -> HealthResponse:
    """Report whether saved artifacts are available for prediction."""
    return HealthResponse(
        status="ok",
        artifacts_ready=True,
        model_loaded=artifacts.model is not None,
        preprocessor_loaded=artifacts.preprocessor is not None,
        model_name=artifacts.selected_model,
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(
    record: dict[str, Any],
    artifacts: Annotated[ModelArtifacts, Depends(get_artifacts)],
) -> PredictionResponse:
    """Predict churn for one raw customer record."""
    result = predict_from_raw(
        [record],
        artifacts.spec,
        artifacts.preprocessor,
        artifacts.model,
        artifacts.threshold,
    )
    customer_id = str(record.get(artifacts.spec.id_column, ""))
    return _response(result, [customer_id])


@router.post("/predict/batch", response_model=PredictionResponse)
def predict_batch(
    payload: BatchPredictionRequest,
    artifacts: Annotated[ModelArtifacts, Depends(get_artifacts)],
) -> PredictionResponse:
    """Predict churn for several raw customer records in one request."""
    result = predict_from_raw(
        payload.records,
        artifacts.spec,
        artifacts.preprocessor,
        artifacts.model,
        artifacts.threshold,
    )
    customer_ids = [
        str(record.get(artifacts.spec.id_column, "")) for record in payload.records
    ]
    return _response(result, customer_ids)


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(
    artifacts: Annotated[ModelArtifacts, Depends(get_artifacts)],
) -> ModelInfoResponse:
    """Describe the loaded model and its transformed feature contract."""
    return ModelInfoResponse(
        selected_model=artifacts.selected_model,
        feature_count=len(artifacts.feature_names),
        feature_names=list(artifacts.feature_names),
        decision_threshold=artifacts.threshold,
        metadata=_sanitize_metadata(artifacts.metadata),
    )


@router.get("/status", include_in_schema=False)
def service_status() -> dict[str, str]:
    """Return a lightweight timestamp without loading model artifacts."""
    return {"service": "smartpredict", "checked_at": datetime.now(tz=UTC).isoformat()}

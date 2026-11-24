"""Request and response contracts for the prediction API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BatchPredictionRequest(BaseModel):
    """A batch of raw customer records requiring predictions."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Prediction output shared by single and batch endpoints."""

    labels: list[int]
    probabilities: list[float] | None = None
    feature_names: list[str]
    customer_ids: list[str] = Field(default_factory=list)
    n_records: int
    threshold: float = 0.5


class HealthResponse(BaseModel):
    """Readiness information for the serving process."""

    status: Literal["ok"]
    artifacts_ready: bool
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: str


class ModelInfoResponse(BaseModel):
    """Metadata for the currently loaded model bundle."""

    selected_model: str
    feature_count: int
    feature_names: list[str]
    decision_threshold: float = 0.5
    metadata: dict[str, Any]


class ErrorResponse(BaseModel):
    """Uniform error payload for domain failures."""

    detail: str

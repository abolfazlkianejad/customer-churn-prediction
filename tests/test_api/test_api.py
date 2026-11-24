"""HTTP contract tests for the SmartPredict inference API."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from smartpredict.api.app import create_app
from smartpredict.api.dependencies import ModelArtifacts, get_artifacts
from smartpredict.config.schema import CandidateModelConfig
from smartpredict.config.settings import Settings
from smartpredict.core.exceptions import ArtifactError
from smartpredict.features.preprocessor import prepare_features_frame
from smartpredict.models.training import TrainingService


def _artifacts(prepared_training_data: dict[str, Any]) -> ModelArtifacts:
    """Build in-memory artifacts from synthetic training data."""
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
    preprocessor = prepared_training_data["preprocessor"]
    feature_names = tuple(
        str(name) for name in preprocessor.get_feature_names_out()
    )
    return ModelArtifacts(
        spec=spec,
        preprocessor=preprocessor,
        model=trained.model,
        metadata={"model": {"selected": "xgboost", "parameters": {}}},
        feature_names=feature_names,
        selected_model="xgboost",
    )


def _raw_records(frame: pd.DataFrame, count: int) -> list[dict[str, Any]]:
    """Convert synthetic rows into JSON-native request dictionaries."""
    return json.loads(frame.head(count).to_json(orient="records"))


@pytest.fixture()
def client(
    prepared_training_data: dict[str, Any], settings: Settings
) -> Iterator[TestClient]:
    """Create an API client backed by synthetic artifacts."""
    artifacts = _artifacts(prepared_training_data)
    app = create_app(settings=settings)
    app.dependency_overrides[get_artifacts] = lambda: artifacts
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_health_reports_loaded_artifacts(client: TestClient) -> None:
    """Health checks report readiness without triggering a prediction."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["artifacts_ready"] is True
    assert payload["model_name"] == "xgboost"


def test_single_prediction_returns_valid_output(
    client: TestClient, prepared_training_data: dict[str, Any]
) -> None:
    """A raw customer record produces one label and one probability."""
    record = _raw_records(prepared_training_data["split"].validation, 1)[0]
    response = client.post("/api/v1/predict", json=record)

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_records"] == 1
    assert len(payload["labels"]) == 1
    assert payload["labels"][0] in (0, 1)
    assert payload["probabilities"] is not None
    assert len(payload["probabilities"]) == 1
    assert 0.0 <= payload["probabilities"][0] <= 1.0
    assert len(payload["feature_names"]) > 0
    assert not any(
        name.startswith("customerID") for name in payload["feature_names"]
    )


def test_batch_prediction_preserves_record_order(
    client: TestClient, prepared_training_data: dict[str, Any]
) -> None:
    """Batch requests return one prediction per input row."""
    records = _raw_records(prepared_training_data["split"].validation, 3)
    response = client.post("/api/v1/predict/batch", json={"records": records})

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_records"] == 3
    assert len(payload["labels"]) == 3
    assert payload["customer_ids"] == [record["customerID"] for record in records]


def test_invalid_feature_domain_returns_bad_request(
    client: TestClient, prepared_training_data: dict[str, Any]
) -> None:
    """Invalid raw values are rejected by prediction validation."""
    record = _raw_records(prepared_training_data["split"].validation, 1)[0]
    record["InternetService"] = "Satellite"

    response = client.post("/api/v1/predict", json=record)

    assert response.status_code == 400
    assert "failed validation" in response.json()["detail"]


def test_prediction_uses_saved_non_default_threshold(
    prepared_training_data: dict[str, Any], settings: Settings
) -> None:
    """A saved threshold changes labels while probabilities remain unchanged."""
    spec = prepared_training_data["spec"]
    artifacts = _artifacts(prepared_training_data)
    validation = prepared_training_data["split"].validation
    features = artifacts.preprocessor.transform(
        prepare_features_frame(validation, spec)
    )
    probabilities = np.asarray(artifacts.model.predict_proba(features)[:, 1])
    unique_scores = np.unique(probabilities)
    candidates = (unique_scores[:-1] + unique_scores[1:]) / 2

    chosen: float | None = None
    flipped_index: int | None = None
    for candidate in candidates:
        flips = np.flatnonzero((probabilities >= 0.5) != (probabilities >= candidate))
        if flips.size > 0:
            chosen = float(candidate)
            flipped_index = int(flips[0])
            break
    assert chosen is not None and flipped_index is not None

    artifacts = replace(artifacts, threshold=chosen)
    app = create_app(settings=settings)
    app.dependency_overrides[get_artifacts] = lambda: artifacts
    record = _raw_records(validation.iloc[[flipped_index]], 1)[0]
    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.post("/api/v1/predict", json=record)

    assert response.status_code == 200
    payload = response.json()
    expected = int(probabilities[flipped_index] >= chosen)
    assert payload["threshold"] == pytest.approx(chosen)
    assert payload["labels"][0] == expected
    assert payload["labels"][0] != int(probabilities[flipped_index] >= 0.5)
    assert payload["probabilities"][0] == pytest.approx(probabilities[flipped_index])


def test_model_info_returns_metadata_and_features(client: TestClient) -> None:
    """Model info exposes the loaded artifact contract."""
    response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] == "xgboost"
    assert payload["feature_count"] == len(payload["feature_names"])
    assert payload["metadata"]["model"]["selected"] == "xgboost"


def test_missing_artifacts_return_service_unavailable(settings: Settings) -> None:
    """An unavailable model bundle is reported distinctly from bad input."""

    def missing_artifacts() -> ModelArtifacts:
        """Simulate a missing saved artifact during dependency resolution."""
        msg = "model bundle missing"
        raise ArtifactError(msg)

    app = create_app(settings=settings)
    app.dependency_overrides[get_artifacts] = missing_artifacts
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "model bundle missing"


def test_api_prefix_is_configurable(
    prepared_training_data: dict[str, Any]
) -> None:
    """Routes are mounted below the configured API prefix."""
    artifacts = _artifacts(prepared_training_data)
    app = create_app(settings=Settings(_env_file=None, api_prefix="/v2"))
    app.dependency_overrides[get_artifacts] = lambda: artifacts
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/v2/health").status_code == 200
        assert client.get("/api/v1/health").status_code == 404

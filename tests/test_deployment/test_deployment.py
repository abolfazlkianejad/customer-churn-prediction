"""Deployment contract tests using synthetic fixtures and mocks only."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from smartpredict.api.app import create_app
from smartpredict.api.dependencies import (
    ModelArtifacts,
    get_artifacts,
    load_model_artifacts,
)
from smartpredict.config.settings import Settings
from smartpredict.core.exceptions import ArtifactError
from smartpredict.core.paths import CONFIGS_DIR


def _raw_records(frame: Any, count: int) -> list[dict[str, Any]]:
    """Convert synthetic dataframe rows into JSON-native dictionaries."""
    return json.loads(frame.head(count).to_json(orient="records"))


def _client(settings: Settings) -> TestClient:
    """Create a test client that executes application lifespan hooks."""
    return TestClient(create_app(settings=settings), raise_server_exceptions=False)


def test_application_imports() -> None:
    """The FastAPI application module imports without side effects."""
    module = importlib.import_module("smartpredict.api.app")
    assert callable(module.create_app)


def test_synthetic_artifacts_load_with_contract_and_saved_threshold(
    deployment_models_dir: Path,
) -> None:
    """The production loader validates the complete saved bundle."""
    artifacts = load_model_artifacts(models_dir=deployment_models_dir)

    assert artifacts.model is not None
    assert artifacts.preprocessor is not None
    assert artifacts.selected_model == "xgboost"
    assert artifacts.threshold == pytest.approx(0.46)
    assert artifacts.feature_names == tuple(
        str(name) for name in artifacts.preprocessor.get_feature_names_out()
    )
    assert artifacts.model.n_features_in_ == len(artifacts.feature_names)


def test_health_endpoint(
    deployment_bundle: dict[str, Any], deployment_settings: Settings
) -> None:
    """The readiness endpoint reports that artifacts are loaded."""
    with _client(deployment_settings) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "artifacts_ready": True,
        "model_loaded": True,
        "preprocessor_loaded": True,
        "model_name": "xgboost",
    }


def test_single_prediction(
    deployment_settings: Settings, prepared_training_data: dict[str, Any]
) -> None:
    """One synthetic raw record produces one threshold-aware prediction."""
    record = _raw_records(prepared_training_data["split"].validation, 1)[0]

    with _client(deployment_settings) as client:
        response = client.post("/api/v1/predict", json=record)

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_records"] == 1
    assert len(payload["labels"]) == 1
    assert len(payload["feature_names"]) > 0
    assert payload["threshold"] == pytest.approx(0.46)
    assert payload["customer_ids"] == [record["customerID"]]


def test_batch_prediction(
    deployment_settings: Settings, prepared_training_data: dict[str, Any]
) -> None:
    """A batch of synthetic records preserves input order."""
    records = _raw_records(prepared_training_data["split"].validation, 3)

    with _client(deployment_settings) as client:
        response = client.post("/api/v1/predict/batch", json={"records": records})

    assert response.status_code == 200
    payload = response.json()
    assert payload["n_records"] == 3
    assert payload["customer_ids"] == [record["customerID"] for record in records]
    assert len(payload["labels"]) == 3
    assert payload["probabilities"] is not None


def test_non_default_threshold_changes_labels_not_probabilities(
    deployment_bundle: dict[str, Any],
    deployment_settings: Settings,
    prepared_training_data: dict[str, Any],
) -> None:
    """The saved 0.46 cutoff drives labels while scores stay threshold-free."""
    artifacts: ModelArtifacts = deployment_bundle["artifacts"]
    record = _raw_records(prepared_training_data["split"].validation, 1)[0]

    app = create_app(settings=deployment_settings)
    app.dependency_overrides[get_artifacts] = lambda: artifacts
    with TestClient(app, raise_server_exceptions=False) as client:
        saved = client.post("/api/v1/predict", json=record).json()

    probability = saved["probabilities"][0]
    assert saved["threshold"] == pytest.approx(0.46)
    assert saved["labels"][0] == int(probability >= 0.46)

    alternate = (
        0.9 if probability >= 0.46 else probability / 2 if probability > 0 else 0.0
    )
    alternate_artifacts = replace(artifacts, threshold=alternate)
    app = create_app(settings=deployment_settings)
    app.dependency_overrides[get_artifacts] = lambda: alternate_artifacts
    with TestClient(app, raise_server_exceptions=False) as client:
        changed = client.post("/api/v1/predict", json=record).json()

    assert changed["threshold"] == pytest.approx(alternate)
    assert changed["probabilities"] == pytest.approx(saved["probabilities"])
    assert changed["labels"][0] != saved["labels"][0]


def test_model_info_hides_internal_paths(
    deployment_settings: Settings,
) -> None:
    """Model metadata responses do not expose internal filesystem locations."""
    with _client(deployment_settings) as client:
        response = client.get("/api/v1/model/info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model"] == "xgboost"
    assert payload["feature_count"] == len(payload["feature_names"])
    assert payload["decision_threshold"] == pytest.approx(0.46)
    assert payload["metadata"]["dataset"]["raw_path"] == "synthetic.csv"
    assert "developer" not in json.dumps(payload)
    assert "C:\\" not in json.dumps(payload)


def test_missing_artifacts_produce_clear_error(tmp_path: Path) -> None:
    """A missing bundle is reported without leaking the absolute directory."""
    empty_dir = tmp_path / "missing-model-bundle"

    with pytest.raises(ArtifactError, match="missing") as exc_info:
        load_model_artifacts(models_dir=empty_dir)

    assert str(empty_dir) not in str(exc_info.value)


def test_invalid_feature_contract_produces_clear_error(
    deployment_models_dir: Path,
) -> None:
    """Metadata that disagrees with the transformer is rejected at load time."""
    metadata_path = deployment_models_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["features"]["count"] += 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ArtifactError, match="does not match") as exc_info:
        load_model_artifacts(models_dir=deployment_models_dir)

    assert str(deployment_models_dir) not in str(exc_info.value)


def test_inference_never_fits_estimator_or_preprocessor(
    deployment_bundle: dict[str, Any],
    deployment_settings: Settings,
    prepared_training_data: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction uses transform/predict only; fit methods are never invoked."""
    artifacts: ModelArtifacts = deployment_bundle["artifacts"]

    def forbidden_fit(*_: Any, **__: Any) -> None:
        """Fail loudly if inference attempts to retrain a saved object."""
        msg = "Inference must not fit saved artifacts."
        raise AssertionError(msg)

    monkeypatch.setattr(artifacts.model, "fit", forbidden_fit, raising=False)
    monkeypatch.setattr(artifacts.preprocessor, "fit", forbidden_fit, raising=False)
    monkeypatch.setattr(
        artifacts.preprocessor, "fit_transform", forbidden_fit, raising=False
    )

    app = create_app(settings=deployment_settings)
    app.dependency_overrides[get_artifacts] = lambda: artifacts
    record = _raw_records(prepared_training_data["split"].validation, 1)[0]
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/predict", json=record)

    assert response.status_code == 200


def test_startup_and_shutdown_are_logged(
    deployment_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """Application lifecycle transitions produce explicit log messages."""
    with caplog.at_level(logging.INFO, logger="smartpredict.api.app"), _client(
        deployment_settings
    ):
        pass

    messages = [record.message for record in caplog.records]
    assert any("API starting" in message for message in messages)
    assert any("API ready" in message for message in messages)
    assert any("API shutdown complete" in message for message in messages)


def test_startup_failure_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unusable artifact directory fails startup and logs the reason."""
    settings = Settings(
        _env_file=None,
        models_dir_override=tmp_path / "does-not-exist",
        configs_dir=CONFIGS_DIR,
    )

    with (
        caplog.at_level(logging.INFO, logger="smartpredict.api.app"),
        pytest.raises(ArtifactError),
        TestClient(create_app(settings=settings)),
    ):
        pass

    assert any("API startup failed" in record.message for record in caplog.records)

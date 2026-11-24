"""Tests for model artifact serialization and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pytest

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import CandidateModelConfig
from smartpredict.core.exceptions import ArtifactError
from smartpredict.evaluation.metrics import evaluate_predictions
from smartpredict.evaluation.service import evaluate_model
from smartpredict.models.artifacts import (
    build_artifact_paths,
    ensure_artifacts_writable,
    save_model_bundle,
)
from smartpredict.models.metadata import build_model_metadata
from smartpredict.models.training import TrainingService


def test_build_artifact_paths_uses_configured_names(
    fast_training_configs: dict[str, Path],
) -> None:
    """Artifact destinations follow train.yaml and CLI overrides."""
    config = load_training_config(train_config=fast_training_configs["train_config"])
    paths = build_artifact_paths(
        config.artifacts,
        models_dir=fast_training_configs["models_dir"],
        reports_dir=fast_training_configs["reports_dir"],
        plots_dir=fast_training_configs["plots_dir"],
    )
    assert paths["best_model"].name == "best_model.joblib"
    assert paths["preprocessor"].name == "preprocessing_pipeline.joblib"
    assert paths["metadata"].name == "model_metadata.json"
    assert paths["metadata"].parent == fast_training_configs["models_dir"]
    assert paths["comparison"].parent == fast_training_configs["reports_dir"]
    assert paths["confusion_matrix"].name == "confusion_matrix.png"


def test_existing_artifacts_require_explicit_overwrite(tmp_path: Path) -> None:
    """Existing outputs are never silently replaced."""
    target = tmp_path / "best_model.joblib"
    target.write_bytes(b"old")
    with pytest.raises(ArtifactError, match="--overwrite"):
        ensure_artifacts_writable({"model": target}, overwrite=False)
    ensure_artifacts_writable({"model": target}, overwrite=True)


def test_save_model_bundle_roundtrip(
    tmp_path: Path, prepared_training_data: dict[str, Any]
) -> None:
    """Joblib artifacts and JSON reports can be reloaded from disk."""
    config = prepared_training_data["config"]
    spec = prepared_training_data["spec"]
    split = prepared_training_data["split"]
    trained = TrainingService().train(
        "logistic_regression",
        CandidateModelConfig(params={"max_iter": 100}),
        prepared_training_data["features"],
        split.train["Churn"],
        split.validation["Churn"],
        seed=42,
        positive_label="Yes",
    )
    test_output = evaluate_model(
        trained.model,
        prepared_training_data["features"].test,
        split.test["Churn"],
        model_name=trained.name,
        split="test",
        positive_label="Yes",
    )
    paths = build_artifact_paths(
        config.artifacts,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        plots_dir=tmp_path / "plots",
    )
    # Keep this test focused on the model bundle, not every plot path.
    paths = {
        key: value
        for key, value in paths.items()
        if key in {"best_model", "preprocessor", "metadata", "comparison", "final_test"}
    }
    metadata = build_model_metadata(
        config=config,
        spec=spec,
        dataset_fingerprint="abc123",
        selected_model=trained.name,
        selected_params={"max_iter": 100},
        training_duration_seconds=float(trained.metadata["duration_seconds"]),
        dataset_dimensions={
            "rows": split.metadata.n_total,
            "columns": len(spec.expected_columns),
        },
        split_sizes=split.metadata.sizes,
        feature_names=prepared_training_data["features"].feature_names,
        validation_metrics={"roc_auc": 0.8},
        test_result=test_output.result,
    )
    written = save_model_bundle(
        model=trained.model,
        preprocessor=prepared_training_data["preprocessor"],
        metadata=metadata,
        comparison={"selected_model": trained.name},
        final_test_report={"test": test_output.result.to_dict()},
        paths=paths,
    )

    assert joblib.load(written["best_model"]).predict(
        prepared_training_data["features"].test
    ).shape == (split.test.shape[0],)
    assert joblib.load(written["preprocessor"]) is not None
    payload = json.loads(Path(written["metadata"]).read_text(encoding="utf-8"))
    assert payload["dataset"]["fingerprint_sha256"] == "abc123"
    assert payload["selection"]["primary_metric"] == "roc_auc"
    assert payload["final_test"]["split"] == "test"


def test_build_model_metadata_contains_reproducibility_fields(
    prepared_training_data: dict[str, Any],
) -> None:
    """Metadata includes configuration, data size, selection, and versions."""
    config = prepared_training_data["config"]
    spec = load_dataset_spec()
    split = prepared_training_data["split"]
    y_true = prepared_training_data["y_train"]
    output = evaluate_predictions(
        y_true,
        y_true,
        model_name="dummy",
        split="test",
    )
    metadata = build_model_metadata(
        config=config,
        spec=spec,
        dataset_fingerprint="digest",
        selected_model="logistic_regression",
        selected_params={"max_iter": 100},
        training_duration_seconds=0.125,
        dataset_dimensions={
            "rows": split.metadata.n_total,
            "columns": len(spec.expected_columns),
        },
        split_sizes=split.metadata.sizes,
        feature_names=prepared_training_data["features"].feature_names,
        validation_metrics={"roc_auc": 0.7},
        test_result=output,
    )
    assert metadata["model"]["selected"] == "logistic_regression"
    assert metadata["model"]["training_duration_seconds"] == 0.125
    assert metadata["dataset"]["dimensions"] == {
        "rows": split.metadata.n_total,
        "columns": len(spec.expected_columns),
    }
    assert metadata["data_sizes"] == split.metadata.sizes
    assert metadata["selection"]["split"] == "validation"
    assert "scikit-learn" in metadata["software_versions"]
    assert metadata["features"]["count"] == len(
        prepared_training_data["features"].feature_names
    )

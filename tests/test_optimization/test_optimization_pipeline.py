"""Synthetic end-to-end tests for the optimization workflow."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from smartpredict.config.schema import (
    DatasetSpec,
    SupportedModel,
    ThresholdConfig,
    TrainingConfig,
)
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.training import TrainingService
from smartpredict.optimization.pipeline import (
    OptimizationInputs,
    load_optimization_inputs,
    run_optimization_pipeline,
)
from smartpredict.optimization.search_space import SearchSpaceParameter, SearchSpaceType


def _pipeline_config(
    training_config: TrainingConfig, output_dir: Path, metadata_dir: Path
) -> TrainingConfig:
    """Return a small synthetic optimization configuration."""
    artifacts = replace(
        training_config.artifacts,
        models_dir=metadata_dir,
        reports_dir=metadata_dir,
        plots_dir=metadata_dir,
    )
    optimization = replace(
        training_config.optimization,
        model_type=SupportedModel.XGBOOST,
        n_trials=2,
        timeout_seconds=None,
        cv_folds=2,
        seed=42,
        output_dir=output_dir,
        study_name="synthetic_pipeline_study",
        search_space={
            "xgboost": {
                "max_depth": SearchSpaceParameter(
                    type=SearchSpaceType.INTEGER, low=2, high=3
                )
            }
        },
        threshold=ThresholdConfig(minimum=0.2, maximum=0.8, step=0.1),
    )
    return replace(
        training_config, optimization=optimization, artifacts=artifacts
    )


def _synthetic_model_factory(name: str, params: Mapping[str, Any]) -> Any:
    """Ignore search-only parameters and build a fast synthetic classifier."""
    return LogisticRegression(max_iter=100, random_state=42)


def test_pipeline_generates_artifacts_and_updates_metadata(
    prepared_training_data: dict[str, Any], tmp_path: Path
) -> None:
    """A synthetic run produces all artifacts without touching the test split."""
    spec = prepared_training_data["spec"]
    config = _pipeline_config(
        prepared_training_data["config"],
        tmp_path / "optimization",
        tmp_path / "models",
    )
    trained = TrainingService().train(
        "logistic_regression",
        prepared_training_data["config"].models["logistic_regression"],
        prepared_training_data["features"],
        prepared_training_data["y_train"],
        prepared_training_data["split"].validation[spec.target_column],
        seed=42,
        positive_label=spec.positive_label,
    )
    metadata_path = tmp_path / "models" / "model_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "model": {"selected": "xgboost", "parameters": {"max_iter": 100}},
                "final_test": {"metrics": {"roc_auc": 0.837}},
            }
        ),
        encoding="utf-8",
    )
    inputs = OptimizationInputs(
        train_frame=prepared_training_data["split"].train,
        validation_frame=prepared_training_data["split"].validation,
        baseline_name="xgboost",
        baseline_params={"max_iter": 100},
        baseline_model=trained.model,
        baseline_preprocessor=prepared_training_data["preprocessor"],
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")),
        metadata_path=metadata_path,
    )

    result = run_optimization_pipeline(
        config,
        spec,
        inputs,
        model_factory=_synthetic_model_factory,
    )

    output_dir = tmp_path / "optimization"
    for filename in (
        "optuna_study.json",
        "best_parameters.json",
        "optimization_history.png",
        "parameter_importance.png",
        "threshold_analysis.json",
        "threshold_precision_recall.png",
        "threshold_f1.png",
        "model_comparison.json",
    ):
        assert (output_dir / filename).is_file()

    study_payload = json.loads(
        (output_dir / "optuna_study.json").read_text(encoding="utf-8")
    )
    comparison = json.loads(
        (output_dir / "model_comparison.json").read_text(encoding="utf-8")
    )
    updated_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.study_result.completed_trials == 2
    assert study_payload["data_access"]["test_split"] is None
    assert comparison["data_access"]["test_split_evaluated"] is False
    assert updated_metadata["final_test"] == {"metrics": {"roc_auc": 0.837}}
    assert updated_metadata["decision_threshold"]["source_split"] == "validation"
    assert updated_metadata["optimization"]["test_split_evaluated"] is False
    assert result.threshold.source_split == "validation"


def test_loader_reads_only_train_and_validation_files(
    training_config: TrainingConfig,
    dataset_spec: DatasetSpec,
    synthetic_frame: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Input loading never requests the test split from disk."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    for name in ("train.csv", "validation.csv"):
        synthetic_frame.to_csv(data_dir / name, index=False)

    artifacts = replace(
        training_config.artifacts,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        plots_dir=tmp_path / "plots",
    )
    paths = build_artifact_paths(artifacts)
    for label in ("best_model", "preprocessor"):
        paths[label].parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(object(), paths[label])
    paths["metadata"].write_text(
        json.dumps({"model": {"selected": "xgboost", "parameters": {}}}),
        encoding="utf-8",
    )

    config = replace(
        training_config,
        artifacts=artifacts,
        optimization=replace(training_config.optimization, data_directory=data_dir),
    )
    loaded_paths: list[Path] = []

    def tracking_loader(path: str | Path, **kwargs: Any) -> SimpleNamespace:
        """Record requested CSV paths while returning synthetic rows."""
        candidate = Path(path)
        loaded_paths.append(candidate)
        return SimpleNamespace(frame=synthetic_frame.head(2))

    monkeypatch.setattr(
        "smartpredict.optimization.pipeline.load_csv", tracking_loader
    )
    inputs = load_optimization_inputs(config, dataset_spec)

    assert [path.name for path in loaded_paths] == ["train.csv", "validation.csv"]
    assert inputs.baseline_name == "xgboost"

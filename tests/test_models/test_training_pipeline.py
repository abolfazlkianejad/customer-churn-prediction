"""Tests for the end-to-end training workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pytest
import yaml

from smartpredict.config.schema import CandidateModelConfig
from smartpredict.core.exceptions import ArtifactError, DataError
from smartpredict.evaluation.service import evaluate_model as real_evaluate_model
from smartpredict.models.pipeline import run_training_workflow


def write_yaml(path: Path, content: dict[str, Any]) -> Path:
    """Write a YAML mapping to a temporary path."""
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_training_workflow_creates_artifacts_and_separates_splits(
    fast_training_configs: dict[str, Path],
) -> None:
    """A synthetic run produces all required artifacts and reports."""
    result = run_training_workflow(
        data_config=fast_training_configs["data_config"],
        train_config=fast_training_configs["train_config"],
        models_dir=fast_training_configs["models_dir"],
        reports_dir=fast_training_configs["reports_dir"],
        plots_dir=fast_training_configs["plots_dir"],
    )

    assert result.comparison.selected_model is not None
    assert set(result.validation_results) == {"logistic_regression", "decision_tree"}
    assert result.final_test.result.split == "test"
    assert result.final_test.result.evaluated_at
    for name in (
        "best_model",
        "preprocessor",
        "metadata",
        "comparison",
        "final_test",
        "confusion_matrix",
        "roc_curve",
        "precision_recall_curve",
    ):
        assert Path(result.artifact_paths[name]).is_file()

    report_payload = yaml.safe_load(
        Path(result.artifact_paths["final_test"]).read_text(encoding="utf-8")
    )
    assert report_payload["selection_split"] == "validation"
    assert report_payload["validation"]["split"] == "validation"
    assert report_payload["test"]["split"] == "test"

    reports_dir = Path(result.artifact_paths["validation_reports_dir"])
    for name in result.validation_results:
        report_path = reports_dir / f"{name}_validation_report.json"
        assert report_path.is_file()
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["model_name"] == name
        assert payload["split"] == "validation"
        assert payload["metrics"]
        assert payload["confusion_matrix"]
        assert payload["classification_report"]
        assert payload["metadata"]["model_name"] == name
        assert payload["metadata"]["evaluation_split"] == "validation"
        assert payload["metadata"]["created_at"]
        assert (
            payload["metadata"]["configuration"]["selection"]["split"]
            == "validation"
        )


def test_test_split_is_evaluated_exactly_once(
    fast_training_configs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No test-set evaluation occurs before final selection."""
    test_calls: list[str] = []

    def counting_evaluate_model(*args: Any, **kwargs: Any) -> Any:
        """Count test evaluations while delegating to the real implementation."""
        if kwargs.get("split") == "test":
            test_calls.append(str(kwargs.get("model_name")))
        return real_evaluate_model(*args, **kwargs)

    monkeypatch.setattr(
        "smartpredict.models.pipeline.evaluate_model", counting_evaluate_model
    )
    run_training_workflow(
        data_config=fast_training_configs["data_config"],
        train_config=fast_training_configs["train_config"],
        models_dir=fast_training_configs["models_dir"],
        reports_dir=fast_training_configs["reports_dir"],
        plots_dir=fast_training_configs["plots_dir"],
    )
    assert len(test_calls) == 1


def test_preprocessing_was_fitted_on_training_rows_only(
    fast_training_configs: dict[str, Path],
) -> None:
    """Saved preprocessing statistics match the training split."""
    result = run_training_workflow(
        data_config=fast_training_configs["data_config"],
        train_config=fast_training_configs["train_config"],
        models_dir=fast_training_configs["models_dir"],
        reports_dir=fast_training_configs["reports_dir"],
        plots_dir=fast_training_configs["plots_dir"],
    )
    preprocessor = joblib.load(result.artifact_paths["preprocessor"])
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    tenure_index = result.spec.numeric_columns.index("tenure")
    assert imputer.statistics_[tenure_index] == pytest.approx(
        result.split.train["tenure"].median()
    )


def test_failed_model_does_not_discard_successful_results(
    tmp_path: Path, fast_training_configs: dict[str, Path]
) -> None:
    """A bad candidate is reported while other models still train."""
    source = yaml.safe_load(
        fast_training_configs["train_config"].read_text(encoding="utf-8")
    )
    source["models"]["random_forest"] = {
        "enabled": True,
        "params": {"unknown_option": 1},
    }
    config_path = write_yaml(tmp_path / "with_failure.yaml", source)
    result = run_training_workflow(
        data_config=fast_training_configs["data_config"],
        train_config=config_path,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        plots_dir=tmp_path / "plots",
    )
    assert result.comparison.selected_model in {"logistic_regression", "decision_tree"}
    assert any(
        failure.model_name == "random_forest"
        for failure in result.comparison.failures
    )


def test_workflow_refuses_to_overwrite_existing_artifacts(
    fast_training_configs: dict[str, Path],
) -> None:
    """Existing artifact names require an explicit overwrite flag."""
    fast_training_configs["models_dir"].mkdir(parents=True)
    (fast_training_configs["models_dir"] / "best_model.joblib").write_bytes(b"old")

    with pytest.raises(ArtifactError, match="--overwrite"):
        run_training_workflow(
            data_config=fast_training_configs["data_config"],
            train_config=fast_training_configs["train_config"],
            models_dir=fast_training_configs["models_dir"],
            reports_dir=fast_training_configs["reports_dir"],
            plots_dir=fast_training_configs["plots_dir"],
        )

    result = run_training_workflow(
        data_config=fast_training_configs["data_config"],
        train_config=fast_training_configs["train_config"],
        models_dir=fast_training_configs["models_dir"],
        reports_dir=fast_training_configs["reports_dir"],
        plots_dir=fast_training_configs["plots_dir"],
        overwrite=True,
    )
    assert Path(result.artifact_paths["best_model"]).is_file()


def test_missing_dataset_is_reported(
    tmp_path: Path, project_root: Path
) -> None:
    """The workflow fails clearly when the configured CSV is absent."""
    source = yaml.safe_load(
        (project_root / "configs" / "data.yaml").read_text(encoding="utf-8")
    )
    source["dataset"]["raw_path"] = str(tmp_path / "absent.csv")
    data_config = write_yaml(tmp_path / "data.yaml", source)
    with pytest.raises(DataError, match="does not exist"):
        run_training_workflow(
            data_config=data_config,
            models_dir=tmp_path / "models",
            reports_dir=tmp_path / "reports",
            plots_dir=tmp_path / "plots",
        )


def test_candidate_model_config_is_used_by_training(
    prepared_training_data: dict[str, Any],
) -> None:
    """Candidate configuration flows through the training service."""
    from smartpredict.models.training import TrainingService

    split = prepared_training_data["split"]
    result = TrainingService().train(
        "logistic_regression",
        CandidateModelConfig(params={"max_iter": 37}),
        prepared_training_data["features"],
        split.train["Churn"],
        split.validation["Churn"],
        seed=42,
        positive_label="Yes",
    )
    assert result.metadata["parameters"] == {"max_iter": 37}

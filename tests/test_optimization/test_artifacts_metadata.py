"""Synthetic tests for optimization artifacts and additive metadata updates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from smartpredict.core.exceptions import ArtifactError, OptimizationError
from smartpredict.models.metadata import (
    merge_optimization_metadata,
    read_decision_threshold,
)
from smartpredict.optimization.artifacts import (
    save_optimization_artifacts,
    write_json_artifact,
)


def test_all_required_optimization_artifacts_are_written(tmp_path: Path) -> None:
    """The documented artifact set is generated from synthetic payloads."""
    output_dir = tmp_path / "optimization"
    plot_names = (
        "optimization_history",
        "parameter_importance",
        "threshold_precision_recall",
        "threshold_f1",
    )
    plot_paths = {}
    for name in plot_names:
        path = output_dir / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-png")
        plot_paths[name] = str(path)

    written = save_optimization_artifacts(
        output_dir,
        study_report={"schema_version": "1.0", "completed_trials": 2},
        best_parameters={"max_depth": 5},
        comparison={"test_split_evaluated": False},
        threshold_analysis={"selected_threshold": 0.4},
        plot_paths=plot_paths,
    )

    for filename in (
        "optuna_study.json",
        "best_parameters.json",
        "threshold_analysis.json",
        "model_comparison.json",
        "optimization_history.png",
        "parameter_importance.png",
        "threshold_precision_recall.png",
        "threshold_f1.png",
    ):
        assert (output_dir / filename).is_file()
        assert str(output_dir / filename) in written.values()

    report = json.loads((output_dir / "optuna_study.json").read_text(encoding="utf-8"))
    assert report["completed_trials"] == 2


def test_write_json_artifact_rejects_non_serializable_data(tmp_path: Path) -> None:
    """Persistence fails loudly instead of writing a partial report."""
    with pytest.raises(OptimizationError, match="Could not write"):
        write_json_artifact(tmp_path / "bad.json", {"value": object()})


def test_metadata_update_is_additive_and_threshold_is_readable() -> None:
    """Optimization metadata preserves all prior training information."""
    metadata = {
        "schema_version": "1.0",
        "model": {"selected": "xgboost", "parameters": {"max_depth": 4}},
        "final_test": {"split": "test", "metrics": {"roc_auc": 0.83}},
        "trained_at": "2026-01-01T00:00:00+00:00",
    }
    threshold_report = {
        "selected_threshold": 0.37,
        "baseline_threshold": 0.5,
        "source_split": "validation",
        "criterion": "weighted_score",
        "selected_metrics": {"precision": 0.7, "recall": 0.8},
        "generated_at": "2026-01-02T00:00:00+00:00",
        "note": "synthetic",
    }
    updated = merge_optimization_metadata(
        metadata,
        optimization={"completed_trials": 3, "selected_threshold": 0.37},
        threshold_report=threshold_report,
    )

    assert updated["model"] == metadata["model"]
    assert updated["final_test"] == metadata["final_test"]
    assert updated["trained_at"] == metadata["trained_at"]
    assert updated["optimization"]["completed_trials"] == 3
    assert updated["decision_threshold"]["selected_threshold"] == 0.37
    assert read_decision_threshold(updated) == 0.37
    assert read_decision_threshold({}) == 0.5

    with pytest.raises(ArtifactError, match="between 0 and 1"):
        read_decision_threshold({"decision_threshold": {"selected_threshold": 1.4}})

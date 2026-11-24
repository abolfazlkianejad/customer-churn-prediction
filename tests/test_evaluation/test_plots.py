"""Tests for evaluation visualization generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from smartpredict.config.schema import CandidateModelConfig
from smartpredict.evaluation.plots import generate_evaluation_plots
from smartpredict.evaluation.service import evaluate_model
from smartpredict.models.training import TrainingService


def test_generate_evaluation_plots_creates_all_required_files(
    tmp_path: Path, prepared_training_data: dict[str, Any]
) -> None:
    """All three required figures are written under the plots directory."""
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
    output = evaluate_model(
        trained.model,
        prepared_training_data["features"].test,
        split.test["Churn"],
        model_name=trained.name,
        split="test",
        positive_label="Yes",
    )
    paths = generate_evaluation_plots(output, tmp_path / "plots")
    assert len(paths) == 3
    for path in paths.values():
        assert Path(path).is_file()
        assert Path(path).stat().st_size > 0

"""Tests for SHAP explanation generation with synthetic artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from smartpredict.config.schema import CandidateModelConfig
from smartpredict.core.exceptions import ExplainabilityError
from smartpredict.explainability import service
from smartpredict.explainability.service import generate_explanations
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.training import TrainingService


class _FakeExplanation:
    """Minimal stand-in for ``shap.Explanation`` in plot tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Retain the values passed by the service."""
        self.values = kwargs.get("values")


class _FakePlots:
    """Stand-ins for SHAP plotting helpers that create real figures."""

    @staticmethod
    def bar(explanation: _FakeExplanation, show: bool = False) -> None:
        """Draw a trivial bar chart."""
        plt.figure()
        plt.bar(["a"], [1.0])

    @staticmethod
    def waterfall(
        explanation: _FakeExplanation,
        show: bool = False,
        max_display: int = 10,
    ) -> None:
        """Draw a trivial horizontal chart."""
        plt.figure()
        plt.barh(["a"], [1.0])


class _FakeShap:
    """Minimal SHAP module used to avoid depending on real explainers."""

    Explanation = _FakeExplanation
    plots = _FakePlots()

    @staticmethod
    def summary_plot(
        values: np.ndarray,
        features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
        plot_type: str = "dot",
        show: bool = False,
    ) -> None:
        """Draw a trivial summary figure."""
        plt.figure()
        plt.hist(np.asarray(values).ravel(), bins=5)


class _FakeTreeExplainer:
    """Deterministic explainer that returns one SHAP value per feature."""

    expected_value = 0.25

    def __init__(self, model: Any) -> None:
        """Retain the saved model reference."""
        self.model = model

    def shap_values(
        self, features: np.ndarray, check_additivity: bool = False
    ) -> np.ndarray:
        """Return a correctly shaped, constant SHAP matrix."""
        return np.full(np.asarray(features).shape, 0.01, dtype=float)


class _FailingExplainer(_FakeTreeExplainer):
    """Explainer that simulates a SHAP runtime failure."""

    def shap_values(
        self, features: np.ndarray, check_additivity: bool = False
    ) -> np.ndarray:
        """Raise as a real SHAP backend might."""
        msg = "simulated SHAP failure"
        raise RuntimeError(msg)


def _saved_artifacts(
    prepared_training_data: dict[str, Any], tmp_path: Path
) -> dict[str, Path]:
    """Train once for test setup and save reusable fake production artifacts."""
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
    config = prepared_training_data["config"]
    paths = build_artifact_paths(
        config.artifacts,
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
    )
    for key in ("best_model", "preprocessor", "metadata"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained.model, paths["best_model"])
    joblib.dump(prepared_training_data["preprocessor"], paths["preprocessor"])
    paths["metadata"].write_text(
        json.dumps({"model": {"selected": "xgboost", "parameters": {}}}),
        encoding="utf-8",
    )
    return paths


def _call(
    paths: dict[str, Path],
    output_dir: Path,
    records: Any,
    **overrides: Any,
) -> Any:
    """Invoke explanation generation with the fake SHAP backend."""
    options: dict[str, Any] = {"explainer_factory": _FakeTreeExplainer}
    options.update(overrides)
    return generate_explanations(
        models_dir=paths["best_model"].parent,
        reports_dir=paths["metadata"].parent,
        output_dir=output_dir,
        sample_size=4,
        local_examples=2,
        records=records,
        **options,
    )


def test_generate_explanations_creates_global_and_local_outputs(
    prepared_training_data: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explanations use saved artifacts and never refit the saved model."""
    paths = _saved_artifacts(prepared_training_data, tmp_path)
    monkeypatch.setattr(service, "_import_shap", lambda: _FakeShap)

    def fail_fit(*args: Any, **kwargs: Any) -> None:
        """Fail if any code attempts to refit the model."""
        raise AssertionError("Explanations must not fit the saved model.")

    monkeypatch.setattr(LogisticRegression, "fit", fail_fit)
    output_dir = tmp_path / "explainability"
    result = _call(
        paths,
        output_dir,
        prepared_training_data["split"].validation,
    )

    global_payload = json.loads(result.global_importance.read_text(encoding="utf-8"))
    local_payload = json.loads(result.local_report.read_text(encoding="utf-8"))
    assert result.sample_size == 4
    assert global_payload["explanation_type"] == "global"
    assert global_payload["evaluation_split"] == "validation"
    assert global_payload["feature_names"] == list(result.feature_names)
    assert len(global_payload["feature_importance"]) == len(result.feature_names)
    assert global_payload["feature_importance"][0]["mean_abs_shap"] >= (
        global_payload["feature_importance"][-1]["mean_abs_shap"]
    )
    assert local_payload["explanation_type"] == "local"
    assert len(local_payload["records"]) == 2
    for entry in local_payload["records"]:
        assert entry["explanation_type"] == "local"
        assert entry["customer_id"]
        assert len(entry["feature_contributions"]) == len(result.feature_names)
    assert result.summary_plot.is_file()
    assert result.bar_plot.is_file()
    assert len(result.local_plots) == 2
    assert all(path.is_file() for path in result.local_plots)


def test_existing_explanations_require_overwrite(
    prepared_training_data: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explanation outputs are protected by the project's overwrite policy."""
    paths = _saved_artifacts(prepared_training_data, tmp_path)
    monkeypatch.setattr(service, "_import_shap", lambda: _FakeShap)
    output_dir = tmp_path / "explainability"
    _call(paths, output_dir, prepared_training_data["split"].validation)

    with pytest.raises(ExplainabilityError, match="--overwrite"):
        _call(paths, output_dir, prepared_training_data["split"].validation)


def test_non_xgboost_metadata_is_rejected(
    prepared_training_data: dict[str, Any], tmp_path: Path
) -> None:
    """The CLI refuses to mislabel another model's explanations as XGBoost."""
    paths = _saved_artifacts(prepared_training_data, tmp_path)
    paths["metadata"].write_text(
        json.dumps({"model": {"selected": "logistic_regression"}}),
        encoding="utf-8",
    )

    with pytest.raises(ExplainabilityError, match="XGBoost"):
        _call(
            paths,
            tmp_path / "explainability",
            prepared_training_data["split"].validation,
        )


def test_shap_failures_use_project_exception(
    prepared_training_data: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SHAP runtime errors are translated into ExplainabilityError."""
    paths = _saved_artifacts(prepared_training_data, tmp_path)
    monkeypatch.setattr(service, "_import_shap", lambda: _FakeShap)

    with pytest.raises(ExplainabilityError, match="SHAP value computation failed"):
        _call(
            paths,
            tmp_path / "explainability",
            prepared_training_data["split"].validation,
            explainer_factory=_FailingExplainer,
        )

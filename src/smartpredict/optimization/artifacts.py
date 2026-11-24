"""Persistence helpers for optimization studies and comparison reports."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smartpredict.config.schema import OptimizationConfig
from smartpredict.core.exceptions import OptimizationError
from smartpredict.core.logging import get_logger
from smartpredict.evaluation.metrics import EvaluationResult
from smartpredict.optimization.study import StudyResult

logger = get_logger(__name__)


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> str:
    """Write one JSON artifact with stable formatting.

    Args:
        path: Destination path.
        payload: JSON-serializable mapping.

    Returns:
        String representation of the written path.

    Raises:
        OptimizationError: If the artifact cannot be written.

    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Could not write optimization artifact '{path}': {exc}"
        raise OptimizationError(msg) from exc
    return str(path)


def build_comparison_report(
    config: OptimizationConfig,
    study_result: StudyResult,
    *,
    baseline_name: str,
    baseline_params: Mapping[str, Any],
    baseline_metrics: EvaluationResult,
    optimized_params: Mapping[str, Any],
    optimized_metrics: EvaluationResult,
    threshold_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a validation-only baseline-versus-optimized comparison payload.

    Args:
        config: Typed optimization configuration.
        study_result: Completed Optuna study summary.
        baseline_name: Name of the deployed baseline model.
        baseline_params: Hyperparameters of the deployed baseline.
        baseline_metrics: Baseline validation evaluation.
        optimized_params: Full parameters of the optimized candidate.
        optimized_metrics: Optimized candidate validation evaluation.
        threshold_report: Validation threshold analysis payload.

    Returns:
        JSON-serializable comparison report.

    """
    baseline_values = baseline_metrics.metrics
    optimized_values = optimized_metrics.metrics
    differences: dict[str, float | None] = {}
    for name in baseline_values:
        baseline_value = baseline_values[name]
        optimized_value = optimized_values.get(name)
        if baseline_value is None or optimized_value is None:
            differences[name] = None
        else:
            differences[name] = float(optimized_value - baseline_value)

    cv_result = study_result.best_trial.user_attrs.get("cv_result") or {}
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "experiment": {
            "model_type": config.model_type.value,
            "metric": config.metric.value,
            "configured_direction": config.direction.value,
            "objective_direction": study_result.objective_direction,
            "study_name": study_result.study.study_name,
            "seed": config.seed,
            "duration_seconds": study_result.duration_seconds,
        },
        "data_access": {
            "validation_split": str(config.data_directory / config.validation_file),
            "test_split_evaluated": False,
            "note": "Only train and validation splits were read in this phase.",
        },
        "baseline": {
            "model": baseline_name,
            "parameters": dict(baseline_params),
            "validation": baseline_metrics.to_dict(),
        },
        "optimized_candidate": {
            "model": config.model_type.value,
            "parameters": dict(optimized_params),
            "validation": optimized_metrics.to_dict(),
            "cross_validation": dict(cv_result),
            "deployed": False,
            "deployment_note": (
                "The optimized candidate is stored for comparison only; promoting "
                "it would require a new isolated test evaluation."
            ),
        },
        "metric_differences_optimized_minus_baseline": differences,
        "threshold": dict(threshold_report),
    }


def save_optimization_artifacts(
    output_dir: Path,
    *,
    study_report: Mapping[str, Any],
    best_parameters: Mapping[str, Any],
    comparison: Mapping[str, Any],
    threshold_analysis: Mapping[str, Any],
    plot_paths: Mapping[str, str],
) -> dict[str, str]:
    """Persist every required optimization artifact.

    Args:
        output_dir: Destination directory.
        study_report: Full Optuna study payload.
        best_parameters: Best-parameter payload.
        comparison: Baseline-versus-optimized payload.
        threshold_analysis: Validation threshold payload.
        plot_paths: Already generated PNG paths keyed by artifact name.

    Returns:
        Mapping from artifact name to written path.

    Raises:
        OptimizationError: If a required plot path is missing.

    """
    required_plots = (
        "optimization_history",
        "parameter_importance",
        "threshold_precision_recall",
        "threshold_f1",
    )
    missing = [name for name in required_plots if name not in plot_paths]
    if missing:
        msg = f"Missing required optimization plots: {', '.join(missing)}."
        raise OptimizationError(msg)

    written: dict[str, str] = {}
    written["optuna_study"] = write_json_artifact(
        output_dir / "optuna_study.json", study_report
    )
    written["best_parameters"] = write_json_artifact(
        output_dir / "best_parameters.json", best_parameters
    )
    written["threshold_analysis"] = write_json_artifact(
        output_dir / "threshold_analysis.json", threshold_analysis
    )
    written["comparison"] = write_json_artifact(
        output_dir / "model_comparison.json", comparison
    )
    for name in required_plots:
        written[name] = plot_paths[name]
    logger.info("Saved %s optimization artifacts in %s", len(written), output_dir)
    return written

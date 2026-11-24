"""Reproducibility metadata for training runs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from platform import python_version
from typing import Any

from smartpredict.config.schema import DatasetSpec, TrainingConfig
from smartpredict.core.exceptions import ArtifactError
from smartpredict.evaluation.metrics import EvaluationResult

METADATA_SCHEMA_VERSION = "1.0"


def _package_version(name: str) -> str:
    """Return an installed package version or ``unknown``."""
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - environment dependent
        return "unknown"


def software_versions() -> dict[str, str]:
    """Collect relevant runtime package versions."""
    return {
        "python": python_version(),
        "smartpredict": _package_version("smartpredict"),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit-learn": _package_version("scikit-learn"),
        "joblib": _package_version("joblib"),
        "matplotlib": _package_version("matplotlib"),
        "seaborn": _package_version("seaborn"),
        "xgboost": _package_version("xgboost"),
    }


def build_model_metadata(
    *,
    config: TrainingConfig,
    spec: DatasetSpec,
    dataset_fingerprint: str,
    selected_model: str,
    selected_params: dict[str, Any],
    training_duration_seconds: float,
    dataset_dimensions: dict[str, int],
    split_sizes: dict[str, int],
    feature_names: tuple[str, ...],
    validation_metrics: dict[str, float | None],
    test_result: EvaluationResult,
) -> dict[str, Any]:
    """Assemble JSON-serializable experiment metadata.

    Args:
        config: Loaded training configuration.
        spec: Loaded dataset specification.
        dataset_fingerprint: SHA-256 of the raw dataset file.
        selected_model: Name of the validation-selected model.
        selected_params: Hyperparameters used for the selected model.
        training_duration_seconds: Fit time of the selected model.
        dataset_dimensions: Raw row and column counts.
        split_sizes: Row counts for train, validation, and test.
        feature_names: Ordered transformed feature names.
        validation_metrics: Validation metrics for the selected model.
        test_result: Final isolated test evaluation.

    Returns:
        Metadata dictionary suitable for JSON serialization.

    """
    return {
        "schema_version": METADATA_SCHEMA_VERSION,
        "experiment": {
            "name": config.experiment_name,
            "description": config.description,
            "seed": config.seed,
        },
        "model": {
            "selected": selected_model,
            "parameters": dict(selected_params),
            "random_seed": config.seed,
            "training_duration_seconds": training_duration_seconds,
        },
        "selection": {
            "split": "validation",
            "primary_metric": config.evaluation.primary_metric.value,
            "optimization_direction": config.evaluation.optimization_direction.value,
            "validation_metrics": dict(validation_metrics),
        },
        "dataset": {
            "name": spec.name,
            "fingerprint_sha256": dataset_fingerprint,
            "raw_path": str(spec.raw_path),
            "dimensions": dict(dataset_dimensions),
            "target_column": spec.target_column,
            "positive_label": spec.positive_label,
            "negative_label": spec.negative_label,
        },
        "data_sizes": dict(split_sizes),
        "features": {
            "count": len(feature_names),
            "names": list(feature_names),
        },
        "final_test": {
            "split": test_result.split,
            "metrics": dict(test_result.metrics),
            "evaluated_at": test_result.evaluated_at,
        },
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "software_versions": software_versions(),
    }


def read_decision_threshold(metadata: Mapping[str, Any]) -> float:
    """Return the saved decision threshold, defaulting to 0.5.

    Args:
        metadata: Loaded model metadata mapping.

    Returns:
        Positive-class decision threshold used by the inference service.

    Raises:
        ArtifactError: If the saved threshold is not a valid probability.

    """
    section = metadata.get("decision_threshold", 0.5)
    if isinstance(section, Mapping):
        value = section.get("selected_threshold", 0.5)
    else:
        value = section
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = "Saved decision_threshold must be a number."
        raise ArtifactError(msg)
    if not 0.0 <= float(value) <= 1.0:
        msg = "Saved decision_threshold must be between 0 and 1."
        raise ArtifactError(msg)
    return float(value)


def merge_optimization_metadata(
    metadata: Mapping[str, Any],
    *,
    optimization: Mapping[str, Any],
    threshold_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Add optimization and threshold metadata without dropping existing fields.

    Args:
        metadata: Existing model metadata.
        optimization: Optimization experiment payload to merge.
        threshold_report: Validation threshold analysis payload.

    Returns:
        New metadata dictionary with all prior top-level information retained.

    """
    updated = deepcopy(dict(metadata))
    current_optimization = updated.get("optimization")
    merged_optimization = (
        deepcopy(dict(current_optimization))
        if isinstance(current_optimization, Mapping)
        else {}
    )
    merged_optimization.update(deepcopy(dict(optimization)))
    updated["optimization"] = merged_optimization

    current_threshold = updated.get("decision_threshold")
    merged_threshold = (
        deepcopy(dict(current_threshold))
        if isinstance(current_threshold, Mapping)
        else {}
    )
    merged_threshold.update(
        {
            "schema_version": "1.0",
            "selected_threshold": threshold_report.get("selected_threshold"),
            "baseline_threshold": threshold_report.get("baseline_threshold"),
            "source_split": threshold_report.get("source_split"),
            "criterion": threshold_report.get("criterion"),
            "selected_metrics": threshold_report.get("selected_metrics"),
            "baseline_metrics": threshold_report.get("baseline_metrics"),
            "generated_at": threshold_report.get("generated_at"),
            "note": threshold_report.get("note"),
        }
    )
    updated["decision_threshold"] = merged_threshold
    return updated

"""Joblib serialization and JSON report persistence."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import ClassifierMixin

from smartpredict.config.schema import ArtifactConfig, TrainingConfig
from smartpredict.core.exceptions import ArtifactError
from smartpredict.core.logging import get_logger

logger = get_logger(__name__)


def build_artifact_paths(
    config: ArtifactConfig,
    *,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    plots_dir: Path | None = None,
) -> dict[str, Path]:
    """Resolve all output paths from artifact configuration and overrides."""
    model_dir = models_dir or config.models_dir
    report_dir = reports_dir or config.reports_dir
    plot_dir = plots_dir or config.plots_dir
    return {
        "best_model": model_dir / config.best_model_file,
        "preprocessor": model_dir / config.preprocessor_file,
        "metadata": model_dir / config.metadata_file,
        "comparison": report_dir / config.comparison_file,
        "final_test": report_dir / config.final_test_file,
        "confusion_matrix": plot_dir / config.confusion_matrix_file,
        "roc_curve": plot_dir / config.roc_curve_file,
        "precision_recall_curve": plot_dir / config.precision_recall_file,
    }


def ensure_artifacts_writable(paths: dict[str, Path], *, overwrite: bool) -> None:
    """Fail before writing when any target artifact already exists.

    Args:
        paths: Artifact destinations.
        overwrite: Explicit permission to replace existing artifacts.

    Raises:
        ArtifactError: If outputs exist and overwriting was not requested.

    """
    if overwrite:
        return
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing:
        msg = (
            "Refusing to overwrite existing artifacts without --overwrite: "
            + ", ".join(existing)
        )
        raise ArtifactError(msg)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON document with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


VALIDATION_REPORTS_DIRECTORY = "validation_reports"
VALIDATION_REPORT_SCHEMA_VERSION = "1.0"


def build_validation_report_paths(
    model_names: Iterable[str], reports_dir: Path
) -> dict[str, Path]:
    """Resolve one validation-report path per candidate model."""
    directory = reports_dir / VALIDATION_REPORTS_DIRECTORY
    return {name: directory / f"{name}_validation_report.json" for name in model_names}


def build_validation_report_payload(
    result: Any,
    config: TrainingConfig,
    *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Wrap an evaluation result with audit metadata.

    Existing :meth:`EvaluationResult.to_dict` fields remain at the top level
    so older report readers keep working; new metadata is additive.
    """
    payload: dict[str, Any] = result.to_dict()
    candidate = config.models.get(result.model_name)
    payload["metadata"] = {
        "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
        "model_name": result.model_name,
        "evaluation_split": result.split,
        "created_at": generated_at or datetime.now(tz=UTC).isoformat(),
        "configuration": {
            "experiment": {
                "name": config.experiment_name,
                "seed": config.seed,
            },
            "model": {
                "name": result.model_name,
                "enabled": candidate.enabled if candidate else None,
                "parameters": dict(candidate.params) if candidate else {},
                "legacy_selected_model": config.model.name,
            },
            "features": {
                "numeric_imputation": config.features.numeric_imputation.value,
                "categorical_imputation": (
                    config.features.categorical_imputation.value
                ),
                "categorical_encoding": config.features.categorical_encoding.value,
                "scale_numeric": config.features.scale_numeric,
                "handle_unknown": config.features.handle_unknown,
            },
            "selection": {
                "split": "validation",
                "primary_metric": config.evaluation.primary_metric.value,
                "optimization_direction": (
                    config.evaluation.optimization_direction.value
                ),
            },
            "data": {
                "target_column": config.data.target_column,
                "positive_label": config.data.positive_label,
                "negative_label": config.data.negative_label,
                "id_column": config.data.id_column,
                "split": {
                    "strategy": config.data.split.strategy.value,
                    "seed": config.data.split.seed,
                    "train": config.data.split.train,
                    "validation": config.data.split.validation,
                    "test": config.data.split.test,
                },
            },
        },
    }
    return payload


def save_validation_reports(
    results: Mapping[str, Any],
    config: TrainingConfig,
    paths: Mapping[str, Path],
) -> dict[str, str]:
    """Persist validation metrics, confusion matrices, and reports for all models."""
    written: dict[str, str] = {}
    try:
        for name, result in results.items():
            path = paths.get(name)
            if path is None:
                msg = f"No validation report path configured for model '{name}'."
                raise ArtifactError(msg)
            _write_json(path, build_validation_report_payload(result, config))
            written[name] = str(path)
    except ArtifactError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Could not persist validation reports: {exc}"
        raise ArtifactError(msg) from exc
    logger.info("Saved %s validation reports", len(written))
    return written


def save_model_bundle(
    *,
    model: ClassifierMixin,
    preprocessor: Any,
    metadata: dict[str, Any],
    comparison: dict[str, Any],
    final_test_report: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, str]:
    """Persist the selected model, preprocessing pipeline, and reports.

    Args:
        model: Fitted best model.
        preprocessor: Fitted preprocessing transformer from the data phase.
        metadata: Experiment and reproducibility metadata.
        comparison: Validation comparison report.
        final_test_report: Final isolated test evaluation report.
        paths: Artifact destinations from :func:`build_artifact_paths`.

    Returns:
        Mapping from artifact label to written path.

    Raises:
        ArtifactError: If serialization or file writing fails.

    """
    try:
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, paths["best_model"])
        joblib.dump(preprocessor, paths["preprocessor"])
        _write_json(paths["metadata"], metadata)
        _write_json(paths["comparison"], comparison)
        _write_json(paths["final_test"], final_test_report)
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Could not persist model artifacts: {exc}"
        raise ArtifactError(msg) from exc

    written = {name: str(path) for name, path in paths.items()}
    logger.info("Saved %s training artifacts", len(written))
    return written

"""End-to-end configuration-driven training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.compose import ColumnTransformer

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import DatasetSpec, TrainingConfig
from smartpredict.core.exceptions import (
    DataError,
    ModelTrainingError,
    SmartPredictError,
)
from smartpredict.core.logging import get_logger
from smartpredict.data.loader import DatasetMetadata, load_csv
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.splitter import SplitResult, stratified_split
from smartpredict.data.validation import ValidationReport, validate_dataset
from smartpredict.evaluation.comparison import (
    ModelComparisonReport,
    compare_models,
    select_best_model,
)
from smartpredict.evaluation.metrics import EvaluationOutput, EvaluationResult
from smartpredict.evaluation.plots import generate_evaluation_plots
from smartpredict.features.preprocessor import FeatureSet, fit_transform_features
from smartpredict.models.artifacts import (
    build_artifact_paths,
    build_validation_report_paths,
    ensure_artifacts_writable,
    save_model_bundle,
    save_validation_reports,
)
from smartpredict.models.metadata import build_model_metadata
from smartpredict.models.training import TrainingResult, TrainingService

logger = get_logger(__name__)


def evaluate_model(*args: Any, **kwargs: Any) -> Any:
    """Delegate to the evaluation service without an import-time cycle."""
    from smartpredict.evaluation.service import evaluate_model as service_evaluate

    return service_evaluate(*args, **kwargs)


@dataclass(frozen=True)
class TrainingWorkflowResult:
    """All in-memory outputs of one training run."""

    spec: DatasetSpec
    config: TrainingConfig
    load_metadata: DatasetMetadata
    validation_report: ValidationReport
    split: SplitResult
    features: FeatureSet
    training_results: tuple[TrainingResult, ...]
    validation_results: dict[str, EvaluationResult]
    comparison: ModelComparisonReport
    final_test: EvaluationOutput
    artifact_paths: dict[str, str]


def _prepare_training_inputs(
    spec: DatasetSpec, config: TrainingConfig
) -> tuple[
    DatasetMetadata,
    ValidationReport,
    SplitResult,
    FeatureSet,
    ColumnTransformer,
]:
    """Load, validate, split, and preprocess data using Phase 2 components."""
    if not spec.raw_path.is_file():
        msg = f"Dataset file does not exist: {spec.raw_path}"
        raise DataError(msg)

    loaded = load_csv(spec.raw_path)
    schema = DatasetSchema.from_spec(spec)
    coerced, report = validate_dataset(
        loaded.frame, schema, source_path=loaded.metadata.path
    )
    report.raise_if_invalid()

    split = stratified_split(coerced, spec, seed=spec.split.seed)
    features, preprocessor = fit_transform_features(
        split.train,
        split.validation,
        split.test,
        spec,
        config.features,
    )
    return loaded.metadata, report, split, features, preprocessor


def run_training_workflow(
    *,
    data_config: str | Path | None = None,
    train_config: str | Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    plots_dir: Path | None = None,
    overwrite: bool = False,
    registry: Any | None = None,
) -> TrainingWorkflowResult:
    """Train, compare, select, and finally evaluate candidate models.

    Validation data is used only for model selection. The test split is
    evaluated exactly once, after the best model has been chosen.

    Args:
        data_config: Path to ``data.yaml``.
        train_config: Path to ``train.yaml``.
        models_dir: Optional output directory override for model artifacts.
        reports_dir: Optional output directory override for JSON reports.
        plots_dir: Optional output directory override for evaluation plots.
        overwrite: Explicit permission to replace existing artifacts.
        registry: Optional model registry override, mainly useful in tests.

    Returns:
        In-memory workflow outputs and persisted artifact paths.

    Raises:
        SmartPredictError: If any stage of the workflow fails.

    """
    spec = load_dataset_spec(data_config)
    config = load_training_config(data_config, train_config)
    paths = build_artifact_paths(
        config.artifacts,
        models_dir=models_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
    )
    effective_reports_dir = reports_dir or config.artifacts.reports_dir
    validation_report_paths = build_validation_report_paths(
        config.models.keys(), effective_reports_dir
    )
    ensure_artifacts_writable(paths, overwrite=overwrite)
    ensure_artifacts_writable(validation_report_paths, overwrite=overwrite)

    (
        load_metadata,
        validation_report,
        split,
        features,
        preprocessor,
    ) = _prepare_training_inputs(spec, config)
    logger.info(
        "Prepared data: train=%s validation=%s test=%s",
        split.train.shape[0],
        split.validation.shape[0],
        split.test.shape[0],
    )

    service = TrainingService(registry=registry)
    training_results: list[TrainingResult] = []
    validation_results: dict[str, EvaluationResult] = {}
    failures: dict[str, str] = {}

    for name, candidate in config.models.items():
        if not candidate.enabled:
            logger.info("Skipping disabled model '%s'", name)
            continue
        try:
            result = service.train(
                name,
                candidate,
                features,
                split.train[spec.target_column],
                split.validation[spec.target_column],
                seed=config.seed,
                positive_label=spec.positive_label,
            )
            evaluation = evaluate_model(
                result.model,
                features.validation,
                split.validation[spec.target_column],
                model_name=name,
                split="validation",
                positive_label=spec.positive_label,
            )
        except SmartPredictError as exc:
            failures[name] = str(exc)
            logger.error("Model '%s' failed: %s", name, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            failures[name] = f"{type(exc).__name__}: {exc}"
            logger.exception("Unexpected failure while training model '%s'", name)
            continue

        training_results.append(result)
        validation_results[name] = evaluation.result

    if not training_results:
        details = "; ".join(f"{name}: {message}" for name, message in failures.items())
        msg = f"No model was trained successfully. {details}".strip()
        raise ModelTrainingError(msg)

    comparison = compare_models(
        training_results,
        validation_results,
        primary_metric=config.evaluation.primary_metric,
        direction=config.evaluation.optimization_direction,
        failure_messages=failures,
    )
    best = select_best_model(comparison, training_results)
    logger.info(
        "Selected model '%s' by validation %s (%s)",
        best.name,
        comparison.primary_metric,
        comparison.optimization_direction,
    )

    # Persist validation-only evidence before the isolated test evaluation.
    validation_written = save_validation_reports(
        validation_results, config, validation_report_paths
    )

    # The isolated test set is touched only here, after model selection.
    final_test = evaluate_model(
        best.model,
        features.test,
        split.test[spec.target_column],
        model_name=best.name,
        split="test",
        positive_label=spec.positive_label,
    )
    logger.info("Final test evaluation completed once for '%s'", best.name)

    plots = generate_evaluation_plots(
        final_test,
        paths["confusion_matrix"].parent,
        confusion_file=paths["confusion_matrix"].name,
        roc_file=paths["roc_curve"].name,
        precision_recall_file=paths["precision_recall_curve"].name,
    )

    selected_params = dict(config.models[best.name].params)
    metadata = build_model_metadata(
        config=config,
        spec=spec,
        dataset_fingerprint=load_metadata.sha256,
        selected_model=best.name,
        selected_params=selected_params,
        training_duration_seconds=float(best.metadata["duration_seconds"]),
        dataset_dimensions={
            "rows": load_metadata.n_rows,
            "columns": load_metadata.n_columns,
        },
        split_sizes=split.metadata.sizes,
        feature_names=features.feature_names,
        validation_metrics=dict(validation_results[best.name].metrics),
        test_result=final_test.result,
    )
    final_test_report: dict[str, Any] = {
        "selected_model": best.name,
        "selection_split": "validation",
        "primary_selection_metric": comparison.primary_metric,
        "optimization_direction": comparison.optimization_direction,
        "note": (
            "The model was selected using validation metrics only; the test "
            "split was evaluated exactly once after selection."
        ),
        "validation": validation_results[best.name].to_dict(),
        "test": final_test.result.to_dict(),
    }

    written = save_model_bundle(
        model=best.model,
        preprocessor=preprocessor,
        metadata=metadata,
        comparison=comparison.to_dict(),
        final_test_report=final_test_report,
        paths=paths,
    )
    written.update(validation_written)
    written["validation_reports_dir"] = str(
        effective_reports_dir / "validation_reports"
    )
    written.update(plots)
    return TrainingWorkflowResult(
        spec=spec,
        config=config,
        load_metadata=load_metadata,
        validation_report=validation_report,
        split=split,
        features=features,
        training_results=tuple(training_results),
        validation_results=validation_results,
        comparison=comparison,
        final_test=final_test,
        artifact_paths=written,
    )

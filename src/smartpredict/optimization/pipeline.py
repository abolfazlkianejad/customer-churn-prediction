"""Orchestration of validation-only Optuna optimization and threshold analysis."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import DatasetSpec, TrainingConfig
from smartpredict.core.exceptions import OptimizationError
from smartpredict.core.logging import get_logger
from smartpredict.core.paths import ensure_directory
from smartpredict.data.loader import load_csv
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.validation import validate_dataset
from smartpredict.evaluation.metrics import EvaluationResult, evaluate_predictions
from smartpredict.features.preprocessor import (
    build_preprocessor,
    prepare_features_frame,
)
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.factory import create_model
from smartpredict.models.metadata import merge_optimization_metadata
from smartpredict.models.prediction import binary_targets, generate_predictions
from smartpredict.optimization.artifacts import (
    build_comparison_report,
    save_optimization_artifacts,
    write_json_artifact,
)
from smartpredict.optimization.cross_validation import (
    PreprocessorFactory,
    cross_validate_params,
)
from smartpredict.optimization.search_space import (
    merge_parameters,
    suggest_parameters,
    validate_search_space,
)
from smartpredict.optimization.study import (
    StudyResult,
    best_parameters_report,
    create_or_load_study,
    run_optimization,
    save_study_plots,
    study_report,
)
from smartpredict.optimization.threshold import (
    ThresholdAnalysisResult,
    analyze_thresholds,
    save_threshold_plots,
)

logger = get_logger(__name__)

ModelFactory = Callable[[str, Mapping[str, Any]], Any]


@dataclass(frozen=True)
class OptimizationResult:
    """Every artifact and result produced by one optimization run."""

    config: TrainingConfig
    study_result: StudyResult
    baseline_parameters: dict[str, Any]
    optimized_parameters: dict[str, Any]
    baseline_validation: EvaluationResult
    optimized_validation: EvaluationResult
    threshold: ThresholdAnalysisResult
    comparison: dict[str, Any]
    metadata: dict[str, Any]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        """Return a compact console/report summary without raw trial arrays."""
        return {
            "study_name": self.study_result.study.study_name,
            "completed_trials": self.study_result.completed_trials,
            "failed_trials": self.study_result.failed_trials,
            "best_parameters": dict(self.study_result.best_params),
            "best_cv_score": self.study_result.best_value,
            "baseline_validation_metrics": dict(self.baseline_validation.metrics),
            "optimized_validation_metrics": dict(self.optimized_validation.metrics),
            "selected_threshold": self.threshold.selected.threshold,
            "threshold_0_5_metrics": self.threshold.baseline.to_dict(),
            "selected_threshold_metrics": self.threshold.selected.to_dict(),
            "test_split_evaluated": False,
        }


@dataclass(frozen=True)
class OptimizationInputs:
    """Saved baseline artifacts and the two splits allowed in this phase."""

    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    baseline_name: str
    baseline_params: dict[str, Any]
    baseline_model: Any
    baseline_preprocessor: ColumnTransformer
    metadata: dict[str, Any]
    metadata_path: Path


def _validated_frame(frame: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    """Validate one split against the dataset contract."""
    coerced, report = validate_dataset(
        frame, DatasetSchema.from_spec(spec), require_target=True
    )
    report.raise_if_invalid()
    return coerced


def load_optimization_inputs(
    config: TrainingConfig, spec: DatasetSpec
) -> OptimizationInputs:
    """Load only train/validation rows and the saved baseline bundle.

    Args:
        config: Loaded training configuration.
        spec: Loaded dataset specification.

    Returns:
        Inputs required by :func:`run_optimization_pipeline`.

    Raises:
        OptimizationError: If required splits or saved baseline artifacts are
            unavailable.

    """
    optimization = config.optimization
    train_path = optimization.data_directory / optimization.train_file
    validation_path = optimization.data_directory / optimization.validation_file
    for label, path in (("train", train_path), ("validation", validation_path)):
        if not path.is_file():
            msg = f"Optimization requires the {label} split at {path}."
            raise OptimizationError(msg)

    artifact_paths = build_artifact_paths(config.artifacts)
    for label in ("best_model", "preprocessor", "metadata"):
        if not artifact_paths[label].is_file():
            msg = (
                f"Optimization requires the saved '{label}' artifact at "
                f"{artifact_paths[label]}."
            )
            raise OptimizationError(msg)

    try:
        train_frame = load_csv(train_path).frame
        validation_frame = load_csv(validation_path).frame
        baseline_model = joblib.load(artifact_paths["best_model"])
        baseline_preprocessor = joblib.load(artifact_paths["preprocessor"])
        metadata = json.loads(artifact_paths["metadata"].read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not load optimization inputs: {exc}"
        raise OptimizationError(msg) from exc
    if not isinstance(metadata, dict):
        msg = "Saved model metadata must be a JSON object."
        raise OptimizationError(msg)

    model_section = metadata.get("model")
    if not isinstance(model_section, Mapping):
        model_section = {}
    raw_params = model_section.get("parameters")
    candidate = config.models.get(config.model.name)
    baseline_params = (
        {str(key): value for key, value in raw_params.items()}
        if isinstance(raw_params, Mapping)
        else dict(candidate.params)
        if candidate is not None
        else {}
    )
    baseline_name = str(
        model_section.get("selected", config.model.name) or config.model.name
    )
    return OptimizationInputs(
        train_frame=train_frame,
        validation_frame=validation_frame,
        baseline_name=baseline_name,
        baseline_params=baseline_params,
        baseline_model=baseline_model,
        baseline_preprocessor=baseline_preprocessor,
        metadata=dict(metadata),
        metadata_path=artifact_paths["metadata"],
    )


def _baseline_validation(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    config: TrainingConfig,
    model: Any,
    preprocessor: ColumnTransformer,
) -> tuple[EvaluationResult, np.ndarray]:
    """Evaluate the saved baseline on validation rows and return its scores."""
    y_true = binary_targets(frame[spec.target_column], spec.positive_label)
    features = preprocessor.transform(prepare_features_frame(frame, spec))
    predictions = generate_predictions(model, features)
    if predictions.probabilities is None:
        msg = "The saved baseline model does not expose positive probabilities."
        raise OptimizationError(msg)
    evaluation = evaluate_predictions(
        y_true,
        predictions.labels,
        predictions.probabilities,
        model_name=config.model.name,
        split="validation",
    )
    return evaluation, np.asarray(predictions.probabilities, dtype=float)


def _optimized_validation(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    spec: DatasetSpec,
    config: TrainingConfig,
    params: Mapping[str, Any],
    *,
    preprocessor_factory: PreprocessorFactory,
    model_factory: ModelFactory | None,
) -> EvaluationResult:
    """Refit the optimized candidate on train rows and score validation rows."""
    optimization = config.optimization
    seed = optimization.seed
    active_model_factory = model_factory or (
        lambda name, candidate: create_model(name, candidate, seed=seed)
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor_factory(spec, config.features)),
            (
                "model",
                active_model_factory(optimization.model_type.value, params),
            ),
        ]
    )
    pipeline.fit(
        prepare_features_frame(train_frame, spec),
        binary_targets(train_frame[spec.target_column], spec.positive_label),
    )
    predictions = generate_predictions(
        pipeline, prepare_features_frame(validation_frame, spec)
    )
    return evaluate_predictions(
        binary_targets(validation_frame[spec.target_column], spec.positive_label),
        predictions.labels,
        predictions.probabilities,
        model_name=optimization.model_type.value,
        split="validation",
    )


def run_optimization_pipeline(
    config: TrainingConfig,
    spec: DatasetSpec,
    inputs: OptimizationInputs,
    *,
    output_dir: Path | None = None,
    resume: bool | None = None,
    preprocessor_factory: PreprocessorFactory = build_preprocessor,
    model_factory: ModelFactory | None = None,
) -> OptimizationResult:
    """Run the complete optimization, comparison, and threshold workflow.

    Only the supplied training rows enter cross-validation and candidate
    fitting. Only the supplied validation rows enter baseline comparison and
    threshold selection. No test data is requested.

    Args:
        config: Loaded training configuration.
        spec: Dataset contract.
        inputs: Saved baseline artifacts and train/validation rows.
        output_dir: Optional optimization output override.
        resume: Optional override for study resumption.
        preprocessor_factory: Injectable unfitted transformer factory.
        model_factory: Injectable unfitted classifier factory.

    Returns:
        Complete run result including all persisted artifact paths.

    Raises:
        OptimizationError: If configuration, optimization, or persistence fails.

    """
    optimization = config.optimization
    destination = ensure_directory(output_dir or optimization.output_dir)
    train_frame = _validated_frame(inputs.train_frame, spec)
    validation_frame = _validated_frame(inputs.validation_frame, spec)

    space = optimization.search_space.get(optimization.model_type.value, {})
    validate_search_space(optimization.model_type, space)

    baseline_validation, baseline_probabilities = _baseline_validation(
        validation_frame,
        spec,
        config,
        inputs.baseline_model,
        inputs.baseline_preprocessor,
    )
    seed = optimization.seed
    active_model_factory = model_factory or (
        lambda name, candidate: create_model(name, candidate, seed=seed)
    )

    def objective(trial: Any) -> float:
        """Score one trial with fold-local preprocessing on training rows."""
        suggested = suggest_parameters(trial, space)
        candidate_params = merge_parameters(inputs.baseline_params, suggested)
        cv_result = cross_validate_params(
            train_frame,
            spec,
            config.features,
            optimization.model_type,
            candidate_params,
            cv_folds=optimization.cv_folds,
            metric=optimization.metric,
            direction=optimization.direction,
            seed=seed,
            positive_label=spec.positive_label,
            preprocessor_factory=preprocessor_factory,
            model_factory=active_model_factory,
        )
        trial.set_user_attr("cv_result", cv_result.to_dict())
        return cv_result.objective_mean

    study, created, storage_path = create_or_load_study(
        optimization, resume=resume, output_dir=destination
    )
    study_result = run_optimization(
        study,
        objective,
        optimization,
        storage_path=storage_path,
        created=created,
        resumed=not created,
    )
    optimized_params = merge_parameters(
        inputs.baseline_params, study_result.best_params
    )

    optimized_validation = _optimized_validation(
        train_frame,
        validation_frame,
        spec,
        config,
        optimized_params,
        preprocessor_factory=preprocessor_factory,
        model_factory=model_factory,
    )
    threshold = analyze_thresholds(
        binary_targets(validation_frame[spec.target_column], spec.positive_label),
        baseline_probabilities,
        optimization.threshold,
        source_split="validation",
    )
    threshold_report = threshold.to_dict()

    plot_paths = dict(save_study_plots(study_result.study, destination))
    plot_paths.update(save_threshold_plots(threshold, destination))

    full_study_report = study_report(
        study_result,
        optimization,
        baseline_params=inputs.baseline_params,
        optimized_params=optimized_params,
    )
    best_report = best_parameters_report(
        study_result,
        optimization,
        baseline_params=inputs.baseline_params,
        optimized_params=optimized_params,
    )
    comparison = build_comparison_report(
        optimization,
        study_result,
        baseline_name=inputs.baseline_name,
        baseline_params=inputs.baseline_params,
        baseline_metrics=baseline_validation,
        optimized_params=optimized_params,
        optimized_metrics=optimized_validation,
        threshold_report=threshold_report,
    )
    artifacts = save_optimization_artifacts(
        destination,
        study_report=full_study_report,
        best_parameters=best_report,
        comparison=comparison,
        threshold_analysis=threshold_report,
        plot_paths=plot_paths,
    )

    cv_result = study_result.best_trial.user_attrs.get("cv_result") or {}
    metadata_payload: dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at": study_result.finished_at,
        "experiment": {
            "name": config.experiment_name,
            "seed": seed,
            "model_type": optimization.model_type.value,
            "study_name": study_result.study.study_name,
            "metric": optimization.metric.value,
            "configured_direction": optimization.direction.value,
            "objective_direction": study_result.objective_direction,
            "duration_seconds": study_result.duration_seconds,
            "completed_trials": study_result.completed_trials,
            "failed_trials": study_result.failed_trials,
        },
        "baseline_parameters": dict(inputs.baseline_params),
        "optimized_parameters": dict(optimized_params),
        "cross_validation": dict(cv_result),
        "validation_metrics": {
            "baseline": dict(baseline_validation.metrics),
            "optimized_candidate": dict(optimized_validation.metrics),
        },
        "selected_threshold": threshold.selected.threshold,
        "selected_threshold_metrics": threshold.selected.to_dict(),
        "threshold_0_5_metrics": threshold.baseline.to_dict(),
        "optimized_candidate_deployed": False,
        "test_split_evaluated": False,
    }
    updated_metadata = merge_optimization_metadata(
        inputs.metadata,
        optimization=metadata_payload,
        threshold_report=threshold_report,
    )
    artifacts["model_metadata"] = write_json_artifact(
        inputs.metadata_path, updated_metadata
    )

    result = OptimizationResult(
        config=config,
        study_result=study_result,
        baseline_parameters=dict(inputs.baseline_params),
        optimized_parameters=optimized_params,
        baseline_validation=baseline_validation,
        optimized_validation=optimized_validation,
        threshold=threshold,
        comparison=comparison,
        metadata=updated_metadata,
        artifacts=artifacts,
    )
    logger.info(
        "Optimization workflow complete: threshold=%.2f selected_source=%s",
        threshold.selected.threshold,
        threshold.source_split,
    )
    return result


def run_optimization_workflow(
    config: TrainingConfig | None = None,
    spec: DatasetSpec | None = None,
    *,
    data_config: str | Path | None = None,
    train_config: str | Path | None = None,
    output_dir: Path | None = None,
    resume: bool | None = None,
) -> OptimizationResult:
    """Load configuration and saved artifacts, then run optimization.

    Args:
        config: Optional preloaded training configuration.
        spec: Optional preloaded dataset specification.
        data_config: Optional data YAML path.
        train_config: Optional training YAML path.
        output_dir: Optional optimization output override.
        resume: Optional override for study resumption.

    Returns:
        Complete optimization run result.

    """
    active_config = config or load_training_config(data_config, train_config)
    active_spec = spec or load_dataset_spec(data_config)
    inputs = load_optimization_inputs(active_config, active_spec)
    return run_optimization_pipeline(
        active_config,
        active_spec,
        inputs,
        output_dir=output_dir,
        resume=resume,
    )

"""Optuna study lifecycle, execution, and serialization helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from smartpredict.config.schema import OptimizationConfig
from smartpredict.core.exceptions import OptimizationError
from smartpredict.core.logging import get_logger
from smartpredict.optimization.search_space import describe_search_space

logger = get_logger(__name__)

Objective = Callable[[Any], float]


def import_optuna() -> Any:
    """Import Optuna lazily and return the module.

    Returns:
        The installed ``optuna`` module.

    Raises:
        OptimizationError: If Optuna is not installed.

    """
    try:
        return import_module("optuna")
    except ImportError as exc:
        msg = "Optuna is required for model optimization. Install 'optuna>=4.0'."
        raise OptimizationError(msg) from exc


@dataclass(frozen=True)
class StudyResult:
    """Execution summary for one completed or resumed Optuna study."""

    study: Any
    storage_path: Path
    created: bool
    resumed: bool
    objective_direction: str
    configured_metric: str
    configured_direction: str
    best_value: float | None
    best_params: dict[str, Any]
    completed_trials: int
    failed_trials: int
    pruned_trials: int
    total_trials: int
    duration_seconds: float
    started_at: str
    finished_at: str

    @property
    def best_trial(self) -> Any:
        """Return the best completed trial when one exists."""
        if self.best_value is None:
            return None
        return self.study.best_trial

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe study summary."""
        return {
            "schema_version": "1.0",
            "study_name": self.study.study_name,
            "storage_path": str(self.storage_path),
            "created": self.created,
            "resumed": self.resumed,
            "objective_direction": self.objective_direction,
            "configured_metric": self.configured_metric,
            "configured_direction": self.configured_direction,
            "best_value": self.best_value,
            "best_parameters": dict(self.best_params),
            "counts": {
                "completed": self.completed_trials,
                "failed": self.failed_trials,
                "pruned": self.pruned_trials,
                "total": self.total_trials,
            },
            "duration_seconds": self.duration_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


def create_or_load_study(
    config: OptimizationConfig,
    *,
    resume: bool | None = None,
    output_dir: Path | None = None,
) -> tuple[Any, bool, Path]:
    """Create a fresh SQLite study or load an existing one for resumption.

    Args:
        config: Typed optimization configuration.
        resume: Override for ``config.resume``.
        output_dir: Optional destination override used by tests and the CLI.

    Returns:
        Tuple of study, whether it was newly created, and storage path.

    Raises:
        OptimizationError: If an existing study is found without ``resume`` or
            if Optuna cannot open the storage.

    """
    optuna = import_optuna()
    directory = (output_dir or config.output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    storage_path = directory / config.storage_file
    storage = f"sqlite:///{storage_path.as_posix()}"
    should_resume = config.resume if resume is None else resume

    try:
        sampler = optuna.samplers.TPESampler(seed=config.seed)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not configure Optuna sampler or pruner: {exc}"
        raise OptimizationError(msg) from exc

    if storage_path.exists():
        if not should_resume:
            msg = (
                f"An Optuna study already exists at {storage_path}. "
                "Pass --resume to continue it or choose a different output directory."
            )
            raise OptimizationError(msg)
        try:
            study = optuna.load_study(
                study_name=config.study_name,
                storage=storage,
                sampler=sampler,
                pruner=pruner,
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not resume Optuna study '{config.study_name}': {exc}"
            raise OptimizationError(msg) from exc
        logger.info(
            "Resumed Optuna study '%s' with %s prior trials",
            config.study_name,
            len(study.trials),
        )
        return study, False, storage_path

    try:
        study = optuna.create_study(
            study_name=config.study_name,
            storage=storage,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not create Optuna study '{config.study_name}': {exc}"
        raise OptimizationError(msg) from exc
    logger.info("Created Optuna study '%s' at %s", config.study_name, storage_path)
    return study, True, storage_path


def _state_name(trial: Any) -> str:
    """Return a lower-case state name for a trial."""
    return str(getattr(trial.state, "name", trial.state)).lower()


def _count_states(study: Any) -> tuple[int, int, int]:
    """Count complete, failed, and pruned trials in a study."""
    states = [_state_name(trial) for trial in study.trials]
    return (
        states.count("complete"),
        states.count("fail"),
        states.count("pruned"),
    )


def run_optimization(
    study: Any,
    objective: Objective,
    config: OptimizationConfig,
    *,
    storage_path: Path,
    created: bool,
    resumed: bool,
) -> StudyResult:
    """Run trials while converting configured minimization to maximization.

    Failed trials are caught and retained in the study rather than aborting the
    experiment. The objective itself must return a maximization score.

    Args:
        study: Optuna study to optimize.
        objective: Callable accepting a trial and returning a score.
        config: Typed optimization configuration.
        storage_path: SQLite storage location for reporting.
        created: Whether the study was newly created.
        resumed: Whether the study was loaded from storage.

    Returns:
        Structured study execution result.

    Raises:
        OptimizationError: If no trial completed successfully.

    """
    started_at = datetime.now(tz=UTC).isoformat()
    started = perf_counter()
    try:
        study.optimize(
            objective,
            n_trials=config.n_trials,
            timeout=config.timeout_seconds,
            catch=(Exception,),
            gc_after_trial=True,
            show_progress_bar=False,
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Optuna optimization stopped unexpectedly: {exc}"
        raise OptimizationError(msg) from exc
    duration = round(perf_counter() - started, 6)
    finished_at = datetime.now(tz=UTC).isoformat()

    completed, failed, pruned = _count_states(study)
    if completed == 0:
        msg = (
            f"Optuna study '{study.study_name}' finished without any completed trial; "
            f"{failed} trials failed and {pruned} were pruned."
        )
        raise OptimizationError(msg)

    best_trial = study.best_trial
    best_params = {str(key): value for key, value in best_trial.params.items()}
    result = StudyResult(
        study=study,
        storage_path=storage_path,
        created=created,
        resumed=resumed,
        objective_direction="maximize",
        configured_metric=config.metric.value,
        configured_direction=config.direction.value,
        best_value=float(best_trial.value),
        best_params=best_params,
        completed_trials=completed,
        failed_trials=failed,
        pruned_trials=pruned,
        total_trials=len(study.trials),
        duration_seconds=duration,
        started_at=started_at,
        finished_at=finished_at,
    )
    logger.info(
        "Optimization finished: completed=%s failed=%s pruned=%s "
        "best=%.6f seconds=%.2f",
        completed,
        failed,
        pruned,
        result.best_value,
        duration,
    )
    return result


def _fallback_plot(path: Path, message: str) -> None:
    """Write a readable placeholder when Optuna plotting is unavailable."""
    figure, axes = plt.subplots(figsize=(8, 5))
    axes.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        wrap=True,
        transform=axes.transAxes,
    )
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_study_plots(study: Any, output_dir: Path) -> dict[str, str]:
    """Persist Optuna history and parameter-importance plots.

    Args:
        study: Completed Optuna study.
        output_dir: Destination directory.

    Returns:
        Mapping from plot label to written path.

    """
    output_dir.mkdir(parents=True, exist_ok=True)
    optuna = import_optuna()
    history_path = output_dir / "optimization_history.png"
    importance_path = output_dir / "parameter_importance.png"

    try:
        axes = optuna.visualization.matplotlib.plot_optimization_history(study)
        figure = axes.get_figure() if hasattr(axes, "get_figure") else axes
        figure.savefig(history_path, dpi=150)
        plt.close(figure)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render optimization history plot: %s", exc)
        _fallback_plot(history_path, "Optimization history is unavailable.")

    try:
        axes = optuna.visualization.matplotlib.plot_param_importances(study)
        figure = axes.get_figure() if hasattr(axes, "get_figure") else axes
        figure.savefig(importance_path, dpi=150)
        plt.close(figure)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not render parameter importance plot: %s", exc)
        _fallback_plot(
            importance_path, "Parameter importance needs multiple completed trials."
        )

    return {
        "optimization_history": str(history_path),
        "parameter_importance": str(importance_path),
    }


def trials_to_dict(study: Any) -> list[dict[str, Any]]:
    """Return a JSON-safe snapshot of all trials in a study."""
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        rows.append(
            {
                "number": trial.number,
                "state": _state_name(trial),
                "value": trial.value,
                "params": {str(key): value for key, value in trial.params.items()},
                "user_attrs": {
                    str(key): value for key, value in trial.user_attrs.items()
                },
            }
        )
    return rows


def study_report(
    result: StudyResult,
    config: OptimizationConfig,
    *,
    baseline_params: dict[str, Any],
    optimized_params: dict[str, Any],
) -> dict[str, Any]:
    """Build the full ``optuna_study.json`` payload."""
    report = result.to_dict()
    report["study_name"] = result.study.study_name
    report["search_space"] = describe_search_space(
        config.search_space.get(config.model_type.value, {})
    )
    report["baseline_parameters"] = dict(baseline_params)
    report["optimized_parameters"] = dict(optimized_params)
    report["cv_result"] = result.best_trial.user_attrs.get("cv_result")
    report["trials"] = trials_to_dict(result.study)
    report["data_access"] = {
        "training_split": str(config.data_directory / config.train_file),
        "validation_split": str(config.data_directory / config.validation_file),
        "test_split": None,
        "note": "The test split was not loaded during optimization.",
    }
    return report


def best_parameters_report(
    result: StudyResult,
    config: OptimizationConfig,
    *,
    baseline_params: dict[str, Any],
    optimized_params: dict[str, Any],
) -> dict[str, Any]:
    """Build the ``best_parameters.json`` payload."""
    cv_result = result.best_trial.user_attrs.get("cv_result") or {}
    return {
        "schema_version": "1.0",
        "generated_at": result.finished_at,
        "study_name": result.study.study_name,
        "model_type": config.model_type.value,
        "metric": config.metric.value,
        "configured_direction": config.direction.value,
        "objective_direction": result.objective_direction,
        "best_objective_value": result.best_value,
        "best_raw_cv_metric": cv_result.get("metric_mean"),
        "baseline_parameters": dict(baseline_params),
        "searched_parameters": dict(result.best_params),
        "optimized_parameters": dict(optimized_params),
        "completed_trials": result.completed_trials,
        "failed_trials": result.failed_trials,
    }

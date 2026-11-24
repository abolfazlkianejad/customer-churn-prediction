"""Synthetic tests for Optuna study lifecycle and failure handling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from smartpredict.config.schema import TrainingConfig
from smartpredict.core.exceptions import OptimizationError
from smartpredict.optimization.study import (
    create_or_load_study,
    run_optimization,
)


def _study_config(
    training_config: TrainingConfig, directory: Path, name: str, n_trials: int
) -> TrainingConfig:
    """Return a fast study-only configuration using synthetic artifacts."""
    optimization = replace(
        training_config.optimization,
        n_trials=n_trials,
        timeout_seconds=None,
        output_dir=directory,
        study_name=name,
        seed=42,
        resume=False,
    )
    return replace(training_config, optimization=optimization)


def _quadratic(trial: Any) -> float:
    """Return a deterministic maximization objective for synthetic trials."""
    value = trial.suggest_float("value", -5.0, 5.0)
    return -((value - 2.0) ** 2)


def test_study_creation_persistence_and_resume(
    training_config: TrainingConfig, tmp_path: Path
) -> None:
    """A SQLite study is created, persisted, and safely resumed."""
    config = _study_config(training_config, tmp_path, "creation_test", 4)
    study, created, storage_path = create_or_load_study(config.optimization)
    assert created is True
    assert storage_path.is_file()

    result = run_optimization(
        study,
        _quadratic,
        config.optimization,
        storage_path=storage_path,
        created=created,
        resumed=False,
    )
    assert result.completed_trials == 4

    resumed, created_again, _ = create_or_load_study(
        config.optimization, resume=True
    )
    assert created_again is False
    assert len(resumed.trials) == 4

    with pytest.raises(OptimizationError, match="--resume"):
        create_or_load_study(config.optimization, resume=False)


def test_studies_are_reproducible_with_same_seed(
    training_config: TrainingConfig, tmp_path: Path
) -> None:
    """Two seeded studies propose identical synthetic parameter sequences."""
    sequences: list[list[dict[str, Any]]] = []
    best_values: list[float | None] = []
    for index in range(2):
        directory = tmp_path / f"run_{index}"
        config = _study_config(training_config, directory, "repro_test", 5)
        study, created, storage_path = create_or_load_study(config.optimization)
        result = run_optimization(
            study,
            _quadratic,
            config.optimization,
            storage_path=storage_path,
            created=created,
            resumed=False,
        )
        sequences.append([dict(trial.params) for trial in study.trials])
        best_values.append(result.best_value)

    assert sequences[0] == sequences[1]
    assert best_values[0] == best_values[1]


def test_failed_trials_do_not_abort_study(
    training_config: TrainingConfig, tmp_path: Path
) -> None:
    """Caught trial exceptions are counted while successful trials complete."""

    def flaky_objective(trial: Any) -> float:
        """Fail every other synthetic trial."""
        value = trial.suggest_float("value", -5.0, 5.0)
        if trial.number % 2 == 0:
            msg = "synthetic trial failure"
            raise ValueError(msg)
        return -((value - 2.0) ** 2)

    config = _study_config(training_config, tmp_path, "failure_test", 4)
    study, created, storage_path = create_or_load_study(config.optimization)
    result = run_optimization(
        study,
        flaky_objective,
        config.optimization,
        storage_path=storage_path,
        created=created,
        resumed=False,
    )

    assert result.completed_trials == 2
    assert result.failed_trials == 2
    assert result.total_trials == 4
    assert result.best_trial is not None


def test_all_failed_trials_raise_domain_error(
    training_config: TrainingConfig, tmp_path: Path
) -> None:
    """A study with no completed trial fails explicitly instead of silently."""

    def always_fails(trial: Any) -> float:
        """Force every synthetic trial to fail."""
        trial.suggest_float("value", -1.0, 1.0)
        msg = "synthetic failure"
        raise ValueError(msg)

    config = _study_config(training_config, tmp_path, "all_failed_test", 2)
    study, created, storage_path = create_or_load_study(config.optimization)
    with pytest.raises(OptimizationError, match="without any completed trial"):
        run_optimization(
            study,
            always_fails,
            config.optimization,
            storage_path=storage_path,
            created=created,
            resumed=False,
        )

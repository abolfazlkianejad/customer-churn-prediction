"""Tests for logging, paths, seeds, and exceptions."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pytest

from smartpredict.config.settings import Settings
from smartpredict.core import exceptions, paths, seed
from smartpredict.core import logging as log_utils


def test_project_root_contains_pyproject(project_root: Path) -> None:
    """Path helpers locate the actual project root."""
    assert paths.get_project_root() == project_root
    assert (project_root / "pyproject.toml").is_file()


def test_resolve_path_handles_relative_and_absolute(project_root: Path) -> None:
    """Relative paths are anchored at the project root."""
    expected = project_root / "configs" / "data.yaml"
    assert paths.resolve_path("configs/data.yaml") == expected
    absolute = project_root / "artifacts"
    assert paths.resolve_path(absolute) == absolute


def test_ensure_directory_creates_missing_directories(tmp_path: Path) -> None:
    """Nested output directories are created on demand."""
    created = paths.ensure_directory("nested/output", base_dir=tmp_path)
    assert created.is_dir()


def test_set_global_seed_is_reproducible() -> None:
    """Seeding produces identical Python and NumPy random streams."""
    seed.set_global_seed(17)
    python_first = random.random()
    numpy_first = float(np.random.random())
    seed.set_global_seed(17)
    assert random.random() == python_first
    assert float(np.random.random()) == numpy_first


def test_set_global_seed_rejects_negative_values() -> None:
    """Negative seeds are invalid."""
    with pytest.raises(ValueError, match="non-negative"):
        seed.set_global_seed(-1)


def test_seed_from_settings(settings: Settings) -> None:
    """The configured seed is applied to the global generators."""
    configured = Settings(_env_file=None, random_seed=99)
    seed.seed_from_settings(configured)
    assert random.random() == random.Random(99).random()
    assert settings.random_seed == 42


def test_configure_logging_sets_root_level() -> None:
    """Logging configuration applies the configured level."""
    active = Settings(_env_file=None, log_level="DEBUG", log_to_file=False)
    log_utils.configure_logging(active)
    assert logging.getLogger().level == logging.DEBUG


def test_json_formatter_emits_structured_records() -> None:
    """JSON logs include level, logger, and message fields."""
    record = logging.LogRecord(
        name="smartpredict.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="churn check",
        args=(),
        exc_info=None,
    )
    payload = json.loads(log_utils.JsonFormatter().format(record))
    assert payload["level"] == "INFO"
    assert payload["logger"] == "smartpredict.test"
    assert payload["message"] == "churn check"


def test_exception_hierarchy() -> None:
    """Domain exceptions share a common base class."""
    assert issubclass(exceptions.DataValidationError, exceptions.DataError)
    assert issubclass(exceptions.ModelNotFoundError, exceptions.ModelError)
    assert issubclass(exceptions.ConfigurationError, exceptions.SmartPredictError)

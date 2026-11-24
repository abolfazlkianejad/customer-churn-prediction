"""Tests for the training command-line interface."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from smartpredict.cli.train import build_parser, main


def write_yaml(path: Path, content: dict[str, object]) -> Path:
    """Write a YAML mapping to a temporary path."""
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_parser_exposes_training_options() -> None:
    """The CLI exposes configuration, artifact, and overwrite options."""
    parser = build_parser()
    actions = {action.dest for action in parser._actions}
    assert {
        "data_config",
        "train_config",
        "models_dir",
        "reports_dir",
        "plots_dir",
        "overwrite",
    }.issubset(actions)


def test_help_exits_successfully() -> None:
    """``--help`` prints usage and exits with code zero."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_cli_trains_with_synthetic_config(
    fast_training_configs: dict[str, Path],
) -> None:
    """The CLI executes the full workflow and returns zero on success."""
    exit_code = main(
        [
            "--data-config",
            str(fast_training_configs["data_config"]),
            "--train-config",
            str(fast_training_configs["train_config"]),
            "--models-dir",
            str(fast_training_configs["models_dir"]),
            "--reports-dir",
            str(fast_training_configs["reports_dir"]),
            "--plots-dir",
            str(fast_training_configs["plots_dir"]),
        ]
    )
    assert exit_code == 0
    assert (fast_training_configs["models_dir"] / "best_model.joblib").is_file()
    assert (
        fast_training_configs["reports_dir"] / "final_test_report.json"
    ).is_file()


def test_cli_returns_nonzero_for_missing_dataset(
    tmp_path: Path, project_root: Path, fast_training_configs: dict[str, Path]
) -> None:
    """A missing configured dataset produces a non-zero exit code."""
    source = yaml.safe_load(
        (project_root / "configs" / "data.yaml").read_text(encoding="utf-8")
    )
    source["dataset"]["raw_path"] = str(tmp_path / "absent.csv")
    data_config = write_yaml(tmp_path / "missing_data.yaml", source)
    exit_code = main(
        [
            "--data-config",
            str(data_config),
            "--train-config",
            str(fast_training_configs["train_config"]),
            "--models-dir",
            str(tmp_path / "models"),
            "--reports-dir",
            str(tmp_path / "reports"),
            "--plots-dir",
            str(tmp_path / "plots"),
        ]
    )
    assert exit_code == 1

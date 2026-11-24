"""Tests for the explanation command-line interface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from smartpredict.cli.explain import build_parser, main


def test_parser_exposes_explanation_options() -> None:
    """The CLI exposes artifact, sampling, and overwrite options."""
    actions = {action.dest for action in build_parser()._actions}
    assert {
        "data_config",
        "train_config",
        "models_dir",
        "reports_dir",
        "output_dir",
        "sample_size",
        "local_examples",
        "overwrite",
    }.issubset(actions)


def test_help_exits_successfully() -> None:
    """``--help`` prints usage and exits with code zero."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_passes_overrides_to_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """CLI options are forwarded to the explanation service."""
    captured: dict[str, Any] = {}

    def fake_generate(**kwargs: Any) -> SimpleNamespace:
        """Capture CLI arguments and return a minimal result object."""
        captured.update(kwargs)
        return SimpleNamespace(
            selected_model="xgboost",
            global_importance=tmp_path / "global.json",
            summary_plot=tmp_path / "summary.png",
            bar_plot=tmp_path / "bar.png",
            local_report=tmp_path / "local.json",
            local_plots=(),
        )

    monkeypatch.setattr(
        "smartpredict.cli.explain.generate_explanations", fake_generate
    )
    code = main(
        [
            "--models-dir",
            str(tmp_path / "models"),
            "--output-dir",
            str(tmp_path / "explain"),
            "--sample-size",
            "7",
            "--local-examples",
            "2",
            "--overwrite",
        ]
    )

    assert code == 0
    assert captured["sample_size"] == 7
    assert captured["local_examples"] == 2
    assert captured["overwrite"] is True
    assert captured["output_dir"] == tmp_path / "explain"

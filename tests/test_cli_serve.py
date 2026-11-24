"""Tests for the API server command-line interface."""

from __future__ import annotations

import pytest

from smartpredict.cli.serve import build_parser, main


def test_parser_uses_configured_defaults() -> None:
    """Host and port defaults come from application settings."""
    args = build_parser().parse_args([])
    assert args.host
    assert 1 <= args.port <= 65535
    assert args.reload is False


def test_help_exits_successfully() -> None:
    """``--help`` exits without starting uvicorn."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0

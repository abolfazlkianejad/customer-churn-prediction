"""Tests for the dependency-free container health probe."""

from __future__ import annotations

import json
from typing import Any

import pytest

from smartpredict.deployment import healthcheck


class _FakeResponse:
    """Minimal urlopen response used by probe tests."""

    def __init__(self, payload: dict[str, Any], status: int = 200) -> None:
        self.payload = payload
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_parser_uses_configured_health_endpoint() -> None:
    """The probe defaults to the configured API prefix."""
    args = healthcheck.build_parser().parse_args([])
    assert args.url.endswith("/api/v1/health")
    assert args.timeout > 0


def test_probe_accepts_ready_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ready health payload returns success."""
    payload = {
        "status": "ok",
        "artifacts_ready": True,
        "model_loaded": True,
        "preprocessor_loaded": True,
        "model_name": "xgboost",
    }
    monkeypatch.setattr(
        healthcheck, "urlopen", lambda url, timeout=None: _FakeResponse(payload)
    )

    assert healthcheck.probe("http://test/health", timeout=1.0) is True


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "ok", "artifacts_ready": False},
        {"status": "ok"},
        {"status": "degraded", "artifacts_ready": True},
    ],
)
def test_probe_rejects_unready_response(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]
) -> None:
    """Incomplete or unhealthy payloads return failure."""
    monkeypatch.setattr(
        healthcheck, "urlopen", lambda url, timeout=None: _FakeResponse(payload)
    )

    assert healthcheck.probe("http://test/health", timeout=1.0) is False


def test_probe_handles_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Connection failures are converted into a boolean result."""

    def fail(*_: Any, **__: Any) -> None:
        raise OSError("offline")

    monkeypatch.setattr(healthcheck, "urlopen", fail)

    assert healthcheck.probe("http://test/health", timeout=1.0) is False


def test_main_returns_process_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    """The command line wrapper maps probe results to 0 and 1."""
    payload = {"status": "ok", "artifacts_ready": True}
    monkeypatch.setattr(
        healthcheck, "urlopen", lambda url, timeout=None: _FakeResponse(payload)
    )

    assert healthcheck.main(["--url", "http://test/health", "--timeout", "1"]) == 0

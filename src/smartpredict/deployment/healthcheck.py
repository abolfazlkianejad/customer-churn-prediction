"""Dependency-free HTTP probe used by container health checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from urllib.error import URLError
from urllib.request import urlopen

from smartpredict.config.settings import get_settings
from smartpredict.core.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the health-check command."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m smartpredict.deployment.healthcheck",
        description="Probe the SmartPredict readiness endpoint.",
    )
    parser.add_argument(
        "--url",
        default=settings.healthcheck_target(),
        help="Health endpoint URL (defaults to the configured API prefix).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=settings.healthcheck_timeout_seconds,
        help="Request timeout in seconds.",
    )
    return parser


def probe(url: str, timeout: float = 3.0) -> bool:
    """Return ``True`` when the readiness endpoint reports loaded artifacts.

    Args:
        url: Fully qualified health endpoint URL.
        timeout: Maximum time to wait for a response.

    Returns:
        ``True`` for a ready service, otherwise ``False``.

    """
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310
            if getattr(response, "status", 200) != 200:  # noqa: PLR2004
                return False
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, TimeoutError, ValueError) as exc:
        logger.warning("Health probe failed: %s", type(exc).__name__)
        return False
    return (
        isinstance(payload, dict)
        and payload.get("status") == "ok"
        and payload.get("artifacts_ready") is True
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one health probe and return a process exit code."""
    args = build_parser().parse_args(argv)
    return 0 if probe(args.url, args.timeout) else 1


if __name__ == "__main__":
    raise SystemExit(main())

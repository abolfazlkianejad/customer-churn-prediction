"""Command-line entry point for the FastAPI inference service."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from smartpredict.api.app import create_app
from smartpredict.config.settings import get_settings
from smartpredict.core.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the API server."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="python -m smartpredict.cli.serve",
        description="Serve the saved churn model over HTTP.",
    )
    parser.add_argument("--host", default=settings.api_host, help="Bind address.")
    parser.add_argument(
        "--port", type=int, default=settings.api_port, help="Bind port."
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Reload the process when source files change (development only).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the API with uvicorn and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()
    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level=get_settings().log_level.lower(),
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

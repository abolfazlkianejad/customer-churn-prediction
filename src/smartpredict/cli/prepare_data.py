"""Command-line entry point for the data preparation pipeline."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from smartpredict.core.exceptions import SmartPredictError
from smartpredict.core.logging import configure_logging, get_logger
from smartpredict.data.pipeline import run_data_pipeline

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for data preparation.

    Returns:
        Configured argument parser.

    """
    parser = argparse.ArgumentParser(
        prog="python -m smartpredict.cli.prepare_data",
        description="Load, validate, and split the configured churn dataset.",
    )
    parser.add_argument(
        "--data-config",
        help="Path to data.yaml (defaults to the configured configs directory).",
    )
    parser.add_argument(
        "--train-config",
        help="Path to train.yaml (defaults to the configured configs directory).",
    )
    parser.add_argument("--output-dir", help="Directory for processed split CSV files.")
    parser.add_argument("--reports-dir", help="Directory for JSON reports.")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Write reports even when validation errors are present.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data pipeline from the command line.

    Args:
        argv: Optional argument sequence; defaults to ``sys.argv``.

    Returns:
        Process exit code (0 on success, 1 on a domain error).

    """
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        result = run_data_pipeline(
            data_config=args.data_config,
            train_config=args.train_config,
            output_dir=args.output_dir,
            reports_dir=args.reports_dir,
            strict=not args.allow_invalid,
        )
    except SmartPredictError as exc:
        logger.error("Data preparation failed: %s", exc)
        return 1

    logger.info(
        "Split sizes: train=%s validation=%s test=%s",
        result.split.metadata.sizes["train"],
        result.split.metadata.sizes["validation"],
        result.split.metadata.sizes["test"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

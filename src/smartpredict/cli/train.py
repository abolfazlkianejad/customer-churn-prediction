"""Command-line entry point for model training and evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from smartpredict.core.exceptions import SmartPredictError
from smartpredict.core.logging import configure_logging, get_logger
from smartpredict.models.pipeline import run_training_workflow

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for model training."""
    parser = argparse.ArgumentParser(
        prog="python -m smartpredict.cli.train",
        description="Train, compare, select, and evaluate churn models.",
    )
    parser.add_argument(
        "--data-config",
        help="Path to data.yaml (defaults to the configured configs directory).",
    )
    parser.add_argument(
        "--train-config",
        help="Path to train.yaml (defaults to the configured configs directory).",
    )
    parser.add_argument("--models-dir", type=Path, help="Output directory for models.")
    parser.add_argument(
        "--reports-dir", type=Path, help="Output directory for reports."
    )
    parser.add_argument("--plots-dir", type=Path, help="Output directory for plots.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifacts with the names configured in train.yaml.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the training workflow from the command line.

    Args:
        argv: Optional argument sequence; defaults to ``sys.argv``.

    Returns:
        Process exit code (0 on success, 1 on a domain failure).

    """
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        result = run_training_workflow(
            data_config=args.data_config,
            train_config=args.train_config,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            plots_dir=args.plots_dir,
            overwrite=args.overwrite,
        )
    except SmartPredictError as exc:
        logger.error("Training workflow failed: %s", exc)
        return 1

    logger.info(
        "Selected model: %s | primary validation metric: %s | test %s: %s",
        result.comparison.selected_model,
        result.comparison.primary_metric,
        result.comparison.primary_metric,
        result.final_test.result.metrics.get(result.comparison.primary_metric),
    )
    logger.info("Artifacts written: %s", sorted(result.artifact_paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

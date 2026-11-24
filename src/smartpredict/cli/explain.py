"""Command-line entry point for SHAP model explanations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from smartpredict.core.exceptions import SmartPredictError
from smartpredict.core.logging import configure_logging, get_logger
from smartpredict.explainability.service import generate_explanations

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for explanation generation."""
    parser = argparse.ArgumentParser(
        prog="python -m smartpredict.cli.explain",
        description="Explain the saved churn model with SHAP; no retraining occurs.",
    )
    parser.add_argument(
        "--data-config",
        help="Path to data.yaml (defaults to the configured configs directory).",
    )
    parser.add_argument(
        "--train-config",
        help="Path to train.yaml (defaults to the configured configs directory).",
    )
    parser.add_argument(
        "--models-dir", type=Path, help="Directory with saved model artifacts."
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Directory containing model_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for explanation artifacts (defaults to train.yaml).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Number of validation rows to explain (defaults to train.yaml).",
    )
    parser.add_argument(
        "--local-examples",
        type=int,
        help="Number of individual customer explanations to generate.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing explanation artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate explanations from saved artifacts and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        result = generate_explanations(
            data_config=args.data_config,
            train_config=args.train_config,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            output_dir=args.output_dir,
            sample_size=args.sample_size,
            local_examples=args.local_examples,
            overwrite=args.overwrite,
        )
    except SmartPredictError as exc:
        logger.error("Explanation generation failed: %s", exc)
        return 1

    logger.info("Selected model: %s", result.selected_model)
    logger.info("Global importance: %s", result.global_importance)
    logger.info("Summary plot: %s", result.summary_plot)
    logger.info("Bar plot: %s", result.bar_plot)
    logger.info("Local explanations: %s", result.local_report)
    for path in result.local_plots:
        logger.info("Local plot: %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

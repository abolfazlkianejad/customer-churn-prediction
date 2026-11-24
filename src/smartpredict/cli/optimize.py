"""Command-line entry point for Optuna optimization and threshold analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from smartpredict.config.loader import load_training_config
from smartpredict.config.schema import (
    MetricName,
    OptimizationDirection,
    SupportedModel,
)
from smartpredict.core.exceptions import SmartPredictError
from smartpredict.core.logging import configure_logging, get_logger
from smartpredict.optimization.pipeline import run_optimization_workflow

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for model optimization."""
    parser = argparse.ArgumentParser(
        prog="python -m smartpredict.cli.optimize",
        description=(
            "Optimize the configured classifier with Optuna using training-only "
            "cross-validation and validation-only threshold analysis."
        ),
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
        "--output-dir",
        type=Path,
        help="Directory for optimization artifacts (overrides train.yaml).",
    )
    parser.add_argument("--n-trials", type=int, help="Number of Optuna trials.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="Maximum optimization duration in seconds.",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducible trials.")
    parser.add_argument(
        "--metric",
        choices=[metric.value for metric in MetricName],
        help="Metric used for cross-validation reporting.",
    )
    parser.add_argument(
        "--direction",
        choices=[direction.value for direction in OptimizationDirection],
        help="Direction of the configured metric.",
    )
    parser.add_argument(
        "--model-type",
        choices=[model.value for model in SupportedModel],
        help="Model family to optimize.",
    )
    parser.add_argument(
        "--cv-folds", type=int, help="Number of stratified cross-validation folds."
    )
    parser.add_argument(
        "--study-name", help="Optuna study name for SQLite persistence."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the persisted study instead of requiring a fresh study.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the optimization workflow from command-line arguments.

    Args:
        argv: Optional argument sequence, mainly used by tests.

    Returns:
        Process exit code (0 on success, 1 on a domain failure).

    """
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        config = load_training_config(args.data_config, args.train_config)
        optimization = config.optimization
        if args.n_trials is not None:
            optimization = replace(optimization, n_trials=args.n_trials)
        if args.timeout_seconds is not None:
            optimization = replace(optimization, timeout_seconds=args.timeout_seconds)
        if args.seed is not None:
            optimization = replace(optimization, seed=args.seed)
        if args.metric is not None:
            optimization = replace(optimization, metric=MetricName(args.metric))
        if args.direction is not None:
            optimization = replace(
                optimization, direction=OptimizationDirection(args.direction)
            )
        if args.model_type is not None:
            optimization = replace(
                optimization, model_type=SupportedModel(args.model_type)
            )
        if args.cv_folds is not None:
            optimization = replace(optimization, cv_folds=args.cv_folds)
        if args.study_name is not None:
            optimization = replace(optimization, study_name=args.study_name)
        config = replace(config, optimization=optimization)

        result = run_optimization_workflow(
            config,
            data_config=args.data_config,
            train_config=args.train_config,
            output_dir=args.output_dir,
            resume=True if args.resume else None,
        )
    except SmartPredictError as exc:
        logger.error("Optimization workflow failed: %s", exc)
        return 1

    logger.info("Optimization summary:\n%s", json.dumps(result.summary(), indent=2))
    logger.info("Artifacts written: %s", sorted(result.artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

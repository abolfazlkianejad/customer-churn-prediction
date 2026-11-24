"""Smoke-test saved model artifacts on raw customer records."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import joblib
import pandas as pd

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.core.exceptions import SmartPredictError
from smartpredict.core.logging import configure_logging, get_logger
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.inference import InferenceResult, predict_from_raw

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the inference smoke test."""
    parser = argparse.ArgumentParser(
        prog="python scripts/inference_smoke.py",
        description="Load saved artifacts and predict on raw customer records.",
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="Optional CSV of raw records; defaults to the first configured rows.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of configured raw rows to use when --records is omitted.",
    )
    return parser


def load_records(path: Path | None, limit: int) -> pd.DataFrame:
    """Load raw records without changing their string representation."""
    spec = load_dataset_spec()
    source = path or spec.raw_path
    records = pd.read_csv(source, dtype=str, keep_default_na=False)
    if limit < 1:
        msg = "--limit must be at least 1."
        raise ValueError(msg)
    return records.head(limit)


def run_smoke(records: pd.DataFrame | None = None, limit: int = 3) -> InferenceResult:
    """Load artifacts from their configured paths and run inference once."""
    spec = load_dataset_spec()
    config = load_training_config()
    paths = build_artifact_paths(config.artifacts)
    model = joblib.load(paths["best_model"])
    preprocessor = joblib.load(paths["preprocessor"])
    raw = records if records is not None else load_records(None, limit)
    return predict_from_raw(raw, spec, preprocessor, model)


def main(argv: Sequence[str] | None = None) -> int:
    """Print a compact inference smoke-test summary and return an exit code."""
    args = build_parser().parse_args(argv)
    configure_logging()
    try:
        result = run_smoke(
            records=load_records(args.records, args.limit)
            if args.records is not None
            else None,
            limit=args.limit,
        )
    except (SmartPredictError, OSError, ValueError) as exc:
        logger.error("Inference smoke test failed: %s", exc)
        return 1

    print(f"records={result.n_records} features={len(result.feature_names)}")
    print(f"labels={result.labels.tolist()}")
    if result.probabilities is not None:
        print(f"probabilities={[round(float(p), 6) for p in result.probabilities]}")
    else:
        print("probabilities=unavailable")
    uses_id = any(name.startswith("customerID") for name in result.feature_names)
    print(f"identifier_used_as_feature={uses_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for reproducible stratified splitting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from smartpredict.config.schema import DatasetSpec, SplitStrategy
from smartpredict.core.exceptions import DataError
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.splitter import save_split_metadata, stratified_split
from smartpredict.data.validation import validate_dataset


@pytest.fixture()
def validated_frame(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> pd.DataFrame:
    """Return a validated frame suitable for splitting."""
    coerced, report = validate_dataset(synthetic_frame, dataset_schema)
    report.raise_if_invalid()
    return coerced


def test_split_respects_configured_ratios(
    validated_frame: pd.DataFrame, dataset_spec: DatasetSpec
) -> None:
    """The three splits follow the configured 70/15/15 ratios."""
    result = stratified_split(validated_frame, dataset_spec)
    sizes = result.metadata.sizes
    assert sizes["train"] + sizes["validation"] + sizes["test"] == 240
    for name in ("train", "validation", "test"):
        expected = getattr(dataset_spec.split, name) * 240
        assert abs(sizes[name] - expected) <= 2


def test_splits_are_disjoint(
    validated_frame: pd.DataFrame, dataset_spec: DatasetSpec
) -> None:
    """No row can belong to two splits."""
    result = stratified_split(validated_frame, dataset_spec)
    indices = result.metadata.indices
    sets = {name: set(values) for name, values in indices.items()}
    assert not sets["train"] & sets["validation"]
    assert not sets["train"] & sets["test"]
    assert not sets["validation"] & sets["test"]
    assert len(set().union(*sets.values())) == 240


def test_stratification_preserves_class_balance(
    validated_frame: pd.DataFrame, dataset_spec: DatasetSpec
) -> None:
    """Positive-class rates stay close to the original distribution."""
    target = dataset_spec.target_column
    overall = (validated_frame[target] == dataset_spec.positive_label).mean()
    result = stratified_split(validated_frame, dataset_spec)
    for subset in (result.train, result.validation, result.test):
        rate = (subset[target] == dataset_spec.positive_label).mean()
        assert abs(rate - overall) < 0.05


def test_split_is_reproducible(
    validated_frame: pd.DataFrame, dataset_spec: DatasetSpec
) -> None:
    """The configured seed produces identical splits."""
    first = stratified_split(validated_frame, dataset_spec)
    second = stratified_split(validated_frame, dataset_spec)
    assert first.metadata.indices == second.metadata.indices

    different = stratified_split(validated_frame, dataset_spec, seed=99)
    assert different.metadata.indices != first.metadata.indices


def test_split_metadata_is_persistable(
    validated_frame: pd.DataFrame, dataset_spec: DatasetSpec, tmp_path: Path
) -> None:
    """Split metadata can be saved as JSON for auditability."""
    result = stratified_split(validated_frame, dataset_spec)
    path = save_split_metadata(result.metadata, tmp_path / "split_metadata.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 42
    assert payload["sizes"]["train"] == result.metadata.sizes["train"]
    assert len(payload["indices"]["test"]) == result.metadata.sizes["test"]


def test_insufficient_class_samples_raise(
    synthetic_frame: pd.DataFrame, dataset_spec: DatasetSpec
) -> None:
    """A class with fewer than two rows cannot be stratified."""
    frame = synthetic_frame.copy()
    frame[dataset_spec.target_column] = dataset_spec.negative_label
    frame.loc[0, dataset_spec.target_column] = dataset_spec.positive_label
    with pytest.raises(DataError, match="at least two rows"):
        stratified_split(frame, dataset_spec)


def test_unsupported_strategy_raises(
    validated_frame: pd.DataFrame, dataset_spec: DatasetSpec
) -> None:
    """Only stratified splitting is currently supported."""
    spec = replace(
        dataset_spec, split=replace(dataset_spec.split, strategy=SplitStrategy.RANDOM)
    )
    with pytest.raises(DataError, match="Unsupported split strategy"):
        stratified_split(validated_frame, spec)

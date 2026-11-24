"""Tests for CSV ingestion and metadata extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from smartpredict.config.schema import DatasetSpec
from smartpredict.core.exceptions import DataError
from smartpredict.data.loader import load_csv


def test_load_csv_returns_frame_and_metadata(
    synthetic_csv: Path, dataset_spec: DatasetSpec
) -> None:
    """A valid CSV loads as raw strings with rich metadata."""
    result = load_csv(synthetic_csv)
    assert result.frame.shape == (240, 21)
    assert tuple(result.frame.columns) == dataset_spec.expected_columns
    assert result.metadata.n_rows == 240
    assert result.metadata.n_columns == 21
    assert len(result.metadata.sha256) == 64
    assert result.metadata.encoding in {"utf-8-sig", "utf-8", "cp1252", "latin-1"}
    assert result.metadata.inferred_dtypes["tenure"] == "integer"
    assert result.metadata.inferred_dtypes[dataset_spec.id_column] == "string"


def test_missing_file_raises(tmp_path: Path) -> None:
    """A missing dataset path is reported clearly."""
    with pytest.raises(DataError, match="does not exist"):
        load_csv(tmp_path / "missing.csv")


def test_empty_file_raises(tmp_path: Path) -> None:
    """A zero-byte CSV is rejected at ingestion."""
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    with pytest.raises(DataError, match="empty"):
        load_csv(path)


def test_non_csv_extension_raises(tmp_path: Path) -> None:
    """Only CSV files are accepted."""
    path = tmp_path / "dataset.txt"
    path.write_text("customerID,Churn\nA,Yes\n", encoding="utf-8")
    with pytest.raises(DataError, match="CSV"):
        load_csv(path)


def test_header_whitespace_is_normalized(
    tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    """Header whitespace is normalized and reported."""
    frame = synthetic_frame.rename(columns={"customerID": "  customerID  "})
    path = tmp_path / "spaced_headers.csv"
    frame.to_csv(path, index=False)
    result = load_csv(path)
    assert "customerID" in result.frame.columns
    assert "  customerID  " in result.metadata.renamed_columns


def test_encoding_fallback(tmp_path: Path, synthetic_frame: pd.DataFrame) -> None:
    """A cp1252-encoded file is decoded without data loss."""
    frame = synthetic_frame.copy()
    frame.loc[0, "customerID"] = "ID-é"
    path = tmp_path / "cp1252.csv"
    frame.to_csv(path, index=False, encoding="cp1252")
    result = load_csv(path)
    assert result.metadata.encoding == "cp1252"
    assert result.frame.loc[0, "customerID"] == "ID-é"


def test_explicit_encoding_skips_detection(
    tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    """An explicitly configured encoding is honored."""
    path = tmp_path / "cp1252.csv"
    synthetic_frame.to_csv(path, index=False, encoding="cp1252")
    result = load_csv(path, encoding="cp1252")
    assert result.metadata.encoding == "cp1252"


def test_duplicate_normalized_headers_raise(tmp_path: Path) -> None:
    """Ambiguous duplicate headers are rejected."""
    path = tmp_path / "duplicate_headers.csv"
    path.write_text("customerID, customerID\nA,B\n", encoding="utf-8")
    with pytest.raises(DataError, match="duplicate columns"):
        load_csv(path)

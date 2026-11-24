"""Tests for the end-to-end data preparation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from smartpredict.core.exceptions import DataError, DataValidationError
from smartpredict.data.pipeline import run_data_pipeline


def _pipeline_config(tmp_path: Path, csv_path: Path) -> Path:
    """Create a temporary data.yaml pointing at a test CSV."""
    source = yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "configs" / "data.yaml").read_text(
            encoding="utf-8"
        )
    )
    source["dataset"]["raw_path"] = str(csv_path)
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    return config_path


def test_pipeline_creates_splits_and_reports(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    """The pipeline persists processed splits and all audit artifacts."""
    config = _pipeline_config(tmp_path, synthetic_csv)
    result = run_data_pipeline(
        data_config=config,
        output_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )
    assert result.validation_report.is_valid
    assert sum(result.split.metadata.sizes.values()) == 240
    for name in ("train", "validation", "test"):
        assert Path(result.output_paths[name]).is_file()
    for name in ("validation_report", "split_metadata", "dataset_metadata"):
        assert Path(result.output_paths[name]).is_file()

    report_payload = json.loads(
        Path(result.output_paths["validation_report"]).read_text(encoding="utf-8")
    )
    assert report_payload["is_valid"] is True


def test_pipeline_writes_split_files_with_all_columns(
    tmp_path: Path, synthetic_csv: Path
) -> None:
    """Processed CSV files preserve the validated columns."""
    config = _pipeline_config(tmp_path, synthetic_csv)
    result = run_data_pipeline(
        data_config=config,
        output_dir=tmp_path / "processed",
        reports_dir=tmp_path / "reports",
    )
    train = pd.read_csv(result.output_paths["train"])
    assert train.shape[1] == 21
    assert "Churn" in train.columns
    assert "TotalCharges" in train.columns


def test_pipeline_strict_mode_raises_on_invalid_data(
    tmp_path: Path, synthetic_frame: pd.DataFrame
) -> None:
    """Strict mode stops before splitting when validation fails."""
    frame = synthetic_frame.copy()
    frame.loc[0, "Churn"] = "Maybe"
    csv_path = tmp_path / "invalid.csv"
    frame.to_csv(csv_path, index=False)
    config = _pipeline_config(tmp_path, csv_path)
    with pytest.raises(DataValidationError, match="INVALID_TARGET"):
        run_data_pipeline(
            data_config=config,
            output_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )


def test_pipeline_missing_raw_file_raises(tmp_path: Path) -> None:
    """A configured but absent raw dataset is reported."""
    config = _pipeline_config(tmp_path, tmp_path / "absent.csv")
    with pytest.raises(DataError, match="does not exist"):
        run_data_pipeline(
            data_config=config,
            output_dir=tmp_path / "processed",
            reports_dir=tmp_path / "reports",
        )

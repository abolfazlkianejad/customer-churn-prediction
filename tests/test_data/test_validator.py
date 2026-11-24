"""Tests for dataset validation rules and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from smartpredict.config.schema import DatasetSpec
from smartpredict.core.exceptions import DataValidationError
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.validation import validate_dataset


def test_valid_dataset_has_only_blank_charge_warnings(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """The synthetic Telco-like dataset passes with documented blank charges."""
    coerced, report = validate_dataset(synthetic_frame, dataset_schema)
    assert report.is_valid
    assert len(coerced) == len(synthetic_frame)
    blank_warnings = [
        issue for issue in report.warnings if issue.code == "BLANK_NUMERIC"
    ]
    assert len(blank_warnings) == 1
    assert blank_warnings[0].column == "TotalCharges"
    assert blank_warnings[0].affected_rows == 2
    assert pd.isna(coerced.loc[3, "TotalCharges"])


def test_missing_required_columns(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """A missing target column is a blocking error."""
    frame = synthetic_frame.drop(columns=["Churn"])
    _, report = validate_dataset(frame, dataset_schema)
    assert not report.is_valid
    assert any(issue.code == "MISSING_COLUMNS" for issue in report.errors)


def test_missing_categorical_values(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Blank categorical values are reported instead of being dropped."""
    frame = synthetic_frame.copy()
    frame.loc[0, "gender"] = ""
    coerced, report = validate_dataset(frame, dataset_schema)
    assert not report.is_valid
    assert any(
        issue.code == "MISSING_VALUES" and issue.column == "gender"
        for issue in report.errors
    )
    assert len(coerced) == len(frame)


def test_duplicate_customer_ids(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Duplicate identifiers are detected."""
    frame = synthetic_frame.copy()
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    _, report = validate_dataset(frame, dataset_schema)
    duplicate = next(
        issue for issue in report.errors if issue.code == "DUPLICATE_ID"
    )
    assert duplicate.affected_rows == 2
    assert duplicate.column == "customerID"


def test_total_charges_non_numeric_token(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Non-numeric TotalCharges values are errors, unlike blank strings."""
    frame = synthetic_frame.copy()
    frame.loc[0, "TotalCharges"] = "unknown"
    _, report = validate_dataset(frame, dataset_schema)
    error = next(
        issue
        for issue in report.errors
        if issue.code == "INVALID_NUMERIC" and issue.column == "TotalCharges"
    )
    assert "unknown" in error.examples


def test_numeric_out_of_range(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Configured numeric bounds are enforced."""
    frame = synthetic_frame.copy()
    frame.loc[0, "tenure"] = 999
    _, report = validate_dataset(frame, dataset_schema)
    assert any(
        issue.code == "NUMERIC_OUT_OF_RANGE" and issue.column == "tenure"
        for issue in report.errors
    )


def test_infinite_numeric_value(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Infinite values are rejected as invalid numerics."""
    frame = synthetic_frame.copy()
    frame["MonthlyCharges"] = frame["MonthlyCharges"].astype(object)
    frame.loc[0, "MonthlyCharges"] = "inf"
    _, report = validate_dataset(frame, dataset_schema)
    assert any(
        issue.code == "INVALID_NUMERIC" and issue.column == "MonthlyCharges"
        for issue in report.errors
    )


def test_unexpected_categorical_value(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Categories outside the configured domain are rejected."""
    frame = synthetic_frame.copy()
    frame.loc[0, "InternetService"] = "Satellite"
    _, report = validate_dataset(frame, dataset_schema)
    error = next(
        issue for issue in report.errors if issue.code == "UNEXPECTED_CATEGORY"
    )
    assert error.column == "InternetService"
    assert "Satellite" in error.examples


def test_invalid_target_value(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """The target must use the configured positive/negative labels."""
    frame = synthetic_frame.copy()
    frame.loc[0, "Churn"] = "Maybe"
    _, report = validate_dataset(frame, dataset_schema)
    assert any(issue.code == "INVALID_TARGET" for issue in report.errors)


def test_empty_dataset(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """An empty dataset is rejected."""
    _, report = validate_dataset(synthetic_frame.iloc[0:0], dataset_schema)
    assert any(issue.code == "EMPTY_DATASET" for issue in report.errors)


def test_extra_columns_are_warnings(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Undeclared columns are reported but do not block the pipeline."""
    frame = synthetic_frame.copy()
    frame["notes"] = "ignored"
    _, report = validate_dataset(frame, dataset_schema)
    assert report.is_valid
    assert any(issue.code == "UNEXPECTED_COLUMNS" for issue in report.warnings)


def test_report_json_roundtrip(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema, tmp_path: Path
) -> None:
    """Validation reports can be persisted as JSON."""
    _, report = validate_dataset(
        synthetic_frame, dataset_schema, source_path="test.csv"
    )
    path = report.save_json(tmp_path / "reports" / "validation.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["is_valid"] is True
    assert payload["n_rows"] == 240
    assert payload["issues"][0]["code"] == "BLANK_NUMERIC"


def test_raise_if_invalid(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Blocking findings can be converted into a domain exception."""
    frame = synthetic_frame.copy()
    frame.loc[0, "Churn"] = "Maybe"
    _, report = validate_dataset(frame, dataset_schema)
    with pytest.raises(DataValidationError, match="INVALID_TARGET"):
        report.raise_if_invalid()


def test_schema_uses_configured_column_names(dataset_spec: DatasetSpec) -> None:
    """The runtime schema is derived from the YAML contract, not hardcoded."""
    schema = DatasetSchema.from_spec(dataset_spec)
    assert schema.target_column == dataset_spec.target_column
    assert schema.numeric_columns == dataset_spec.numeric_columns
    assert schema.allowed_categories == dataset_spec.allowed_categories


def test_single_class_target_still_fails_default_validation(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Training validation continues to require two target classes."""
    frame = synthetic_frame.head(3).copy()
    frame["Churn"] = "No"
    _, report = validate_dataset(frame, dataset_schema)
    assert any(
        issue.code == "INSUFFICIENT_CLASS_SAMPLES" for issue in report.errors
    )


def test_inference_mode_skips_target_rules_but_keeps_features(
    synthetic_frame: pd.DataFrame, dataset_schema: DatasetSchema
) -> None:
    """Inference validation does not require a target while checking domains."""
    frame = synthetic_frame.head(3).copy()
    frame["Churn"] = "No"
    _, report = validate_dataset(frame, dataset_schema, require_target=False)
    assert report.is_valid

    frame.loc[frame.index[0], "InternetService"] = "Satellite"
    _, invalid = validate_dataset(frame, dataset_schema, require_target=False)
    assert any(
        issue.code == "UNEXPECTED_CATEGORY" for issue in invalid.errors
    )


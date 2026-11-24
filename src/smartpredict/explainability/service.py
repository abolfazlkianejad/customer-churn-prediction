"""Generate SHAP explanations from saved model artifacts only."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402

from smartpredict.config.loader import load_dataset_spec, load_training_config
from smartpredict.config.schema import (
    DatasetSpec,
    ExplainabilityMethod,
    TrainingConfig,
)
from smartpredict.core.exceptions import ExplainabilityError
from smartpredict.core.logging import get_logger
from smartpredict.core.paths import resolve_path
from smartpredict.data.schema import DatasetSchema
from smartpredict.data.validation import validate_dataset
from smartpredict.features.preprocessor import prepare_features_frame
from smartpredict.models.artifacts import build_artifact_paths
from smartpredict.models.prediction import generate_predictions

logger = get_logger(__name__)

EXPLANATION_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ExplanationInputs:
    """Saved artifacts and resolved settings required for explanations."""

    spec: DatasetSpec
    config: TrainingConfig
    model: Any
    preprocessor: ColumnTransformer
    metadata: dict[str, Any]
    paths: dict[str, Path]
    output_dir: Path
    sample_size: int
    local_examples: int


@dataclass(frozen=True)
class ExplanationArtifacts:
    """Persisted global and local explanation outputs."""

    global_importance: Path
    summary_plot: Path
    bar_plot: Path
    local_report: Path
    local_plots: tuple[Path, ...]
    global_payload: dict[str, Any]
    local_payload: dict[str, Any]
    feature_names: tuple[str, ...]
    selected_model: str
    sample_size: int


def _read_json(path: Path) -> dict[str, Any]:
    """Load one JSON mapping with a project-specific error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        msg = f"Could not read JSON artifact '{path}': {exc}"
        raise ExplainabilityError(msg) from exc
    if not isinstance(payload, dict):
        msg = f"JSON artifact '{path}' must contain an object."
        raise ExplainabilityError(msg)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON artifact with a project-specific error."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError) as exc:
        msg = f"Could not write explanation JSON '{path}': {exc}"
        raise ExplainabilityError(msg) from exc


def _import_shap() -> Any:
    """Import SHAP lazily and report a meaningful dependency failure."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment dependent
        msg = (
            "SHAP is required for explanations but is not installed. "
            "Install the declared 'shap' dependency and retry."
        )
        raise ExplainabilityError(msg) from exc
    return shap


def _create_tree_explainer(shap_module: Any, model: Any) -> Any:
    """Construct a SHAP tree explainer for the saved model."""
    try:
        return shap_module.TreeExplainer(model)
    except Exception as exc:  # noqa: BLE001
        msg = (
            "SHAP could not explain the selected model. Expected a fitted "
            f"tree model such as XGBoost; got {type(model).__name__}: {exc}"
        )
        raise ExplainabilityError(msg) from exc


def load_explanation_inputs(
    *,
    data_config: str | Path | None = None,
    train_config: str | Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    output_dir: Path | None = None,
    sample_size: int | None = None,
    local_examples: int | None = None,
) -> ExplanationInputs:
    """Load the selected model, preprocessor, and metadata from disk.

    No model or transformer is fitted here; all objects are loaded from the
    artifacts written by the completed Phase 3 training run.
    """
    spec = load_dataset_spec(data_config)
    config = load_training_config(data_config, train_config)
    settings = config.explainability
    if not settings.enabled:
        msg = "Explainability is disabled in the training configuration."
        raise ExplainabilityError(msg)
    if settings.method is not ExplainabilityMethod.SHAP_TREE:
        msg = f"Unsupported explanation method: {settings.method!r}."
        raise ExplainabilityError(msg)

    paths = build_artifact_paths(
        config.artifacts, models_dir=models_dir, reports_dir=reports_dir
    )
    for label in ("best_model", "preprocessor", "metadata"):
        if not paths[label].is_file():
            msg = (
                f"Missing explanation prerequisite '{label}' at "
                f"{paths[label]}. Run the training workflow first."
            )
            raise ExplainabilityError(msg)

    try:
        model = joblib.load(paths["best_model"])
        preprocessor = joblib.load(paths["preprocessor"])
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not load saved model artifacts: {exc}"
        raise ExplainabilityError(msg) from exc
    metadata = _read_json(paths["metadata"])

    selected = metadata.get("model", {}).get("selected")
    if selected is not None and selected != "xgboost":
        msg = (
            "SHAP tree explanations are configured for the selected XGBoost "
            f"model; metadata records '{selected}' instead."
        )
        raise ExplainabilityError(msg)

    effective_sample = settings.sample_size if sample_size is None else sample_size
    effective_local = (
        settings.local_examples if local_examples is None else local_examples
    )
    for label, value in (
        ("sample_size", effective_sample),
        ("local_examples", effective_local),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            msg = f"{label} must be a positive integer."
            raise ExplainabilityError(msg)

    return ExplanationInputs(
        spec=spec,
        config=config,
        model=model,
        preprocessor=preprocessor,
        metadata=metadata,
        paths=paths,
        output_dir=(
            settings.output_dir
            if output_dir is None
            else resolve_path(output_dir)
        ),
        sample_size=effective_sample,
        local_examples=effective_local,
    )


def _load_validation_records(spec: DatasetSpec) -> pd.DataFrame:
    """Load the validation split that was reserved during training."""
    path = spec.raw_path.parent.parent / "processed" / "validation.csv"
    if not path.is_file():
        msg = (
            f"Validation split not found at '{path}'. Run "
            "python -m smartpredict.cli.prepare_data before explaining."
        )
        raise ExplainabilityError(msg)

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    coerced, report = validate_dataset(
        frame, DatasetSchema.from_spec(spec), require_target=False
    )
    try:
        report.raise_if_invalid()
    except Exception as exc:  # noqa: BLE001
        msg = f"Validation records failed feature validation: {exc}"
        raise ExplainabilityError(msg) from exc
    return coerced


def _prepare_explanation_matrix(
    records: pd.DataFrame,
    spec: DatasetSpec,
    preprocessor: ColumnTransformer,
    model: Any,
    sample_size: int,
) -> tuple[np.ndarray, tuple[str, ...], pd.DataFrame]:
    """Transform a raw sample without fitting and verify feature alignment."""
    try:
        prepared = prepare_features_frame(records, spec)
        transformed = preprocessor.transform(prepared)
        names = tuple(
            str(name) for name in preprocessor.get_feature_names_out()
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not transform records with the saved preprocessor: {exc}"
        raise ExplainabilityError(msg) from exc

    features = np.asarray(transformed, dtype=float)
    if features.ndim != 2 or features.shape[1] != len(names):
        msg = (
            f"Transformed feature shape {features.shape} does not match "
            f"{len(names)} feature names."
        )
        raise ExplainabilityError(msg)
    expected_columns = getattr(model, "n_features_in_", None)
    if expected_columns is not None and int(expected_columns) != features.shape[1]:
        msg = (
            f"Saved model expects {expected_columns} features, but the saved "
            f"preprocessor produced {features.shape[1]}."
        )
        raise ExplainabilityError(msg)
    if features.shape[0] == 0:
        msg = "No validation records are available for explanation."
        raise ExplainabilityError(msg)

    count = min(sample_size, features.shape[0])
    return (
        features[:count],
        names,
        records.iloc[:count].reset_index(drop=True),
    )


def _shap_values(explainer: Any, features: np.ndarray) -> np.ndarray:
    """Compute positive-class SHAP values in a shape-independent way."""
    try:
        try:
            raw = explainer.shap_values(features, check_additivity=False)
        except TypeError:
            raw = explainer.shap_values(features)
    except Exception as exc:  # noqa: BLE001
        msg = f"SHAP value computation failed: {exc}"
        raise ExplainabilityError(msg) from exc

    if isinstance(raw, list):
        if not raw:
            msg = "SHAP returned no values for the selected sample."
            raise ExplainabilityError(msg)
        raw = raw[1] if len(raw) > 1 else raw[0]
    try:
        values = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        msg = f"SHAP values could not be converted to a numeric array: {exc}"
        raise ExplainabilityError(msg) from exc

    if values.ndim == 3:
        class_axes = [axis for axis, size in enumerate(values.shape) if size == 2]
        if not class_axes:
            msg = f"Unexpected 3-D SHAP value shape: {values.shape}."
            raise ExplainabilityError(msg)
        values = np.take(values, indices=1, axis=class_axes[-1])
    if (
        values.ndim == 1
        and features.shape[1] == 1
        and values.shape[0] == features.shape[0]
    ):
        values = values.reshape(-1, 1)

    expected_shape = features.shape
    if values.shape != expected_shape:
        msg = (
            f"SHAP value shape {values.shape} does not match feature matrix "
            f"shape {expected_shape}."
        )
        raise ExplainabilityError(msg)
    return values


def _base_value(explainer: Any) -> float:
    """Normalize SHAP's base value to a scalar float."""
    raw = getattr(explainer, "expected_value", 0.0)
    try:
        values = np.asarray(raw, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        msg = f"SHAP base value could not be interpreted: {exc}"
        raise ExplainabilityError(msg) from exc
    if values.size == 0:
        msg = "SHAP returned an empty base value."
        raise ExplainabilityError(msg)
    return float(values[-1])


def _save_current_figure(path: Path) -> None:
    """Persist and close the current matplotlib figure."""
    try:
        figure = plt.gcf()
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not save explanation plot '{path}': {exc}"
        raise ExplainabilityError(msg) from exc


def _save_summary_plot(
    shap_module: Any,
    values: np.ndarray,
    features: np.ndarray,
    names: tuple[str, ...],
    path: Path,
) -> None:
    """Write the global SHAP summary plot."""
    try:
        shap_module.summary_plot(
            values,
            features=features,
            feature_names=list(names),
            plot_type="dot",
            show=False,
        )
        _save_current_figure(path)
    except ExplainabilityError:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not create the SHAP summary plot: {exc}"
        raise ExplainabilityError(msg) from exc


def _save_bar_plot(
    shap_module: Any,
    values: np.ndarray,
    features: np.ndarray,
    names: tuple[str, ...],
    path: Path,
) -> None:
    """Write the global SHAP bar plot."""
    try:
        explanation_class = getattr(shap_module, "Explanation", None)
        plots = getattr(shap_module, "plots", None)
        if explanation_class is None or plots is None:
            shap_module.summary_plot(
                values,
                features=features,
                feature_names=list(names),
                plot_type="bar",
                show=False,
            )
        else:
            explanation = explanation_class(
                values=values,
                data=features,
                feature_names=list(names),
            )
            plots.bar(explanation, show=False)
        _save_current_figure(path)
    except ExplainabilityError:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not create the SHAP bar plot: {exc}"
        raise ExplainabilityError(msg) from exc


def _save_waterfall_plot(
    shap_module: Any,
    values: np.ndarray,
    features: np.ndarray,
    names: tuple[str, ...],
    base_value: float,
    path: Path,
) -> None:
    """Write one local SHAP waterfall plot."""
    try:
        explanation_class = getattr(shap_module, "Explanation", None)
        plots = getattr(shap_module, "plots", None)
        if explanation_class is None or plots is None:
            msg = "This SHAP version does not support waterfall explanations."
            raise ExplainabilityError(msg)
        try:
            explanation = explanation_class(
                values=values,
                data=features,
                base_values=base_value,
                feature_names=list(names),
            )
        except TypeError:
            explanation = explanation_class(
                values=values,
                data=features,
                base_value=base_value,
                feature_names=list(names),
            )
        plots.waterfall(explanation, show=False, max_display=10)
        _save_current_figure(path)
    except ExplainabilityError:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = f"Could not create a local SHAP plot: {exc}"
        raise ExplainabilityError(msg) from exc


def _global_payload(
    values: np.ndarray,
    names: tuple[str, ...],
    metadata: dict[str, Any],
    *,
    sample_size: int,
    source_rows: int,
    base_value: float,
) -> dict[str, Any]:
    """Build the global-importance report."""
    mean_abs = np.abs(values).mean(axis=0)
    ordered = sorted(
        range(len(names)), key=lambda index: float(mean_abs[index]), reverse=True
    )
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "explanation_type": "global",
        "method": ExplainabilityMethod.SHAP_TREE.value,
        "model_name": metadata.get("model", {}).get("selected", "unknown"),
        "evaluation_split": "validation",
        "sample_size": sample_size,
        "source_rows": source_rows,
        "base_value": base_value,
        "feature_names": list(names),
        "feature_importance": [
            {
                "rank": rank,
                "feature": names[index],
                "mean_abs_shap": float(mean_abs[index]),
            }
            for rank, index in enumerate(ordered, start=1)
        ],
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "note": (
            "Global importance is the mean absolute SHAP value across the "
            "sampled validation rows. It explains model behaviour, not target "
            "causality."
        ),
    }


def _local_payload(
    values: np.ndarray,
    features: np.ndarray,
    records: pd.DataFrame,
    names: tuple[str, ...],
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    spec: DatasetSpec,
    metadata: dict[str, Any],
    base_value: float,
    plot_paths: tuple[Path, ...],
) -> dict[str, Any]:
    """Build the individual-prediction explanation report."""
    entries: list[dict[str, Any]] = []
    for index in range(values.shape[0]):
        identifier = records.iloc[index].get(spec.id_column, "")
        entries.append(
            {
                "explanation_type": "local",
                "record_index": index,
                "customer_id": str(identifier),
                "prediction_label": int(labels[index]),
                "prediction_probability": (
                    float(probabilities[index])
                    if probabilities is not None
                    else None
                ),
                "base_value": base_value,
                "local_plot": str(plot_paths[index]) if plot_paths else None,
                "feature_contributions": [
                    {
                        "feature": names[column],
                        "transformed_value": float(features[index, column]),
                        "shap_value": float(values[index, column]),
                    }
                    for column in range(len(names))
                ],
            }
        )
    return {
        "schema_version": EXPLANATION_SCHEMA_VERSION,
        "explanation_type": "local",
        "method": ExplainabilityMethod.SHAP_TREE.value,
        "model_name": metadata.get("model", {}).get("selected", "unknown"),
        "evaluation_split": "validation",
        "base_value": base_value,
        "records": entries,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "note": (
            "Each entry explains one individual prediction and is distinct "
            "from the global feature-importance report."
        ),
    }


def _local_plot_path(output_dir: Path, index: int) -> Path:
    """Return the waterfall plot path for one local explanation."""
    return output_dir / f"local_explanation_{index:02d}.png"


def _expected_paths(output_dir: Path, local_examples: int) -> tuple[Path, ...]:
    """List every file an explanation run will create, for overwrite checks."""
    return (
        output_dir / "global_feature_importance.json",
        output_dir / "global_summary_plot.png",
        output_dir / "global_bar_plot.png",
        output_dir / "local_explanations.json",
        *(_local_plot_path(output_dir, index) for index in range(local_examples)),
    )


def generate_explanations(
    *,
    data_config: str | Path | None = None,
    train_config: str | Path | None = None,
    models_dir: Path | None = None,
    reports_dir: Path | None = None,
    output_dir: Path | None = None,
    sample_size: int | None = None,
    local_examples: int | None = None,
    records: pd.DataFrame | None = None,
    explainer_factory: Callable[[Any], Any] | None = None,
    overwrite: bool = False,
) -> ExplanationArtifacts:
    """Generate and persist global and local SHAP explanations.

    Args:
        data_config: Path to ``data.yaml``.
        train_config: Path to ``train.yaml``.
        models_dir: Override for the saved model directory.
        reports_dir: Override for the report directory containing metadata.
        output_dir: Override for explanation outputs.
        sample_size: Override for the number of explained validation rows.
        local_examples: Override for the number of individual explanations.
        records: Optional raw records; defaults to the validation split.
        explainer_factory: Optional SHAP explainer constructor, mainly for tests.
        overwrite: Explicit permission to replace existing explanation files.

    Returns:
        Paths and payloads for all generated explanation artifacts.

    Raises:
        ExplainabilityError: If artifacts, configuration, transformation, or
            SHAP computation fails.

    """
    inputs = load_explanation_inputs(
        data_config=data_config,
        train_config=train_config,
        models_dir=models_dir,
        reports_dir=reports_dir,
        output_dir=output_dir,
        sample_size=sample_size,
        local_examples=local_examples,
    )
    local_count = min(inputs.local_examples, inputs.sample_size)
    expected_paths = _expected_paths(inputs.output_dir, local_count)
    if not overwrite:
        existing = sorted(str(path) for path in expected_paths if path.exists())
        if existing:
            msg = (
                "Explanation artifacts already exist; pass --overwrite to "
                "replace them: " + ", ".join(existing)
            )
            raise ExplainabilityError(msg)

    shap_module = _import_shap()
    raw_records = (
        records if records is not None else _load_validation_records(inputs.spec)
    )
    features, names, sample_records = _prepare_explanation_matrix(
        raw_records,
        inputs.spec,
        inputs.preprocessor,
        inputs.model,
        inputs.sample_size,
    )
    explainer = (
        explainer_factory(inputs.model)
        if explainer_factory is not None
        else _create_tree_explainer(shap_module, inputs.model)
    )
    values = _shap_values(explainer, features)
    base_value = _base_value(explainer)

    predictions = generate_predictions(inputs.model, features)
    labels = np.asarray(predictions.labels)
    probabilities = predictions.probabilities

    global_importance = inputs.output_dir / "global_feature_importance.json"
    summary_plot = inputs.output_dir / "global_summary_plot.png"
    bar_plot = inputs.output_dir / "global_bar_plot.png"
    local_report = inputs.output_dir / "local_explanations.json"
    local_plots = tuple(
        _local_plot_path(inputs.output_dir, index) for index in range(local_count)
    )

    global_payload = _global_payload(
        values,
        names,
        inputs.metadata,
        sample_size=features.shape[0],
        source_rows=raw_records.shape[0],
        base_value=base_value,
    )
    local_payload = _local_payload(
        values[:local_count],
        features[:local_count],
        sample_records,
        names,
        labels[:local_count],
        (
            probabilities[:local_count]
            if probabilities is not None
            else None
        ),
        inputs.spec,
        inputs.metadata,
        base_value,
        local_plots,
    )

    _write_json(global_importance, global_payload)
    _write_json(local_report, local_payload)
    _save_summary_plot(shap_module, values, features, names, summary_plot)
    _save_bar_plot(shap_module, values, features, names, bar_plot)
    for index, path in enumerate(local_plots):
        _save_waterfall_plot(
            shap_module,
            values[index],
            features[index],
            names,
            base_value,
            path,
        )

    logger.info(
        "Saved SHAP explanations for %s validation rows (%s local) under %s",
        features.shape[0],
        local_count,
        inputs.output_dir,
    )
    return ExplanationArtifacts(
        global_importance=global_importance,
        summary_plot=summary_plot,
        bar_plot=bar_plot,
        local_report=local_report,
        local_plots=local_plots,
        global_payload=global_payload,
        local_payload=local_payload,
        feature_names=names,
        selected_model=str(
            inputs.metadata.get("model", {}).get("selected", "unknown")
        ),
        sample_size=features.shape[0],
    )

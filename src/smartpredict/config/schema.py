"""Typed configuration objects for data and training pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose
from pathlib import Path
from typing import Any

from smartpredict.core.exceptions import ConfigurationError
from smartpredict.core.paths import (
    EXPLAINABILITY_DIR,
    MODELS_DIR,
    OPTIMIZATION_DIR,
    PLOTS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)


class SplitStrategy(StrEnum):
    """Supported train/validation/test split strategies."""

    STRATIFIED = "stratified"
    RANDOM = "random"
    TEMPORAL = "temporal"


class ImputationStrategy(StrEnum):
    """Supported missing-value strategies."""

    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    DROP = "drop"


class EncodingStrategy(StrEnum):
    """Supported categorical encoding strategies."""

    ONE_HOT = "one_hot"
    ORDINAL = "ordinal"


class SupportedModel(StrEnum):
    """Classifier names accepted by the model registry and factory."""

    LOGISTIC_REGRESSION = "logistic_regression"
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"


class OptimizationDirection(StrEnum):
    """Direction used to rank candidate models."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ExplainabilityMethod(StrEnum):
    """Supported post-hoc explanation algorithms."""

    SHAP_TREE = "shap_tree"


class SearchSpaceType(StrEnum):
    """Supported Optuna parameter suggestion types."""

    INTEGER = "int"
    FLOAT = "float"
    CATEGORICAL = "categorical"


class ThresholdCriterion(StrEnum):
    """Configurable objectives for decision-threshold selection."""

    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    ACCURACY = "accuracy"
    SPECIFICITY = "specificity"
    WEIGHTED_SCORE = "weighted_score"


class MetricName(StrEnum):
    """Evaluation metrics that can be reported or used for selection."""

    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    AVERAGE_PRECISION = "average_precision"
    BRIER_SCORE = "brier_score"
    LOG_LOSS = "log_loss"


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for reproducible dataset splitting.

    Ratios must be strictly positive and sum to one.
    """

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15
    strategy: SplitStrategy = SplitStrategy.STRATIFIED
    stratify_column: str = "Churn"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate split ratios.

        Raises:
            ConfigurationError: If a ratio is invalid or ratios do not sum to 1.

        """
        ratios = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        for name, value in ratios.items():
            if not 0.0 < value < 1.0:
                msg = f"Split ratio '{name}' must be between 0 and 1, got {value}."
                raise ConfigurationError(msg)
        if not isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
            msg = f"Split ratios must sum to 1.0, got {sum(ratios.values())}."
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class DataConfig:
    """Declarative description of the input dataset."""

    raw_path: Path
    target_column: str = "Churn"
    positive_label: str = "Yes"
    negative_label: str = "No"
    id_column: str = "customerID"
    split: SplitConfig = field(default_factory=SplitConfig)


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative contract for a dataset, including its validation rules."""

    name: str
    raw_path: Path
    file_format: str
    target_column: str
    positive_label: str
    negative_label: str
    id_column: str
    drop_id_in_model: bool
    expected_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    numeric_ranges: Mapping[str, tuple[float, float]]
    allowed_categories: Mapping[str, tuple[str, ...]]
    split: SplitConfig = field(default_factory=SplitConfig)

    def __post_init__(self) -> None:
        """Check that the declared columns form a consistent schema.

        Raises:
            ConfigurationError: If column declarations are missing or overlap.

        """
        if not self.expected_columns:
            msg = "Dataset schema declares no expected columns."
            raise ConfigurationError(msg)
        if len(set(self.expected_columns)) != len(self.expected_columns):
            msg = "Dataset schema contains duplicate expected columns."
            raise ConfigurationError(msg)
        for required in (self.id_column, self.target_column):
            if required not in self.expected_columns:
                msg = f"Column '{required}' is missing from expected_columns."
                raise ConfigurationError(msg)
        declared = set(self.numeric_columns) | set(self.categorical_columns)
        overlap = set(self.numeric_columns) & set(self.categorical_columns)
        if overlap:
            msg = f"Columns declared both numeric and categorical: {sorted(overlap)}."
            raise ConfigurationError(msg)
        unknown = declared - set(self.expected_columns)
        if unknown:
            msg = f"Declared column types reference unknown columns: {sorted(unknown)}."
            raise ConfigurationError(msg)
        if self.split.stratify_column != self.target_column:
            msg = "The configured stratification column must be the target column."
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class FeatureConfig:
    """Preprocessing choices shared by training and inference."""

    numeric_imputation: ImputationStrategy = ImputationStrategy.MEDIAN
    categorical_imputation: ImputationStrategy = ImputationStrategy.MOST_FREQUENT
    categorical_encoding: EncodingStrategy = EncodingStrategy.ONE_HOT
    scale_numeric: bool = True
    handle_unknown: str = "ignore"


@dataclass(frozen=True)
class CandidateModelConfig:
    """One model candidate considered during training and comparison."""

    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the candidate structure.

        Raises:
            ConfigurationError: If parameter keys are not strings.

        """
        invalid = [key for key in self.params if not isinstance(key, str)]
        if invalid:
            msg = f"Model parameter names must be strings, got {invalid}."
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class EvaluationConfig:
    """Selection and reporting settings for model evaluation."""

    primary_metric: MetricName = MetricName.ROC_AUC
    optimization_direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    secondary_metrics: tuple[MetricName, ...] = (
        MetricName.ACCURACY,
        MetricName.PRECISION,
        MetricName.RECALL,
        MetricName.F1,
        MetricName.BRIER_SCORE,
        MetricName.LOG_LOSS,
    )

    def __post_init__(self) -> None:
        """Normalize and validate metric and direction values.

        Raises:
            ConfigurationError: If an unknown metric or direction is supplied.

        """
        try:
            primary = MetricName(self.primary_metric)
            direction = OptimizationDirection(self.optimization_direction)
            secondary = tuple(MetricName(item) for item in self.secondary_metrics)
        except ValueError as exc:
            msg = f"Invalid evaluation configuration: {exc}"
            raise ConfigurationError(msg) from exc
        object.__setattr__(self, "primary_metric", primary)
        object.__setattr__(self, "optimization_direction", direction)
        object.__setattr__(self, "secondary_metrics", secondary)


@dataclass(frozen=True)
class ArtifactConfig:
    """Paths and file names for persisted training artifacts."""

    models_dir: Path = field(default_factory=lambda: MODELS_DIR)
    reports_dir: Path = field(default_factory=lambda: REPORTS_DIR)
    plots_dir: Path = field(default_factory=lambda: PLOTS_DIR)
    best_model_file: str = "best_model.joblib"
    preprocessor_file: str = "preprocessing_pipeline.joblib"
    metadata_file: str = "model_metadata.json"
    comparison_file: str = "model_comparison.json"
    final_test_file: str = "final_test_report.json"
    confusion_matrix_file: str = "confusion_matrix.png"
    roc_curve_file: str = "roc_curve.png"
    precision_recall_file: str = "precision_recall_curve.png"

    def __post_init__(self) -> None:
        """Validate artifact file names.

        Raises:
            ConfigurationError: If any configured file name is blank.

        """
        names = {
            "best_model_file": self.best_model_file,
            "preprocessor_file": self.preprocessor_file,
            "metadata_file": self.metadata_file,
            "comparison_file": self.comparison_file,
            "final_test_file": self.final_test_file,
            "confusion_matrix_file": self.confusion_matrix_file,
            "roc_curve_file": self.roc_curve_file,
            "precision_recall_file": self.precision_recall_file,
        }
        for label, value in names.items():
            if not value.strip():
                msg = f"Artifact file name '{label}' must not be empty."
                raise ConfigurationError(msg)


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Settings for post-hoc explanations of the selected model."""

    enabled: bool = True
    method: ExplainabilityMethod = ExplainabilityMethod.SHAP_TREE
    max_background_samples: int = 100
    sample_size: int = 100
    local_examples: int = 3
    output_dir: Path = field(default_factory=lambda: EXPLAINABILITY_DIR)

    def __post_init__(self) -> None:
        """Validate explanation settings.

        Raises:
            ConfigurationError: If counts are not positive or the method is
                unsupported.

        """
        try:
            method = ExplainabilityMethod(self.method)
        except ValueError as exc:
            msg = f"Unsupported explainability method: {self.method!r}."
            raise ConfigurationError(msg) from exc
        object.__setattr__(self, "method", method)
        for label, value in (
            ("max_background_samples", self.max_background_samples),
            ("sample_size", self.sample_size),
            ("local_examples", self.local_examples),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                msg = f"explainability.{label} must be a positive integer."
                raise ConfigurationError(msg)
        if not str(self.output_dir).strip():
            msg = "explainability.output_dir must not be empty."
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class SearchSpaceParameter:
    """One typed Optuna parameter distribution declared in YAML."""

    type: SearchSpaceType
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    log: bool = False
    choices: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        """Validate the distribution before any study is created.

        Raises:
            ConfigurationError: If bounds, steps, or choices are inconsistent.

        """
        try:
            kind = SearchSpaceType(self.type)
        except ValueError as exc:
            msg = f"Unsupported search-space type: {self.type!r}."
            raise ConfigurationError(msg) from exc
        object.__setattr__(self, "type", kind)

        if kind is SearchSpaceType.CATEGORICAL:
            if not self.choices:
                msg = "Categorical search-space parameters require choices."
                raise ConfigurationError(msg)
            return

        if self.low is None or self.high is None:
            msg = f"{kind.value} search-space parameters require low and high."
            raise ConfigurationError(msg)
        if isinstance(self.low, bool) or isinstance(self.high, bool):
            msg = "Search-space bounds must be numbers, not booleans."
            raise ConfigurationError(msg)
        if not isinstance(self.low, (int, float)) or not isinstance(
            self.high, (int, float)
        ):
            msg = "Search-space bounds must be numbers."
            raise ConfigurationError(msg)
        if self.low >= self.high:
            msg = f"Search-space low ({self.low}) must be below high ({self.high})."
            raise ConfigurationError(msg)

        if kind is SearchSpaceType.INTEGER:
            if not isinstance(self.low, int) or not isinstance(self.high, int):
                msg = "Integer search-space bounds must be integers."
                raise ConfigurationError(msg)
            if self.log:
                msg = "Optuna integer suggestions do not support log sampling."
                raise ConfigurationError(msg)
            if self.step is not None and (
                isinstance(self.step, bool)
                or not isinstance(self.step, int)
                or self.step < 1
            ):
                msg = "Integer search-space step must be a positive integer."
                raise ConfigurationError(msg)
            return

        if isinstance(self.low, int):
            object.__setattr__(self, "low", float(self.low))
        if isinstance(self.high, int):
            object.__setattr__(self, "high", float(self.high))
        if self.step is not None:
            if (
                isinstance(self.step, bool)
                or not isinstance(self.step, (int, float))
                or self.step <= 0
            ):
                msg = "Float search-space step must be a positive number."
                raise ConfigurationError(msg)
            object.__setattr__(self, "step", float(self.step))
        if not isinstance(self.log, bool):
            msg = "Search-space log flag must be a boolean."
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class ThresholdConfig:
    """Business-oriented decision-threshold analysis settings."""

    enabled: bool = True
    minimum: float = 0.05
    maximum: float = 0.95
    step: float = 0.01
    criterion: ThresholdCriterion = ThresholdCriterion.WEIGHTED_SCORE
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "precision": 0.25,
            "recall": 0.50,
            "f1": 0.25,
        }
    )
    min_precision: float | None = None
    min_recall: float | None = None
    min_specificity: float | None = None

    def __post_init__(self) -> None:
        """Validate thresholds, objective weights, and business constraints.

        Raises:
            ConfigurationError: If the threshold grid or selection rule is invalid.

        """
        if not isinstance(self.enabled, bool):
            msg = "threshold.enabled must be a boolean."
            raise ConfigurationError(msg)
        if not 0.0 < self.minimum < self.maximum < 1.0:
            msg = "Threshold bounds must satisfy 0 < minimum < maximum < 1."
            raise ConfigurationError(msg)
        if not 0.0 < self.step <= self.maximum - self.minimum:
            msg = "Threshold step must be positive and no larger than the range."
            raise ConfigurationError(msg)

        try:
            criterion = ThresholdCriterion(self.criterion)
        except ValueError as exc:
            msg = f"Unsupported threshold criterion: {self.criterion!r}."
            raise ConfigurationError(msg) from exc
        object.__setattr__(self, "criterion", criterion)

        allowed = {"precision", "recall", "f1", "accuracy", "specificity"}
        cleaned_weights: dict[str, float] = {}
        for key, value in self.weights.items():
            if key not in allowed:
                msg = f"Unsupported threshold weight '{key}'."
                raise ConfigurationError(msg)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                msg = f"Threshold weight '{key}' must be a number."
                raise ConfigurationError(msg)
            if value < 0:
                msg = f"Threshold weight '{key}' must be non-negative."
                raise ConfigurationError(msg)
            cleaned_weights[key] = float(value)
        if criterion is ThresholdCriterion.WEIGHTED_SCORE and not any(
            value > 0 for value in cleaned_weights.values()
        ):
            msg = "weighted_score requires at least one positive threshold weight."
            raise ConfigurationError(msg)
        object.__setattr__(self, "weights", cleaned_weights)

        for label in ("min_precision", "min_recall", "min_specificity"):
            constraint_value: float | None = getattr(self, label)
            if constraint_value is not None:
                if isinstance(constraint_value, bool) or not isinstance(
                    constraint_value, (int, float)
                ):
                    msg = f"threshold.{label} must be a number."
                    raise ConfigurationError(msg)
                if not 0.0 <= float(constraint_value) <= 1.0:
                    msg = f"threshold.{label} must be between 0 and 1."
                    raise ConfigurationError(msg)
                object.__setattr__(self, label, float(constraint_value))


@dataclass(frozen=True)
class OptimizationConfig:
    """Optuna study, cross-validation, and threshold settings."""

    model_type: SupportedModel = SupportedModel.XGBOOST
    n_trials: int = 12
    timeout_seconds: float | None = 900.0
    seed: int = 42
    metric: MetricName = MetricName.ROC_AUC
    direction: OptimizationDirection = OptimizationDirection.MAXIMIZE
    cv_folds: int = 5
    study_name: str = "churn_xgboost_optimization"
    output_dir: Path = field(default_factory=lambda: OPTIMIZATION_DIR)
    storage_file: str = "optuna_study.db"
    data_directory: Path = field(default_factory=lambda: PROCESSED_DATA_DIR)
    train_file: str = "train.csv"
    validation_file: str = "validation.csv"
    resume: bool = False
    search_space: Mapping[str, Mapping[str, SearchSpaceParameter]] = field(
        default_factory=dict
    )
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)

    def __post_init__(self) -> None:
        """Validate study and cross-validation settings.

        Raises:
            ConfigurationError: If any optimization setting is invalid.

        """
        try:
            model_type = SupportedModel(self.model_type)
            metric = MetricName(self.metric)
            direction = OptimizationDirection(self.direction)
        except ValueError as exc:
            msg = f"Invalid optimization configuration: {exc}"
            raise ConfigurationError(msg) from exc
        object.__setattr__(self, "model_type", model_type)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "direction", direction)

        for label, value in (("n_trials", self.n_trials), ("seed", self.seed)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                msg = f"optimization.{label} must be a non-negative integer."
                raise ConfigurationError(msg)
        if self.n_trials == 0:
            msg = "optimization.n_trials must be at least 1."
            raise ConfigurationError(msg)
        if self.cv_folds < 2:
            msg = "optimization.cv_folds must be at least 2."
            raise ConfigurationError(msg)
        if self.timeout_seconds is not None:
            if isinstance(self.timeout_seconds, bool) or not isinstance(
                self.timeout_seconds, (int, float)
            ):
                msg = "optimization.timeout_seconds must be a positive number or null."
                raise ConfigurationError(msg)
            if self.timeout_seconds <= 0:
                msg = "optimization.timeout_seconds must be positive."
                raise ConfigurationError(msg)
            object.__setattr__(self, "timeout_seconds", float(self.timeout_seconds))

        for label in ("study_name", "storage_file", "train_file", "validation_file"):
            if not getattr(self, label).strip():
                msg = f"optimization.{label} must not be empty."
                raise ConfigurationError(msg)
        if not str(self.output_dir).strip() or not str(self.data_directory).strip():
            msg = "Optimization directories must not be empty."
            raise ConfigurationError(msg)
        if not isinstance(self.threshold, ThresholdConfig):
            msg = "optimization.threshold must be a ThresholdConfig."
            raise ConfigurationError(msg)


@dataclass(frozen=True)
class ModelConfig:
    """Legacy single-model identity and hyperparameters."""

    name: str = "xgboost"
    version: str = "v1"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate model identity.

        Raises:
            ConfigurationError: If the name or version is empty.

        """
        if not self.name.strip():
            msg = "Model name must not be empty."
            raise ConfigurationError(msg)
        if not self.version.strip():
            msg = "Model version must not be empty."
            raise ConfigurationError(msg)


def _default_model_candidates() -> dict[str, CandidateModelConfig]:
    """Return all supported models enabled with default (empty) parameters."""
    return {
        model.value: CandidateModelConfig()
        for model in SupportedModel
    }


@dataclass(frozen=True)
class TrainingConfig:
    """End-to-end configuration for a training run."""

    experiment_name: str
    data: DataConfig
    description: str = ""
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cv_folds: int = 5
    scoring: str = "roc_auc"
    seed: int = 42
    models: dict[str, CandidateModelConfig] = field(
        default_factory=_default_model_candidates
    )
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    explainability: ExplainabilityConfig = field(
        default_factory=ExplainabilityConfig
    )
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    def __post_init__(self) -> None:
        """Validate training, evaluation, and candidate-model settings.

        Raises:
            ConfigurationError: If folds, seed, model candidates, or evaluation
                settings are invalid.

        """
        if not self.experiment_name.strip():
            msg = "Experiment name must not be empty."
            raise ConfigurationError(msg)
        if self.cv_folds < 2:
            msg = f"cv_folds must be at least 2, got {self.cv_folds}."
            raise ConfigurationError(msg)
        if self.seed < 0:
            msg = f"seed must be non-negative, got {self.seed}."
            raise ConfigurationError(msg)
        if not self.models:
            msg = "At least one model candidate must be configured."
            raise ConfigurationError(msg)
        for name, candidate in self.models.items():
            try:
                SupportedModel(name)
            except ValueError as exc:
                msg = f"Unsupported model name: {name!r}."
                raise ConfigurationError(msg) from exc
            if not isinstance(candidate, CandidateModelConfig):
                msg = f"Model '{name}' must be a CandidateModelConfig."
                raise ConfigurationError(msg)
        if not any(candidate.enabled for candidate in self.models.values()):
            msg = "At least one configured model must be enabled."
            raise ConfigurationError(msg)

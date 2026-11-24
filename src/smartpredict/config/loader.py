"""Load declarative YAML configuration into typed configuration objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from smartpredict.config.schema import (
    ArtifactConfig,
    CandidateModelConfig,
    DataConfig,
    DatasetSpec,
    EncodingStrategy,
    EvaluationConfig,
    ExplainabilityConfig,
    ExplainabilityMethod,
    FeatureConfig,
    ImputationStrategy,
    MetricName,
    ModelConfig,
    OptimizationConfig,
    OptimizationDirection,
    SearchSpaceParameter,
    SearchSpaceType,
    SplitConfig,
    SplitStrategy,
    SupportedModel,
    ThresholdConfig,
    ThresholdCriterion,
    TrainingConfig,
)
from smartpredict.config.settings import get_settings
from smartpredict.core.exceptions import ConfigurationError
from smartpredict.core.paths import resolve_path

DEFAULT_DATA_CONFIG = "data.yaml"
DEFAULT_TRAIN_CONFIG = "train.yaml"


def load_yaml_mapping(
    path: str | Path, *, base_dir: Path | None = None
) -> dict[str, Any]:
    """Read a YAML file and require its root to be a mapping.

    Args:
        path: Absolute path or path relative to ``base_dir``.
        base_dir: Base directory for relative paths.

    Returns:
        Parsed YAML mapping.

    Raises:
        ConfigurationError: If the file is missing, unreadable, or not a mapping.

    """
    config_path = resolve_path(path, base_dir=base_dir)
    if not config_path.is_file():
        msg = f"Configuration file not found: {config_path}"
        raise ConfigurationError(msg)
    try:
        content = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        msg = f"Could not parse configuration file {config_path}: {exc}"
        raise ConfigurationError(msg) from exc
    if not isinstance(content, dict):
        msg = f"Configuration file {config_path} must contain a YAML mapping."
        raise ConfigurationError(msg)
    return content


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    """Return a value as a mapping or raise a configuration error."""
    if not isinstance(value, dict):
        msg = f"Configuration section '{label}' must be a mapping."
        raise ConfigurationError(msg)
    return value


def _require_sequence(value: Any, label: str) -> list[Any]:
    """Return a value as a list or raise a configuration error."""
    if not isinstance(value, list):
        msg = f"Configuration section '{label}' must be a list."
        raise ConfigurationError(msg)
    return value


def _require_text(value: Any, label: str) -> str:
    """Return a value as a stripped, non-empty string."""
    if not isinstance(value, str) or not value.strip():
        msg = f"Configuration value '{label}' must be a non-empty string."
        raise ConfigurationError(msg)
    return value.strip()


def _require_number(value: Any, label: str) -> float:
    """Return a value as a float or raise a configuration error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"Configuration value '{label}' must be a number."
        raise ConfigurationError(msg)
    return float(value)


def _require_integer(value: Any, label: str) -> int:
    """Return a value as an integer or raise a configuration error."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"Configuration value '{label}' must be an integer."
        raise ConfigurationError(msg)
    return value


def _require_bool(value: Any, label: str) -> bool:
    """Return a value as a boolean or raise a configuration error."""
    if not isinstance(value, bool):
        msg = f"Configuration value '{label}' must be a boolean."
        raise ConfigurationError(msg)
    return value


def _string_tuple(
    value: Any, label: str, *, allow_integers: bool = False
) -> tuple[str, ...]:
    """Return a list of string scalars as a tuple, rejecting ambiguous types.

    YAML 1.1 parses unquoted ``Yes``/``No`` as booleans. Silently stringifying
    them would turn a legitimate category domain into ``True``/``False``, so
    booleans and other non-string scalars are rejected here. Integers are
    allowed only for explicitly declared domains such as
    ``SeniorCitizen: [0, 1]``.

    Args:
        value: Raw YAML value, expected to be a list.
        label: Configuration path used in error messages.
        allow_integers: Whether integer items may be converted to strings.

    Returns:
        Tuple of string values in the original order.

    Raises:
        ConfigurationError: If the value is not a non-empty list of strings,
            or of integers when explicitly allowed.

    """
    items = _require_sequence(value, label)
    if not items:
        msg = f"Configuration section '{label}' must not be empty."
        raise ConfigurationError(msg)

    values: list[str] = []
    for index, item in enumerate(items):
        item_label = f"{label}[{index}]"
        if isinstance(item, bool):
            msg = (
                f"Configuration value '{item_label}' must be a quoted string; "
                f"got YAML boolean {item!r}. Quote boolean-like values "
                "explicitly, for example \"Yes\" / \"No\"."
            )
            raise ConfigurationError(msg)
        if isinstance(item, str):
            if not item.strip():
                msg = f"Configuration value '{item_label}' must not be blank."
                raise ConfigurationError(msg)
            values.append(item)
            continue
        if allow_integers and isinstance(item, int):
            values.append(str(item))
            continue
        msg = (
            f"Configuration value '{item_label}' must be a string"
            f"{' or integer' if allow_integers else ''}; "
            f"got {type(item).__name__}."
        )
        raise ConfigurationError(msg)
    return tuple(values)


def _canonical_model_name(value: Any, label: str) -> str:
    """Validate and normalize a configured model name."""
    name = _require_text(value, label).lower()
    try:
        return SupportedModel(name).value
    except ValueError as exc:
        allowed = ", ".join(model.value for model in SupportedModel)
        msg = (
            f"Unsupported model name {name!r} at '{label}'. "
            f"Expected one of: {allowed}."
        )
        raise ConfigurationError(msg) from exc


def _load_model_candidates(
    content: dict[str, Any], legacy_model: dict[str, Any], config_path: str | Path
) -> dict[str, CandidateModelConfig]:
    """Build the enabled/disabled model candidate map from the YAML content."""
    if "models" in content:
        section = _require_mapping(content.get("models"), "models")
        if not section:
            msg = "Configuration section 'models' must contain at least one candidate."
            raise ConfigurationError(msg)
        candidates: dict[str, CandidateModelConfig] = {}
        for raw_name, raw_value in section.items():
            name = _canonical_model_name(raw_name, f"models.{raw_name}")
            if name in candidates:
                msg = f"Duplicate model candidate '{name}' in {config_path}."
                raise ConfigurationError(msg)
            entry = _require_mapping(raw_value, f"models.{raw_name}")
            candidates[name] = CandidateModelConfig(
                enabled=_require_bool(
                    entry.get("enabled"), f"models.{raw_name}.enabled"
                ),
                params=_require_mapping(
                    entry.get("params", {}), f"models.{raw_name}.params"
                ),
            )
        return candidates

    # Backward compatibility with Phase 2's single-model configuration.
    fallback_name = _canonical_model_name(
        legacy_model.get("name"), "model.name"
    )
    return {
        fallback_name: CandidateModelConfig(
            enabled=True,
            params=_require_mapping(legacy_model.get("params"), "model.params"),
        )
    }


def _load_evaluation_config(content: dict[str, Any]) -> EvaluationConfig:
    """Build evaluation settings from the YAML content."""
    section = _require_mapping(content.get("evaluation"), "evaluation")
    metrics_section = _require_mapping(content.get("metrics", {}), "metrics")

    try:
        primary = MetricName(
            _require_text(section.get("primary_metric"), "evaluation.primary_metric")
        )
        direction = OptimizationDirection(
            _require_text(
                section.get("optimization_direction"),
                "evaluation.optimization_direction",
            )
        )
        raw_secondary = section.get(
            "secondary_metrics", metrics_section.get("secondary", [])
        )
        secondary_list = _require_sequence(
            raw_secondary, "evaluation.secondary_metrics"
        )
        secondary = tuple(
            MetricName(
                _require_text(value, f"evaluation.secondary_metrics[{index}]")
            )
            for index, value in enumerate(secondary_list)
        )
    except ValueError as exc:
        msg = f"Invalid evaluation configuration: {exc}"
        raise ConfigurationError(msg) from exc

    return EvaluationConfig(
        primary_metric=primary,
        optimization_direction=direction,
        secondary_metrics=secondary,
    )


def _load_artifact_config(content: dict[str, Any]) -> ArtifactConfig:
    """Build artifact path settings from the YAML content."""
    section = _require_mapping(content.get("artifacts", {}), "artifacts")
    bundle = _require_mapping(section.get("bundle_files", {}), "artifacts.bundle_files")

    def file_name(key: str, default: str) -> str:
        """Return a configured artifact file name or its default."""
        return _require_text(bundle.get(key, default), f"artifacts.bundle_files.{key}")

    return ArtifactConfig(
        models_dir=resolve_path(
            section.get("directory", "artifacts/models")
        ),
        reports_dir=resolve_path(
            section.get("reports_directory", "artifacts/reports")
        ),
        plots_dir=resolve_path(
            section.get("plots_directory", "artifacts/plots")
        ),
        best_model_file=file_name("model", "best_model.joblib"),
        preprocessor_file=file_name(
            "preprocessor", "preprocessing_pipeline.joblib"
        ),
        metadata_file=file_name("metadata", "model_metadata.json"),
        comparison_file=file_name("comparison", "model_comparison.json"),
        final_test_file=file_name("final_test", "final_test_report.json"),
    )


def load_dataset_spec(
    path: str | Path | None = None, *, base_dir: Path | None = None
) -> DatasetSpec:
    """Build a dataset contract from ``configs/data.yaml``.

    Args:
        path: Config path. Defaults to ``<configs_dir>/data.yaml``.
        base_dir: Base directory for a relative config path.

    Returns:
        Validated dataset specification.

    Raises:
        ConfigurationError: If the configuration is incomplete or inconsistent.

    """
    config_path = path or get_settings().configs_dir / DEFAULT_DATA_CONFIG
    content = load_yaml_mapping(config_path, base_dir=base_dir)

    dataset = _require_mapping(content.get("dataset"), "dataset")
    identity = _require_mapping(content.get("identity"), "identity")
    split_section = _require_mapping(content.get("split"), "split")
    types = _require_mapping(content.get("column_types"), "column_types")
    validation = _require_mapping(content.get("validation"), "validation")

    expected_columns = _string_tuple(
        content.get("expected_columns"), "expected_columns"
    )
    numeric_columns = _string_tuple(types.get("numeric"), "column_types.numeric")
    categorical_columns = _string_tuple(
        types.get("categorical"), "column_types.categorical"
    )

    raw_path_value = _require_text(dataset.get("raw_path"), "dataset.raw_path")
    raw_path = resolve_path(raw_path_value)

    numeric_ranges: dict[str, tuple[float, float]] = {}
    for column, bounds in _require_mapping(
        validation.get("numeric_ranges"), "validation.numeric_ranges"
    ).items():
        pair = _require_sequence(bounds, f"numeric_ranges.{column}")
        if len(pair) != 2:
            msg = f"Numeric range for '{column}' must have exactly two values."
            raise ConfigurationError(msg)
        lower = _require_number(pair[0], f"numeric_ranges.{column}[0]")
        upper = _require_number(pair[1], f"numeric_ranges.{column}[1]")
        if lower > upper:
            msg = f"Numeric range for '{column}' has lower bound above upper bound."
            raise ConfigurationError(msg)
        numeric_ranges[column] = (lower, upper)

    allowed_categories = {
        column: _string_tuple(
            values, f"allowed_categories.{column}", allow_integers=True
        )
        for column, values in _require_mapping(
            validation.get("allowed_categories"), "validation.allowed_categories"
        ).items()
    }

    try:
        strategy = SplitStrategy(
            _require_text(split_section.get("strategy"), "split.strategy")
        )
    except ValueError as exc:
        msg = f"Unsupported split strategy: {split_section.get('strategy')!r}"
        raise ConfigurationError(msg) from exc

    split = SplitConfig(
        train=_require_number(split_section.get("train"), "split.train"),
        validation=_require_number(
            split_section.get("validation"), "split.validation"
        ),
        test=_require_number(split_section.get("test"), "split.test"),
        strategy=strategy,
        stratify_column=_require_text(
            split_section.get("stratify_column"), "split.stratify_column"
        ),
        seed=_require_integer(split_section.get("seed"), "split.seed"),
    )

    return DatasetSpec(
        name=_require_text(dataset.get("name"), "dataset.name"),
        raw_path=raw_path,
        file_format=_require_text(dataset.get("format"), "dataset.format"),
        target_column=_require_text(
            dataset.get("target_column"), "dataset.target_column"
        ),
        positive_label=_require_text(
            dataset.get("positive_label"), "dataset.positive_label"
        ),
        negative_label=_require_text(
            dataset.get("negative_label"), "dataset.negative_label"
        ),
        id_column=_require_text(identity.get("id_column"), "identity.id_column"),
        drop_id_in_model=_require_bool(
            identity.get("drop_id_in_model"), "identity.drop_id_in_model"
        ),
        expected_columns=expected_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_ranges=numeric_ranges,
        allowed_categories=allowed_categories,
        split=split,
    )


def _load_explainability_config(content: dict[str, Any]) -> ExplainabilityConfig:
    """Build the typed explainability section from training YAML."""
    section = _require_mapping(content.get("explainability", {}), "explainability")
    defaults = ExplainabilityConfig()
    try:
        method = ExplainabilityMethod(
            _require_text(
                section.get("method", defaults.method.value),
                "explainability.method",
            )
        )
    except ValueError as exc:
        msg = f"Unsupported explainability method: {section.get('method')!r}"
        raise ConfigurationError(msg) from exc

    output_value = section.get("output_directory", section.get("output_dir"))
    output_dir = (
        resolve_path(_require_text(output_value, "explainability.output_directory"))
        if output_value is not None
        else defaults.output_dir
    )

    def positive_integer(key: str, fallback: int) -> int:
        """Read one optional positive integer setting."""
        value = section.get(key)
        if value is None:
            return fallback
        return _require_integer(value, f"explainability.{key}")

    return ExplainabilityConfig(
        enabled=_require_bool(
            section.get("enabled", defaults.enabled), "explainability.enabled"
        ),
        method=method,
        max_background_samples=positive_integer(
            "max_background_samples", defaults.max_background_samples
        ),
        sample_size=positive_integer("sample_size", defaults.sample_size),
        local_examples=positive_integer("local_examples", defaults.local_examples),
        output_dir=output_dir,
    )


def _literal_choice(value: Any, label: str) -> Any:
    """Return a JSON-safe scalar for a categorical search-space choice."""
    if isinstance(value, (bool, str, int, float)):
        return value
    msg = f"Configuration value '{label}' must be a scalar choice."
    raise ConfigurationError(msg)


def _load_search_space(
    content: dict[str, Any],
) -> dict[str, dict[str, SearchSpaceParameter]]:
    """Build typed Optuna distributions from the training YAML."""
    section = _require_mapping(
        content.get("search_space", {}), "optimization.search_space"
    )
    result: dict[str, dict[str, SearchSpaceParameter]] = {}
    for raw_model, raw_parameters in section.items():
        model_name = _canonical_model_name(
            raw_model, f"optimization.search_space.{raw_model}"
        )
        parameters = _require_mapping(
            raw_parameters, f"optimization.search_space.{raw_model}"
        )
        if not parameters:
            msg = f"Search space for '{model_name}' must not be empty."
            raise ConfigurationError(msg)
        typed: dict[str, SearchSpaceParameter] = {}
        for parameter_name, raw_parameter in parameters.items():
            label = f"optimization.search_space.{model_name}.{parameter_name}"
            definition = _require_mapping(raw_parameter, label)
            try:
                kind = SearchSpaceType(
                    _require_text(definition.get("type"), f"{label}.type")
                )
            except ValueError as exc:
                msg = f"Unsupported search-space type at '{label}.type': {exc}"
                raise ConfigurationError(msg) from exc

            choices: tuple[Any, ...] = ()
            if kind is SearchSpaceType.CATEGORICAL:
                raw_choices = _require_sequence(
                    definition.get("choices"), f"{label}.choices"
                )
                choices = tuple(
                    _literal_choice(value, f"{label}.choices[{index}]")
                    for index, value in enumerate(raw_choices)
                )

            low: int | float | None = None
            high: int | float | None = None
            step: int | float | None = None
            if kind is not SearchSpaceType.CATEGORICAL:
                if "low" not in definition or "high" not in definition:
                    msg = f"'{label}' requires low and high bounds."
                    raise ConfigurationError(msg)
                low_label = f"{label}.low"
                high_label = f"{label}.high"
                if kind is SearchSpaceType.INTEGER:
                    low = _require_integer(definition["low"], low_label)
                    high = _require_integer(definition["high"], high_label)
                    if "step" in definition:
                        step = _require_integer(definition["step"], f"{label}.step")
                else:
                    low = _require_number(definition["low"], low_label)
                    high = _require_number(definition["high"], high_label)
                    if "step" in definition:
                        step = _require_number(definition["step"], f"{label}.step")

            typed[parameter_name] = SearchSpaceParameter(
                type=kind,
                low=low,
                high=high,
                step=step,
                log=_require_bool(definition.get("log", False), f"{label}.log"),
                choices=choices,
            )
        result[model_name] = typed
    return result


def _load_threshold_config(content: dict[str, Any]) -> ThresholdConfig:
    """Build decision-threshold settings from the training YAML."""
    defaults = ThresholdConfig()
    section = _require_mapping(content.get("threshold", {}), "optimization.threshold")

    def number(key: str, default: float) -> float:
        """Read one optional threshold number."""
        value = section.get(key)
        if value is None:
            return default
        return _require_number(value, f"optimization.threshold.{key}")

    try:
        criterion = ThresholdCriterion(
            _require_text(
                section.get("criterion", defaults.criterion.value),
                "optimization.threshold.criterion",
            )
        )
    except ValueError as exc:
        msg = f"Unsupported threshold criterion: {exc}"
        raise ConfigurationError(msg) from exc

    raw_weights = _require_mapping(
        section.get("weights", defaults.weights), "optimization.threshold.weights"
    )
    weights = {
        _require_text(key, "optimization.threshold.weights"): _require_number(
            value, f"optimization.threshold.weights.{key}"
        )
        for key, value in raw_weights.items()
    }

    def constraint(key: str) -> float | None:
        """Read one optional business constraint."""
        value = section.get(key)
        if value is None:
            return None
        return _require_number(value, f"optimization.threshold.{key}")

    return ThresholdConfig(
        enabled=_require_bool(
            section.get("enabled", defaults.enabled), "optimization.threshold.enabled"
        ),
        minimum=number("minimum", number("min", defaults.minimum)),
        maximum=number("maximum", number("max", defaults.maximum)),
        step=number("step", defaults.step),
        criterion=criterion,
        weights=weights,
        min_precision=constraint("min_precision"),
        min_recall=constraint("min_recall"),
        min_specificity=constraint("min_specificity"),
    )


def _load_optimization_config(
    content: dict[str, Any], experiment_seed: int
) -> OptimizationConfig:
    """Build the typed optimization section from training YAML."""
    defaults = OptimizationConfig()
    section = _require_mapping(content.get("optimization", {}), "optimization")
    if not section:
        return defaults

    try:
        model_type = SupportedModel(
            _canonical_model_name(
                section.get("model_type", defaults.model_type.value),
                "optimization.model_type",
            )
        )
        metric = MetricName(
            _require_text(
                section.get("metric", defaults.metric.value), "optimization.metric"
            )
        )
        direction = OptimizationDirection(
            _require_text(
                section.get("direction", defaults.direction.value),
                "optimization.direction",
            )
        )
    except ValueError as exc:
        msg = f"Invalid optimization configuration: {exc}"
        raise ConfigurationError(msg) from exc

    def integer(key: str, fallback: int) -> int:
        """Read one optional integer setting."""
        value = section.get(key)
        if value is None:
            return fallback
        return _require_integer(value, f"optimization.{key}")

    def boolean(key: str, fallback: bool) -> bool:
        """Read one optional boolean setting."""
        value = section.get(key)
        if value is None:
            return fallback
        return _require_bool(value, f"optimization.{key}")

    def text(key: str, fallback: str) -> str:
        """Read one optional text setting."""
        value = section.get(key)
        if value is None:
            return fallback
        return _require_text(value, f"optimization.{key}")

    def path(key: str, fallback: Path) -> Path:
        """Read one optional resolved path setting."""
        if key not in section:
            return fallback
        return resolve_path(_require_text(section[key], f"optimization.{key}"))

    timeout_value = section.get("timeout_seconds")
    timeout_seconds = (
        defaults.timeout_seconds
        if timeout_value is None
        else _require_number(timeout_value, "optimization.timeout_seconds")
    )

    return OptimizationConfig(
        model_type=model_type,
        n_trials=integer("n_trials", defaults.n_trials),
        timeout_seconds=timeout_seconds,
        seed=integer("seed", experiment_seed),
        metric=metric,
        direction=direction,
        cv_folds=integer("cv_folds", defaults.cv_folds),
        study_name=text("study_name", defaults.study_name),
        output_dir=path("output_directory", defaults.output_dir),
        storage_file=text("storage_file", defaults.storage_file),
        data_directory=path("data_directory", defaults.data_directory),
        train_file=text("train_file", defaults.train_file),
        validation_file=text("validation_file", defaults.validation_file),
        resume=boolean("resume", defaults.resume),
        search_space=_load_search_space(section),
        threshold=_load_threshold_config(section),
    )


def load_training_config(
    data_config: str | Path | None = None,
    train_config: str | Path | None = None,
    *,
    base_dir: Path | None = None,
) -> TrainingConfig:
    """Build a training configuration from both declarative YAML files.

    Args:
        data_config: Path to the dataset configuration.
        train_config: Path to the training configuration.
        base_dir: Base directory for relative config paths.

    Returns:
        Validated training configuration.

    Raises:
        ConfigurationError: If the configuration is incomplete or inconsistent.

    """
    spec = load_dataset_spec(data_config, base_dir=base_dir)
    config_path = train_config or get_settings().configs_dir / DEFAULT_TRAIN_CONFIG
    content = load_yaml_mapping(config_path, base_dir=base_dir)

    experiment = _require_mapping(content.get("experiment"), "experiment")
    model_section = _require_mapping(content.get("model"), "model")
    feature_section = _require_mapping(content.get("features"), "features")
    validation = _require_mapping(content.get("validation"), "validation")

    try:
        numeric_strategy = ImputationStrategy(
            _require_text(
                feature_section.get("impute_numeric_with"),
                "features.impute_numeric_with",
            )
        )
        categorical_strategy = ImputationStrategy(
            _require_text(
                feature_section.get("impute_categorical_with"),
                "features.impute_categorical_with",
            )
        )
        encoding = EncodingStrategy(
            _require_text(
                feature_section.get("encode_categorical"),
                "features.encode_categorical",
            )
        )
    except ValueError as exc:
        msg = f"Unsupported preprocessing strategy in {config_path}: {exc}"
        raise ConfigurationError(msg) from exc

    data_config_object = DataConfig(
        raw_path=spec.raw_path,
        target_column=spec.target_column,
        positive_label=spec.positive_label,
        negative_label=spec.negative_label,
        id_column=spec.id_column,
        split=spec.split,
    )
    feature_config = FeatureConfig(
        numeric_imputation=numeric_strategy,
        categorical_imputation=categorical_strategy,
        categorical_encoding=encoding,
        scale_numeric=_require_bool(
            feature_section.get("scale_numeric"), "features.scale_numeric"
        ),
        handle_unknown=_require_text(
            feature_section.get("handle_unknown"), "features.handle_unknown"
        ),
    )
    model_config = ModelConfig(
        name=_require_text(model_section.get("name"), "model.name"),
        version=_require_text(model_section.get("version"), "model.version"),
        params=_require_mapping(model_section.get("params", {}), "model.params"),
    )
    experiment_seed = _require_integer(experiment.get("seed"), "experiment.seed")

    return TrainingConfig(
        experiment_name=_require_text(experiment.get("name"), "experiment.name"),
        description=_require_text(
            experiment.get("description", ""), "experiment.description"
        )
        if experiment.get("description")
        else "",
        data=data_config_object,
        model=model_config,
        features=feature_config,
        cv_folds=_require_integer(validation.get("cv_folds"), "validation.cv_folds"),
        scoring=_require_text(
            validation.get("scoring", "roc_auc"), "validation.scoring"
        ),
        seed=experiment_seed,
        models=_load_model_candidates(content, model_section, config_path),
        evaluation=_load_evaluation_config(content),
        artifacts=_load_artifact_config(content),
        explainability=_load_explainability_config(content),
        optimization=_load_optimization_config(content, experiment_seed),
    )

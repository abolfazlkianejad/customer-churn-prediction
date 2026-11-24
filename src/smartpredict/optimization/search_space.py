"""Typed, YAML-driven Optuna search spaces."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from smartpredict.config.schema import (
    SearchSpaceParameter,
    SearchSpaceType,
    SupportedModel,
)
from smartpredict.core.exceptions import OptimizationError

# Deliberately limited to parameters with a clear effect on tree churn models.
ALLOWED_SEARCH_PARAMETERS: dict[str, tuple[str, ...]] = {
    SupportedModel.XGBOOST.value: (
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ),
    SupportedModel.RANDOM_FOREST.value: (
        "n_estimators",
        "max_depth",
        "min_samples_leaf",
        "max_features",
    ),
}


def validate_search_space(
    model_type: SupportedModel,
    space: Mapping[str, SearchSpaceParameter],
) -> None:
    """Validate that a search space is non-empty and model-appropriate.

    Args:
        model_type: Model whose parameters will be suggested.
        space: Typed parameter distributions loaded from YAML.

    Raises:
        OptimizationError: If the space is empty, malformed, or contains a
            parameter that is not approved for the model type.

    """
    if not space:
        msg = f"No search space is configured for '{model_type.value}'."
        raise OptimizationError(msg)

    allowed = ALLOWED_SEARCH_PARAMETERS.get(model_type.value, ())
    if not allowed:
        msg = f"No approved search-space parameters for '{model_type.value}'."
        raise OptimizationError(msg)

    unknown = sorted(set(space) - set(allowed))
    if unknown:
        msg = (
            f"Unsupported optimization parameters for '{model_type.value}': "
            + ", ".join(unknown)
            + f". Approved parameters: {', '.join(allowed)}."
        )
        raise OptimizationError(msg)

    for name, parameter in space.items():
        if not isinstance(parameter, SearchSpaceParameter):
            msg = f"Search-space parameter '{name}' must be typed configuration."
            raise OptimizationError(msg)


def suggest_parameters(
    trial: Any, space: Mapping[str, SearchSpaceParameter]
) -> dict[str, Any]:
    """Ask an Optuna trial for one value per configured parameter.

    Args:
        trial: Active :class:`optuna.trial.Trial`.
        space: Typed distributions loaded from YAML.

    Returns:
        Mapping of parameter names to suggested values.

    Raises:
        OptimizationError: If a distribution cannot be suggested.

    """
    suggested: dict[str, Any] = {}
    for name in sorted(space):
        parameter = space[name]
        try:
            if parameter.type is SearchSpaceType.INTEGER:
                low, high = parameter.low, parameter.high
                if low is None or high is None:
                    msg = f"Search-space parameter '{name}' is missing numeric bounds."
                    raise OptimizationError(msg)
                step = 1 if parameter.step is None else int(parameter.step)
                suggested[name] = trial.suggest_int(
                    name, int(low), int(high), step=step
                )
            elif parameter.type is SearchSpaceType.FLOAT:
                low, high = parameter.low, parameter.high
                if low is None or high is None:
                    msg = f"Search-space parameter '{name}' is missing numeric bounds."
                    raise OptimizationError(msg)
                options: dict[str, Any] = {
                    "low": float(low),
                    "high": float(high),
                }
                if parameter.log:
                    options["log"] = True
                elif parameter.step is not None:
                    options["step"] = float(parameter.step)
                suggested[name] = trial.suggest_float(name, **options)
            else:
                suggested[name] = trial.suggest_categorical(
                    name, list(parameter.choices)
                )
        except Exception as exc:  # noqa: BLE001
            msg = f"Could not suggest parameter '{name}': {exc}"
            raise OptimizationError(msg) from exc
    return suggested


def merge_parameters(
    baseline_params: Mapping[str, Any], suggested_params: Mapping[str, Any]
) -> dict[str, Any]:
    """Combine fixed baseline settings with suggested search parameters."""
    return {**dict(baseline_params), **dict(suggested_params)}


def describe_search_space(space: Mapping[str, SearchSpaceParameter]) -> dict[str, Any]:
    """Return a JSON-safe description of the configured search space."""
    described: dict[str, Any] = {}
    for name, parameter in sorted(space.items()):
        described[name] = {
            "type": parameter.type.value,
            "low": parameter.low,
            "high": parameter.high,
            "step": parameter.step,
            "log": parameter.log,
            "choices": list(parameter.choices),
        }
    return described

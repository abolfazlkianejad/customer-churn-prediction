"""Registry of supported classifier constructors."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from smartpredict.config.schema import SupportedModel
from smartpredict.core.exceptions import ModelError, ModelNotFoundError
from smartpredict.core.logging import get_logger

logger = get_logger(__name__)

ModelConstructor = Callable[[dict[str, Any], int], Any]


def _with_seed(params: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Return parameters with a reproducible random state applied."""
    values = dict(params)
    values.setdefault("random_state", seed)
    return values


def _build_logistic_regression(params: Mapping[str, Any], seed: int) -> Any:
    """Construct a logistic regression classifier."""
    return LogisticRegression(**_with_seed(params, seed))


def _build_decision_tree(params: Mapping[str, Any], seed: int) -> Any:
    """Construct a decision tree classifier."""
    return DecisionTreeClassifier(**_with_seed(params, seed))


def _build_random_forest(params: Mapping[str, Any], seed: int) -> Any:
    """Construct a random forest classifier."""
    return RandomForestClassifier(**_with_seed(params, seed))


def _build_xgboost(params: Mapping[str, Any], seed: int) -> Any:
    """Construct XGBoost's classifier when the optional dependency is present."""
    try:
        module = import_module("xgboost")
    except ImportError as exc:
        msg = (
            "XGBoost is not installed. Install the 'xgboost' package or disable "
            "the xgboost model candidate."
        )
        raise ModelError(msg) from exc
    return module.XGBClassifier(**_with_seed(params, seed))


class ModelRegistry:
    """Mutable registry mapping canonical model names to constructors."""

    def __init__(self) -> None:
        """Create a registry pre-populated with all supported classifiers."""
        self._constructors: dict[str, ModelConstructor] = {}
        self.register(SupportedModel.LOGISTIC_REGRESSION, _build_logistic_regression)
        self.register(SupportedModel.DECISION_TREE, _build_decision_tree)
        self.register(SupportedModel.RANDOM_FOREST, _build_random_forest)
        self.register(SupportedModel.XGBOOST, _build_xgboost)

    @staticmethod
    def _canonical_name(name: str | SupportedModel) -> str:
        """Validate and normalize a model name."""
        try:
            return SupportedModel(str(name).strip().lower()).value
        except ValueError as exc:
            allowed = ", ".join(model.value for model in SupportedModel)
            msg = f"Unsupported model name {name!r}. Expected one of: {allowed}."
            raise ModelError(msg) from exc

    def register(
        self,
        name: str | SupportedModel,
        constructor: ModelConstructor,
        *,
        replace: bool = False,
    ) -> None:
        """Register a constructor for a canonical model name.

        Args:
            name: Canonical model name.
            constructor: Callable accepting parameters and a random seed.
            replace: Whether an existing registration may be replaced.

        Raises:
            ModelError: If the name is unknown or already registered.

        """
        canonical = self._canonical_name(name)
        if canonical in self._constructors and not replace:
            msg = f"Model '{canonical}' is already registered."
            raise ModelError(msg)
        self._constructors[canonical] = constructor
        logger.debug("Registered model constructor '%s'", canonical)

    def get_constructor(self, name: str | SupportedModel) -> ModelConstructor:
        """Return the constructor registered for a model name.

        Args:
            name: Canonical model name.

        Returns:
            Registered constructor callable.

        Raises:
            ModelNotFoundError: If the model is not registered.

        """
        canonical = self._canonical_name(name)
        constructor = self._constructors.get(canonical)
        if constructor is None:
            msg = f"Model '{canonical}' is not registered."
            raise ModelNotFoundError(msg)
        return constructor

    def create(
        self,
        name: str | SupportedModel,
        params: Mapping[str, Any] | None = None,
        *,
        seed: int = 42,
    ) -> Any:
        """Create a classifier instance with configured parameters.

        Args:
            name: Canonical model name.
            params: Hyperparameters from configuration.
            seed: Random seed used when ``random_state`` is not configured.

        Returns:
            Unfitted classifier instance.

        Raises:
            ModelError: If constructor parameters are invalid or construction
                fails.

        """
        constructor = self.get_constructor(name)
        try:
            return constructor(dict(params or {}), seed)
        except ModelError:
            raise
        except (TypeError, ValueError) as exc:
            msg = f"Invalid parameters for model '{name}': {exc}"
            raise ModelError(msg) from exc

    def available_models(self) -> tuple[str, ...]:
        """Return the sorted names currently registered."""
        return tuple(sorted(self._constructors))


_default_registry = ModelRegistry()


def get_default_registry() -> ModelRegistry:
    """Return the process-wide default model registry."""
    return _default_registry

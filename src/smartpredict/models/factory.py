"""Configuration-driven classifier factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from smartpredict.config.schema import CandidateModelConfig, SupportedModel
from smartpredict.core.exceptions import ModelError
from smartpredict.models.registry import ModelRegistry, get_default_registry


def create_model(
    name: str | SupportedModel,
    params: Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    registry: ModelRegistry | None = None,
) -> Any:
    """Build one classifier from a model name and hyperparameters.

    Args:
        name: Canonical model name.
        params: Hyperparameters loaded from ``train.yaml``.
        seed: Experiment seed applied when ``random_state`` is not configured.
        registry: Optional registry override, mainly useful in tests.

    Returns:
        Unfitted classifier.

    Raises:
        ModelError: If the model cannot be constructed.

    """
    active_registry = registry or get_default_registry()
    try:
        return active_registry.create(name, params, seed=seed)
    except ModelError:
        raise
    except ImportError as exc:
        msg = f"Optional dependency missing while creating model '{name}': {exc}"
        raise ModelError(msg) from exc


def create_candidate_model(
    name: str,
    candidate: CandidateModelConfig,
    *,
    seed: int = 42,
    registry: ModelRegistry | None = None,
) -> Any:
    """Build a classifier from a typed model candidate configuration."""
    return create_model(
        name, candidate.params, seed=seed, registry=registry
    )

"""Reproducibility helpers for global random number generators."""

from __future__ import annotations

import random

import numpy as np

from smartpredict.config.settings import Settings, get_settings


def set_global_seed(seed: int) -> None:
    """Seed the global Python and NumPy random number generators.

    Model-level randomness must additionally use the configured seed via each
    estimator's ``random_state`` parameter.

    Args:
        seed: Non-negative integer seed.

    Raises:
        ValueError: If ``seed`` is negative.

    """
    if seed < 0:
        msg = f"seed must be non-negative, got {seed}."
        raise ValueError(msg)
    random.seed(seed)
    np.random.seed(seed)


def seed_from_settings(settings: Settings | None = None) -> None:
    """Apply the globally configured random seed.

    Args:
        settings: Settings to read the seed from. Defaults to process settings.

    """
    set_global_seed((settings or get_settings()).random_seed)

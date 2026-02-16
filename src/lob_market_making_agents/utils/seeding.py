"""Deterministic random seeding utilities."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> np.random.Generator:
    """Seed Python and NumPy RNG state and return a local generator."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def derive_seed(base_seed: int, offset: int, modulo: Optional[int] = 2**32) -> int:
    """Derive a deterministic child seed from base seed and offset."""
    child = base_seed + (offset * 9973)
    if modulo is None:
        return child
    return child % modulo

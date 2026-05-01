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
    """Derive a deterministic child seed from base seed and offset.

    WHY linear (not hashing): we want full reproducibility under Python's
    optional hash randomisation (PYTHONHASHSEED unset). `hash()` would do that
    too if PYTHONHASHSEED is fixed, but a plain affine combination with a
    coprime stride (9973 is prime relative to 2**32) preserves seed diversity
    across nearby base/offset pairs without depending on the interpreter's
    hash configuration. Run-to-run reproducibility of `run_metrics.csv` is a
    submission-level requirement.
    """
    child = base_seed + (offset * 9973)
    if modulo is None:
        return child
    return child % modulo

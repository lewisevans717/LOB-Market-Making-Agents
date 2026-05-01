"""Shared helpers used by the analysis notebooks (notebooks/01..03).

Kept deliberately small so each function does one thing the notebooks would
otherwise repeat. The notebooks own the report-driven plotting decisions
(facet layout, palette, annotations)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AGENT_ORDER: tuple[str, ...] = ("A", "B", "C", "C_PLUS")
VOL_ORDER: tuple[str, ...] = ("low", "medium", "high")
TOX_ORDER: tuple[float, ...] = (0.0, 10.0, 30.0)


def bootstrap_ci(
    samples: Sequence[float] | np.ndarray,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return (mean, lo, hi) for a percentile bootstrap on the sample mean."""
    arr = np.asarray(samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(arr.mean()), float(lo), float(hi))


def parse_lambda_label(config_name: str) -> float | None:
    """Extract λ from a Pareto config_name suffix like 'mvp_bse_first_pareto_lam0p010000'."""
    marker = "_pareto_lam"
    idx = config_name.rfind(marker)
    if idx < 0:
        return None
    tail = config_name[idx + len(marker):]
    try:
        return float(tail.replace("p", "."))
    except ValueError:
        return None


def transfer_phase(config_name: str) -> str | None:
    """Map a transfer run's config_name onto a short phase label."""
    suffix = config_name.split("_transfer_", 1)[-1]
    return suffix or None


def save_figure(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=144, bbox_inches="tight")
    return path


def agent_palette() -> dict[str, str]:
    """Stable, colour-blind-friendly palette."""
    return {
        "A": "#4C72B0",
        "B": "#55A868",
        "C": "#C44E52",
        "C_PLUS": "#8172B2",
    }


def order_categorical(df: pd.DataFrame, col: str, order: Sequence[Any]) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=list(order), ordered=True)
    return df.sort_values(col, kind="stable")

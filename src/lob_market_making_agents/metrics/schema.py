"""Shared schemas and column contracts for metrics outputs."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MARKOUT_HORIZON = 5
DEFAULT_INVENTORY_THRESHOLD = 3.0

REQUIRED_RUN_FILES = (
    "events.csv",
    "summary.json",
    "meta.json",
)

RUN_CONTEXT_COLUMNS = (
    "run_id",
    "config_name",
    "agent",
    "volatility",
    "toxicity",
    "competition",
    "seed",
)

CONDITION_GROUP_COLUMNS = (
    "config_name",
    "agent",
    "volatility",
    "toxicity",
    "competition",
)


@dataclass(frozen=True)
class MetricsSettings:
    """Runtime configuration for metrics computation."""

    markout_horizon: int = DEFAULT_MARKOUT_HORIZON
    inventory_threshold: float = DEFAULT_INVENTORY_THRESHOLD


@dataclass(frozen=True)
class RunContext:
    """Metadata needed to place a run in its condition bucket."""

    run_id: str
    config_name: str
    agent: str
    volatility: str
    toxicity: float
    competition: str
    seed: int


def run_metric_columns(markout_horizon: int) -> tuple[str, ...]:
    """Column order for run-level metrics output."""
    return (
        *RUN_CONTEXT_COLUMNS,
        "steps",
        "fills",
        "fill_rate",
        "final_midprice",
        "final_inventory",
        "final_cash",
        "final_pnl",
        "inventory_mean",
        "inventory_var",
        "inventory_max_abs",
        "inventory_exposure_frac_abs_gt_threshold",
        f"markout_mean_h{markout_horizon}",
        f"markout_std_h{markout_horizon}",
        f"markout_p5_h{markout_horizon}",
        f"markout_count_h{markout_horizon}",
    )


def condition_metric_columns(markout_horizon: int) -> tuple[str, ...]:
    """Column order for condition-level aggregate output."""
    return (
        *CONDITION_GROUP_COLUMNS,
        "runs",
        "pnl_mean",
        "pnl_median",
        "pnl_std",
        "pnl_p5",
        "inventory_mean",
        "inventory_var",
        "inventory_max_abs",
        "inventory_exposure_frac_abs_gt_threshold",
        "fill_rate_mean",
        f"markout_mean_h{markout_horizon}",
        f"markout_std_h{markout_horizon}",
        f"markout_p5_h{markout_horizon}",
        f"markout_count_h{markout_horizon}",
    )


"""Run-level and condition-level metrics pipeline for BSE-native artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lob_market_making_agents.metrics.schema import (
    CONDITION_GROUP_COLUMNS,
    REQUIRED_RUN_FILES,
    MetricsSettings,
    RunContext,
    condition_metric_columns,
    run_metric_columns,
)


def discover_run_dirs(runs_dir: Path) -> tuple[list[Path], int]:
    """Discover valid run directories and count skipped candidates."""
    if not runs_dir.exists():
        return [], 0
    valid: list[Path] = []
    skipped = 0
    for child in sorted(runs_dir.iterdir(), key=lambda item: item.name):
        if not child.is_dir():
            continue
        if all((child / filename).exists() for filename in REQUIRED_RUN_FILES):
            valid.append(child)
        else:
            skipped += 1
    return valid, skipped


def parse_mm_state_csv(path: Path) -> list[dict[str, Any]]:
    """Load and parse typed MM state rows."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    parsed: list[dict[str, Any]] = []
    for row in rows:
        fill_price_raw = (row.get("fill_price") or "").strip()
        parsed.append(
            {
                "timestamp": int(float(row["timestamp"])),
                "inventory": float(row["inventory"]),
                "cash": float(row["cash"]),
                "pnl": float(row["pnl"]),
                "fill_side": str(row.get("fill_side", "none")),
                "fill_price": float(fill_price_raw) if fill_price_raw else None,
                "fill_qty": float(row.get("fill_qty", 0.0)),
                "informed": int(float(row.get("informed", 0))),
            }
        )
    return parsed


def parse_lob_frames_csv(path: Path) -> list[dict[str, Any]]:
    """Load and parse typed top-of-book rows."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    parsed: list[dict[str, Any]] = []
    for row in rows:
        parsed.append(
            {
                "timestamp": int(float(row["timestamp"])),
                "midprice": float(row["midprice"]),
            }
        )
    return parsed


def parse_run_context(meta: dict[str, Any], run_dir_name: str) -> RunContext:
    """Extract normalized context from run metadata."""
    regime = dict(meta.get("regime", {}))
    return RunContext(
        run_id=str(meta.get("run_id", run_dir_name)),
        config_name=str(meta["config_name"]),
        agent=str(meta["agent"]),
        volatility=str(regime["volatility"]),
        toxicity=float(regime["toxicity"]),
        competition=str(regime["competition"]),
        seed=int(meta["seed"]),
    )


def compute_run_metrics(
    *,
    mm_state_rows: list[dict[str, Any]],
    lob_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    context: RunContext,
    settings: MetricsSettings,
) -> dict[str, Any]:
    """Compute run-level profitability, inventory-risk, and markout metrics."""
    steps = len(mm_state_rows)
    fills = float(sum(1 for row in mm_state_rows if row["fill_side"] != "none"))
    fill_rate = fills / float(steps) if steps > 0 else 0.0

    inventories = np.asarray([row["inventory"] for row in mm_state_rows], dtype=float)
    if inventories.size == 0:
        inventory_mean = 0.0
        inventory_var = 0.0
        inventory_max_abs = 0.0
        inventory_exposure = 0.0
    else:
        abs_inventory = np.abs(inventories)
        inventory_mean = float(np.mean(inventories))
        inventory_var = float(np.var(inventories))
        inventory_max_abs = float(np.max(abs_inventory))
        inventory_exposure = float(np.mean(abs_inventory > settings.inventory_threshold))

    markouts = _compute_markouts(
        mm_state_rows=mm_state_rows,
        lob_rows=lob_rows,
        horizon=settings.markout_horizon,
    )
    key_mean = f"markout_mean_h{settings.markout_horizon}"
    key_std = f"markout_std_h{settings.markout_horizon}"
    key_p5 = f"markout_p5_h{settings.markout_horizon}"
    key_count = f"markout_count_h{settings.markout_horizon}"

    if markouts:
        markout_arr = np.asarray(markouts, dtype=float)
        markout_mean = float(np.mean(markout_arr))
        markout_std = float(np.std(markout_arr))
        markout_p5 = float(np.percentile(markout_arr, 5))
        markout_count = float(markout_arr.size)
    else:
        markout_mean = 0.0
        markout_std = 0.0
        markout_p5 = 0.0
        markout_count = 0.0

    final_midprice = float(summary.get("final_midprice", lob_rows[-1]["midprice"] if lob_rows else 0.0))
    final_inventory = float(summary.get("final_inventory", mm_state_rows[-1]["inventory"] if mm_state_rows else 0.0))
    final_cash = float(summary.get("final_cash", mm_state_rows[-1]["cash"] if mm_state_rows else 0.0))
    final_pnl = float(summary.get("final_pnl", mm_state_rows[-1]["pnl"] if mm_state_rows else 0.0))

    pnl_per_step = final_pnl / float(steps) if steps > 0 else 0.0

    return {
        "run_id": context.run_id,
        "config_name": context.config_name,
        "agent": context.agent,
        "volatility": context.volatility,
        "toxicity": context.toxicity,
        "competition": context.competition,
        "seed": context.seed,
        "steps": float(steps),
        "fills": fills,
        "fill_rate": fill_rate,
        "final_midprice": final_midprice,
        "final_inventory": final_inventory,
        "final_cash": final_cash,
        "final_pnl": final_pnl,
        "pnl_per_step": pnl_per_step,
        "inventory_mean": inventory_mean,
        "inventory_var": inventory_var,
        "inventory_max_abs": inventory_max_abs,
        "inventory_exposure_frac_abs_gt_threshold": inventory_exposure,
        key_mean: markout_mean,
        key_std: markout_std,
        key_p5: markout_p5,
        key_count: markout_count,
    }


def aggregate_condition_metrics(
    run_rows: list[dict[str, Any]],
    settings: MetricsSettings,
) -> list[dict[str, Any]]:
    """Aggregate run-level rows into condition-level summaries."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = tuple(row[column] for column in CONDITION_GROUP_COLUMNS)
        grouped.setdefault(key, []).append(row)

    mk_mean = f"markout_mean_h{settings.markout_horizon}"
    mk_std = f"markout_std_h{settings.markout_horizon}"
    mk_p5 = f"markout_p5_h{settings.markout_horizon}"
    mk_count = f"markout_count_h{settings.markout_horizon}"

    condition_rows: list[dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        rows = grouped[key]

        pnl_arr = np.asarray([row["final_pnl"] for row in rows], dtype=float)
        pnl_per_step_arr = np.asarray([row["pnl_per_step"] for row in rows], dtype=float)
        inv_mean_arr = np.asarray([row["inventory_mean"] for row in rows], dtype=float)
        inv_var_arr = np.asarray([row["inventory_var"] for row in rows], dtype=float)
        inv_max_arr = np.asarray([row["inventory_max_abs"] for row in rows], dtype=float)
        inv_exp_arr = np.asarray(
            [row["inventory_exposure_frac_abs_gt_threshold"] for row in rows],
            dtype=float,
        )
        fill_rate_arr = np.asarray([row["fill_rate"] for row in rows], dtype=float)
        mk_mean_arr = np.asarray([row[mk_mean] for row in rows], dtype=float)
        mk_std_arr = np.asarray([row[mk_std] for row in rows], dtype=float)
        mk_p5_arr = np.asarray([row[mk_p5] for row in rows], dtype=float)
        mk_count_arr = np.asarray([row[mk_count] for row in rows], dtype=float)

        condition_rows.append(
            {
                "config_name": key[0],
                "agent": key[1],
                "volatility": key[2],
                "toxicity": key[3],
                "competition": key[4],
                "runs": float(len(rows)),
                "pnl_mean": float(np.mean(pnl_arr)) if pnl_arr.size else 0.0,
                "pnl_median": float(np.median(pnl_arr)) if pnl_arr.size else 0.0,
                "pnl_std": float(np.std(pnl_arr)) if pnl_arr.size else 0.0,
                "pnl_p5": float(np.percentile(pnl_arr, 5)) if pnl_arr.size else 0.0,
                "pnl_per_step_mean": float(np.mean(pnl_per_step_arr)) if pnl_per_step_arr.size else 0.0,
                "pnl_per_step_std": float(np.std(pnl_per_step_arr)) if pnl_per_step_arr.size else 0.0,
                "pnl_per_step_p5": float(np.percentile(pnl_per_step_arr, 5)) if pnl_per_step_arr.size else 0.0,
                "inventory_mean": float(np.mean(inv_mean_arr)) if inv_mean_arr.size else 0.0,
                "inventory_var": float(np.mean(inv_var_arr)) if inv_var_arr.size else 0.0,
                "inventory_max_abs": float(np.mean(inv_max_arr)) if inv_max_arr.size else 0.0,
                "inventory_exposure_frac_abs_gt_threshold": (
                    float(np.mean(inv_exp_arr)) if inv_exp_arr.size else 0.0
                ),
                "fill_rate_mean": float(np.mean(fill_rate_arr)) if fill_rate_arr.size else 0.0,
                mk_mean: float(np.mean(mk_mean_arr)) if mk_mean_arr.size else 0.0,
                mk_std: float(np.mean(mk_std_arr)) if mk_std_arr.size else 0.0,
                mk_p5: float(np.mean(mk_p5_arr)) if mk_p5_arr.size else 0.0,
                mk_count: float(np.sum(mk_count_arr)) if mk_count_arr.size else 0.0,
            }
        )

    return condition_rows


def run_metrics_pipeline(
    *,
    runs_dir: Path,
    output_dir: Path,
    settings: MetricsSettings,
) -> dict[str, Any]:
    """Execute metrics computation and write outputs to disk."""
    valid_run_dirs, skipped_dirs = discover_run_dirs(runs_dir)
    run_rows: list[dict[str, Any]] = []
    parse_skips = 0

    for run_dir in valid_run_dirs:
        try:
            mm_state = parse_mm_state_csv(run_dir / "mm_state.csv")
            lob_rows = parse_lob_frames_csv(run_dir / "lob_frames.csv")
            summary = _load_json(run_dir / "summary.json")
            meta = _load_json(run_dir / "meta.json")
            context = parse_run_context(meta=meta, run_dir_name=run_dir.name)
            row = compute_run_metrics(
                mm_state_rows=mm_state,
                lob_rows=lob_rows,
                summary=summary,
                context=context,
                settings=settings,
            )
            run_rows.append(row)
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
            parse_skips += 1

    if not run_rows:
        raise ValueError(
            f"No valid run metrics could be computed from {runs_dir}. "
            "Check that run artifacts exist and are well-formed."
        )

    run_rows_sorted = sorted(
        run_rows,
        key=lambda row: (
            str(row["config_name"]),
            str(row["agent"]),
            str(row["volatility"]),
            float(row["toxicity"]),
            str(row["competition"]),
            int(row["seed"]),
            str(row["run_id"]),
        ),
    )
    condition_rows = aggregate_condition_metrics(run_rows_sorted, settings=settings)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_metrics_path = output_dir / "run_metrics.csv"
    condition_metrics_path = output_dir / "condition_metrics.csv"
    meta_path = output_dir / "metrics_meta.json"

    _write_csv(
        path=run_metrics_path,
        rows=run_rows_sorted,
        columns=run_metric_columns(settings.markout_horizon),
    )
    _write_csv(
        path=condition_metrics_path,
        rows=condition_rows,
        columns=condition_metric_columns(settings.markout_horizon),
    )

    metrics_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runs_discovered": int(len(valid_run_dirs) + skipped_dirs),
        "runs_processed": int(len(run_rows_sorted)),
        "runs_skipped": int(skipped_dirs + parse_skips),
        "group_count": int(len(condition_rows)),
        "markout_horizon": int(settings.markout_horizon),
        "inventory_threshold": float(settings.inventory_threshold),
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_meta, handle, indent=2, sort_keys=True)

    return {
        "run_metrics_path": str(run_metrics_path),
        "condition_metrics_path": str(condition_metrics_path),
        "metrics_meta_path": str(meta_path),
        **metrics_meta,
    }


def _compute_markouts(mm_state_rows: list[dict[str, Any]], lob_rows: list[dict[str, Any]], horizon: int) -> list[float]:
    mid_by_timestamp = {int(row["timestamp"]): float(row["midprice"]) for row in lob_rows}
    markouts: list[float] = []

    for row in mm_state_rows:
        fill_side = str(row["fill_side"])
        if fill_side == "none":
            continue
        fill_price = row["fill_price"]
        if fill_price is None:
            continue

        t = int(row["timestamp"])
        horizon_mid = mid_by_timestamp.get(t + horizon)
        if horizon_mid is None:
            continue

        if fill_side == "bid":
            markouts.append(float(horizon_mid - float(fill_price)))
        elif fill_side == "ask":
            markouts.append(float(float(fill_price) - horizon_mid))

    return markouts


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

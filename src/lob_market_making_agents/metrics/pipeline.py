"""Run-level and condition-level metrics pipeline."""

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


def parse_events_csv(path: Path) -> list[dict[str, Any]]:
    """Load and parse typed event rows."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    events: list[dict[str, Any]] = []
    for row in rows:
        fill_price_raw = (row.get("fill_price") or "").strip()
        fill_price = float(fill_price_raw) if fill_price_raw else None
        events.append(
            {
                "timestamp": int(row["timestamp"]),
                "midprice": float(row["midprice"]),
                "bid": float(row["bid"]),
                "ask": float(row["ask"]),
                "fill_side": str(row["fill_side"]),
                "fill_qty": float(row["fill_qty"]),
                "fill_price": fill_price,
                "inventory": float(row["inventory"]),
                "cash": float(row["cash"]),
                "pnl": float(row["pnl"]),
            }
        )
    return events


def parse_run_context(meta: dict[str, Any], run_dir_name: str) -> RunContext:
    """Extract normalized context from meta.json."""
    regime = meta.get("regime", {})
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
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    context: RunContext,
    settings: MetricsSettings,
) -> dict[str, Any]:
    """Compute run-level profitability, inventory-risk, and markout metrics."""
    steps = len(events)
    fills = float(sum(1 for event in events if event["fill_side"] != "none"))
    fill_rate = fills / float(steps) if steps > 0 else 0.0

    inventories = np.asarray([event["inventory"] for event in events], dtype=float)
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

    markouts = _compute_markouts(events=events, horizon=settings.markout_horizon)
    markout_key_mean = f"markout_mean_h{settings.markout_horizon}"
    markout_key_std = f"markout_std_h{settings.markout_horizon}"
    markout_key_p5 = f"markout_p5_h{settings.markout_horizon}"
    markout_key_count = f"markout_count_h{settings.markout_horizon}"

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

    final_midprice = float(summary.get("final_midprice", events[-1]["midprice"] if events else 0.0))
    final_inventory = float(summary.get("final_inventory", events[-1]["inventory"] if events else 0.0))
    final_cash = float(summary.get("final_cash", events[-1]["cash"] if events else 0.0))
    final_pnl = float(summary.get("final_pnl", events[-1]["pnl"] if events else 0.0))

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
        "inventory_mean": inventory_mean,
        "inventory_var": inventory_var,
        "inventory_max_abs": inventory_max_abs,
        "inventory_exposure_frac_abs_gt_threshold": inventory_exposure,
        markout_key_mean: markout_mean,
        markout_key_std: markout_std,
        markout_key_p5: markout_p5,
        markout_key_count: markout_count,
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
            events = parse_events_csv(run_dir / "events.csv")
            summary = _load_json(run_dir / "summary.json")
            meta = _load_json(run_dir / "meta.json")
            context = parse_run_context(meta=meta, run_dir_name=run_dir.name)
            row = compute_run_metrics(events=events, summary=summary, context=context, settings=settings)
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


def _compute_markouts(events: list[dict[str, Any]], horizon: int) -> list[float]:
    mid_by_timestamp = {int(event["timestamp"]): float(event["midprice"]) for event in events}
    markouts: list[float] = []
    for event in events:
        side = str(event["fill_side"])
        if side == "none":
            continue
        fill_price = event["fill_price"]
        if fill_price is None:
            continue
        timestamp = int(event["timestamp"])
        horizon_mid = mid_by_timestamp.get(timestamp + horizon)
        if horizon_mid is None:
            continue
        if side == "bid":
            markouts.append(float(horizon_mid - float(fill_price)))
        elif side == "ask":
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


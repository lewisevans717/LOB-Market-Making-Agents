import csv
import json
from pathlib import Path

from lob_market_making_agents.experiments.artifacts import write_run_artifacts
from lob_market_making_agents.metrics.pipeline import run_metrics_pipeline
from lob_market_making_agents.metrics.schema import (
    MetricsSettings,
    condition_metric_columns,
    run_metric_columns,
)


def _make_events(pnl_offset: float) -> list[dict[str, float | str | None]]:
    events: list[dict[str, float | str | None]] = []
    for timestamp in range(1, 7):
        fill_side: str = "none"
        fill_qty = 0.0
        fill_price: float | None = None
        bid = 99.5 + timestamp * 0.1
        ask = 100.5 + timestamp * 0.1
        if timestamp == 1:
            fill_side = "ask"
            fill_qty = 1.0
            fill_price = ask
        midprice = 100.0 + (timestamp * 0.05) + pnl_offset
        inventory = -1.0 if timestamp >= 1 else 0.0
        cash = 100.0 + pnl_offset
        pnl = cash + inventory * midprice
        events.append(
            {
                "timestamp": timestamp,
                "midprice": midprice,
                "bid": bid,
                "ask": ask,
                "fill_side": fill_side,
                "fill_qty": fill_qty,
                "fill_price": fill_price,
                "inventory": inventory,
                "cash": cash,
                "pnl": pnl,
            }
        )
    return events


def _write_run(tmp_runs: Path, run_id: str, seed: int, pnl_offset: float) -> None:
    events = _make_events(pnl_offset=pnl_offset)
    last = events[-1]
    summary = {
        "steps": float(len(events)),
        "final_midprice": float(last["midprice"]),
        "final_inventory": float(last["inventory"]),
        "final_cash": float(last["cash"]),
        "final_pnl": float(last["pnl"]),
        "fills": 1.0,
    }
    meta = {
        "run_id": run_id,
        "config_name": "mvp_scaffold",
        "agent": "A",
        "seed": seed,
        "regime": {
            "volatility": "low",
            "toxicity": 0.0,
            "competition": "solo",
        },
    }
    write_run_artifacts(run_dir=tmp_runs / run_id, events=events, summary=summary, meta=meta)


def test_metrics_pipeline_outputs_and_counts(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    output_dir = tmp_path / "metrics"
    runs_dir.mkdir(parents=True, exist_ok=True)

    _write_run(runs_dir, run_id="run_001", seed=11, pnl_offset=0.0)
    _write_run(runs_dir, run_id="run_002", seed=17, pnl_offset=0.2)
    (runs_dir / "broken_run").mkdir(parents=True, exist_ok=True)

    report = run_metrics_pipeline(
        runs_dir=runs_dir,
        output_dir=output_dir,
        settings=MetricsSettings(markout_horizon=5, inventory_threshold=3.0),
    )

    run_metrics_path = output_dir / "run_metrics.csv"
    condition_metrics_path = output_dir / "condition_metrics.csv"
    meta_path = output_dir / "metrics_meta.json"
    assert run_metrics_path.exists()
    assert condition_metrics_path.exists()
    assert meta_path.exists()

    with run_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == run_metric_columns(5)
        run_rows = list(reader)
    assert len(run_rows) == 2

    with condition_metrics_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == condition_metric_columns(5)
        condition_rows = list(reader)
    assert len(condition_rows) == 1

    loaded_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded_meta["runs_discovered"] == 3
    assert loaded_meta["runs_processed"] == 2
    assert loaded_meta["runs_skipped"] == 1
    assert loaded_meta["group_count"] == 1
    assert loaded_meta["markout_horizon"] == 5
    assert loaded_meta["inventory_threshold"] == 3.0

    assert report["runs_processed"] == 2
    assert report["group_count"] == 1

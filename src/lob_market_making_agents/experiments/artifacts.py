"""Writers for experiment artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EVENT_COLUMNS = (
    "timestamp",
    "midprice",
    "bid",
    "ask",
    "fill_side",
    "fill_qty",
    "fill_price",
    "inventory",
    "cash",
    "pnl",
)


def build_run_id(*, config_name: str, agent: str, volatility: str, toxicity: float, competition: str, seed: int) -> str:
    """Build a deterministic run identifier from run metadata."""
    payload = f"{config_name}|{agent}|{volatility}|{toxicity}|{competition}|{seed}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    return f"{config_name}_{agent}_{volatility}_tox{int(toxicity)}_{competition}_s{seed}_{digest}"


def write_run_artifacts(
    run_dir: Path,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    """Persist one run's structured outputs."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_events(run_dir / "events.csv", events)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "meta.json", meta)


def _write_events(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow({column: event.get(column) for column in EVENT_COLUMNS})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)


"""Writers for BSE-native run artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TAPE_COLUMNS = (
    "sequence",
    "timestamp",
    "event_type",
    "price",
    "qty",
    "party1",
    "party2",
    "tid",
    "side",
    "qid",
)

LOB_FRAME_COLUMNS = (
    "timestamp",
    "best_bid",
    "best_ask",
    "midprice",
    "bid_depth",
    "ask_depth",
)

MM_STATE_COLUMNS = (
    "timestamp",
    "inventory",
    "cash",
    "pnl",
    "bid_quote",
    "ask_quote",
    "fill_side",
    "fill_price",
    "fill_qty",
    "informed",
    "flow_imbalance",
)

COMPETITOR_MM_STATE_COLUMNS = (
    "timestamp",
    "trader_id",
    "inventory",
    "cash",
    "pnl",
    "bid_quote",
    "ask_quote",
)


def build_run_id(*, config_name: str, agent: str, volatility: str, toxicity: float, competition: str, seed: int) -> str:
    """Build a deterministic run identifier from run metadata.

    WHY a SHA-1 digest tail: the human-readable prefix collides for any two
    runs whose readable tuple matches but whose `config_name` differs only by
    a sub-experiment suffix (e.g. `_pareto_lam0p010000`). Appending a 10-char
    digest makes run_id collision-resistant *without* relying on the suffix
    being unique — the resume logic and the metrics aggregator both depend on
    run_id ↔ run_dir being a bijection.
    """
    payload = f"{config_name}|{agent}|{volatility}|{toxicity}|{competition}|{seed}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
    tox = int(round(float(toxicity)))
    return f"{config_name}_{agent}_{volatility}_tox{tox}_{competition}_s{seed}_{digest}"


def write_run_artifacts(
    *,
    run_dir: Path,
    tape_rows: list[dict[str, Any]],
    lob_frames: list[dict[str, Any]],
    mm_state_rows: list[dict[str, Any]],
    competitor_state_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    meta: dict[str, Any],
) -> None:
    """Persist one BSE-native run output set."""
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(path=run_dir / "tape.csv", rows=tape_rows, columns=TAPE_COLUMNS)
    _write_csv(path=run_dir / "lob_frames.csv", rows=lob_frames, columns=LOB_FRAME_COLUMNS)
    _write_csv(path=run_dir / "mm_state.csv", rows=mm_state_rows, columns=MM_STATE_COLUMNS)
    _write_csv(
        path=run_dir / "competitor_mm_state.csv",
        rows=competitor_state_rows,
        columns=COMPETITOR_MM_STATE_COLUMNS,
    )
    _write_json(path=run_dir / "summary.json", payload=summary)
    _write_json(path=run_dir / "meta.json", payload=meta)


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)

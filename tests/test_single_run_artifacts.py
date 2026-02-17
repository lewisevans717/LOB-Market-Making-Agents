import csv
import json
from pathlib import Path

from lob_market_making_agents.experiments.artifacts import EVENT_COLUMNS, build_run_id
from lob_market_making_agents.experiments.run import run_single


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_single_run_artifacts_schema_and_determinism(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_artifacts.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: mvp_scaffold",
                "seeds: [11]",
                "regimes:",
                "  volatility: [low]",
                "  toxicity: [0]",
                "  competition: [solo]",
                "agents: [A, B, C]",
                "single_run:",
                "  agent: A",
                "  agent_params:",
                "    spread: 1.0",
                "  regime:",
                "    volatility: low",
                "    toxicity: 0",
                "    competition: solo",
                "  episode_steps: 200",
                "  base_midprice: 100.0",
                "  size: 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    seed = 314

    code = run_single(config_path=config_path, seed=seed, output_dir=tmp_path)
    assert code == 0

    run_id = build_run_id(
        config_name="mvp_scaffold",
        agent="A",
        volatility="low",
        toxicity=0.0,
        competition="solo",
        seed=seed,
    )
    run_dir = tmp_path / "runs" / run_id

    events_path = run_dir / "events.csv"
    summary_path = run_dir / "summary.json"
    meta_path = run_dir / "meta.json"

    assert events_path.exists()
    assert summary_path.exists()
    assert meta_path.exists()

    with events_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == EVENT_COLUMNS
        rows = list(reader)

    assert len(rows) == 200
    assert set(rows[0].keys()) == set(EVENT_COLUMNS)

    summary = json.loads(_read_text(summary_path))
    meta = json.loads(_read_text(meta_path))
    assert "final_pnl" in summary
    assert meta["run_id"] == run_id
    assert meta["seed"] == seed

    events_before = _read_text(events_path)
    summary_before = _read_text(summary_path)
    meta_before = _read_text(meta_path)

    rerun_code = run_single(config_path=config_path, seed=seed, output_dir=tmp_path)
    assert rerun_code == 0
    assert _read_text(events_path) == events_before
    assert _read_text(summary_path) == summary_before
    assert _read_text(meta_path) == meta_before

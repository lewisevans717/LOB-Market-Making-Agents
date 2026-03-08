from pathlib import Path

from lob_market_making_agents.experiments.run import build_parser, run_grid, run_metrics, run_single
from lob_market_making_agents.utils.seeding import derive_seed


def test_parser_includes_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "single" in help_text
    assert "grid" in help_text
    assert "metrics" in help_text


def _write_bse_config(path: Path, *, agent: str = "A") -> None:
    path.write_text(
        "\n".join(
            [
                "name: mvp_bse_test",
                "seeds: [11, 17]",
                "regimes:",
                "  volatility: [low]",
                "  toxicity: [0]",
                "  competition: [solo]",
                "agents: [A, B]",
                "bse:",
                "  session:",
                "    episode_steps: 20",
                "    base_midprice: 100.0",
                "    size: 1.0",
                "    external_order_probability: 0.7",
                "    vol_window: 20",
                "  population:",
                "    participant_count: 6",
                "    mm_trader_id: MM0",
                "    counterparty_prefix: CP",
                "    toxic_activation_prob: 1.0",
                "    toxic_aggression_ticks: 3",
                "  agents:",
                "    A:",
                "      spread: 2.0",
                "    B:",
                "      base_spread: 2.0",
                "      inventory_skew_per_unit: 0.08",
                "      max_skew: 1.5",
                "      vol_spread_multiplier: 20.0",
                "      min_spread: 1.0",
                "      max_spread: 8.0",
                "    competitor_agent: A",
                "    competitor_params:",
                "      spread: 2.0",
                "metrics:",
                "  markout_horizon: 5",
                "  inventory_threshold: 3.0",
                "single_run:",
                f"  agent: {agent}",
                "  agent_params: {}",
                "  regime:",
                "    volatility: low",
                "    toxicity: 0",
                "    competition: solo",
                "  episode_steps: 20",
                "  base_midprice: 100.0",
                "  size: 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_run_single_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "single.yaml"
    _write_bse_config(config_path, agent="A")
    code = run_single(config_path=config_path, seed=123, output_dir=tmp_path)
    assert code == 0


def test_run_single_agent_b_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "single_b.yaml"
    _write_bse_config(config_path, agent="B")
    code = run_single(config_path=config_path, seed=123, output_dir=tmp_path)
    assert code == 0


def test_run_grid_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "grid.yaml"
    _write_bse_config(config_path, agent="A")
    code = run_grid(config_path=config_path, max_runs=2, output_dir=tmp_path)
    assert code == 0


def test_run_metrics_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "metrics.yaml"
    _write_bse_config(config_path, agent="A")
    assert run_single(config_path=config_path, seed=101, output_dir=tmp_path) == 0
    assert run_single(config_path=config_path, seed=202, output_dir=tmp_path) == 0

    assert (
        run_metrics(
            config_path=config_path,
            runs_dir=tmp_path / "runs",
            output_dir=tmp_path / "metrics",
        )
        == 0
    )
    assert (tmp_path / "metrics" / "run_metrics.csv").exists()
    assert (tmp_path / "metrics" / "condition_metrics.csv").exists()
    assert (tmp_path / "metrics" / "metrics_meta.json").exists()


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(100, 2) == derive_seed(100, 2)

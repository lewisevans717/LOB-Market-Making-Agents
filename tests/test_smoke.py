from pathlib import Path

from lob_market_making_agents.experiments.run import build_parser, run_grid, run_metrics, run_single
from lob_market_making_agents.utils.seeding import derive_seed


def test_parser_includes_subcommands() -> None:
    parser = build_parser()
    help_text = parser.format_help()
    assert "single" in help_text
    assert "grid" in help_text
    assert "metrics" in help_text


def test_run_single_smoke(tmp_path: Path) -> None:
    config_path = Path("configs/experiment_mvp.yaml")
    code = run_single(config_path=config_path, seed=123, output_dir=tmp_path)
    assert code == 0


def test_run_grid_smoke() -> None:
    config_path = Path("configs/experiment_mvp.yaml")
    code = run_grid(config_path=config_path, max_runs=2)
    assert code == 0


def test_run_metrics_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_metrics_smoke.yaml"
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
                "metrics:",
                "  markout_horizon: 5",
                "  inventory_threshold: 3.0",
                "single_run:",
                "  agent: A",
                "  agent_params:",
                "    spread: 1.0",
                "  regime:",
                "    volatility: low",
                "    toxicity: 0",
                "    competition: solo",
                "  episode_steps: 25",
                "  base_midprice: 100.0",
                "  size: 1.0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert run_single(config_path=config_path, seed=123, output_dir=tmp_path) == 0
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


def test_run_single_with_agent_b_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment_b.yaml"
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
                "  agent: B",
                "  agent_params:",
                "    base_spread: 1.0",
                "    inventory_skew_per_unit: 0.05",
                "    max_skew: 0.75",
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
    code = run_single(config_path=config_path, seed=123, output_dir=tmp_path)
    assert code == 0


def test_derive_seed_is_deterministic() -> None:
    assert derive_seed(100, 2) == derive_seed(100, 2)

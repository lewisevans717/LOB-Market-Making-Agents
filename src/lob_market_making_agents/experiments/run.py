"""Experiment runner CLI for BSE-first market-making experiments."""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lob_market_making_agents.env import (
    PopulationConfig,
    RegimeConfig,
    SessionConfig,
    get_bse_commit,
    run_bse_session,
)
from lob_market_making_agents.experiments.artifacts import build_run_id, write_run_artifacts
from lob_market_making_agents.metrics import (
    DEFAULT_INVENTORY_THRESHOLD,
    DEFAULT_MARKOUT_HORIZON,
    MetricsSettings,
    run_metrics_pipeline,
)
from lob_market_making_agents.utils.config import load_yaml_config
from lob_market_making_agents.utils.seeding import set_global_seed


DEFAULT_CONFIG = Path("configs/experiment_mvp.yaml")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_RUNS_DIR = DEFAULT_OUTPUT_DIR / "runs"
DEFAULT_METRICS_DIR = DEFAULT_OUTPUT_DIR / "metrics"


@dataclass(frozen=True)
class RunSpec:
    config_name: str
    agent: str
    agent_params: dict[str, Any]
    regime: RegimeConfig
    session_config: SessionConfig
    population_config: PopulationConfig
    agent_settings: dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lobmm",
        description="Run LOB market-making experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run one configured BSE session.")
    single.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    grid = subparsers.add_parser("grid", help="Run a BSE grid from experiment config.")
    grid.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    grid.add_argument("--max-runs", type=int, default=10)
    grid.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    grid.add_argument(
        "--force",
        action="store_true",
        help="Re-execute conditions whose summary.json already exists (default: skip).",
    )

    pareto = subparsers.add_parser(
        "pareto",
        help="Sweep Agent C inventory penalty lambda × volatility × seed at fixed (toxicity, competition).",
    )
    pareto.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    pareto.add_argument("--max-runs", type=int, default=1080)
    pareto.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "runs" / "pareto")
    pareto.add_argument("--force", action="store_true")
    pareto.add_argument(
        "--agents",
        type=str,
        default="C,C_PLUS",
        help="Comma-separated agent labels to sweep (default: C,C_PLUS).",
    )

    transfer = subparsers.add_parser(
        "transfer",
        help="Regime-shift transfer: train Agent C in source, evaluate frozen policy in target (and reverse).",
    )
    transfer.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    transfer.add_argument("--max-runs", type=int, default=240)
    transfer.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "runs" / "transfer")
    transfer.add_argument("--force", action="store_true")
    transfer.add_argument(
        "--agents",
        type=str,
        default="C,C_PLUS",
        help="Comma-separated agent labels to sweep (default: C,C_PLUS).",
    )

    metrics = subparsers.add_parser("metrics", help="Aggregate metrics from BSE-native run artifacts.")
    metrics.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    metrics.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    metrics.add_argument("--output-dir", type=Path, default=DEFAULT_METRICS_DIR)
    metrics.add_argument("--markout-horizon", type=int, default=None)
    metrics.add_argument("--inventory-threshold", type=float, default=None)

    return parser


def _first_or_default(values: Any, default: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    if values is None:
        return default
    return values


def _as_list(values: Any, fallback: list[Any]) -> list[Any]:
    if isinstance(values, list) and values:
        return values
    if values is None:
        return fallback
    return [values]


def _base_blocks(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    bse_cfg = dict(config.get("bse", {}))
    session_cfg = dict(bse_cfg.get("session", {}))
    population_cfg = dict(bse_cfg.get("population", {}))
    agents_cfg = dict(bse_cfg.get("agents", {}))
    return bse_cfg, session_cfg, population_cfg, agents_cfg


def _session_config_from_maps(session_cfg: dict[str, Any], single_run: dict[str, Any]) -> SessionConfig:
    return SessionConfig(
        episode_steps=int(single_run.get("episode_steps", session_cfg.get("episode_steps", 200))),
        base_midprice=float(single_run.get("base_midprice", session_cfg.get("base_midprice", 100.0))),
        size=float(single_run.get("size", session_cfg.get("size", 1.0))),
        external_order_probability=float(
            session_cfg.get("external_order_probability", 0.7)
        ),
        vol_window=int(session_cfg.get("vol_window", 20)),
    )


def _population_config_from_map(population_cfg: dict[str, Any]) -> PopulationConfig:
    return PopulationConfig(
        participant_count=int(population_cfg.get("participant_count", 8)),
        mm_trader_id=str(population_cfg.get("mm_trader_id", "MM0")),
        counterparty_prefix=str(population_cfg.get("counterparty_prefix", "CP")),
        toxic_activation_prob=float(population_cfg.get("toxic_activation_prob", 1.0)),
        toxic_aggression_ticks=int(population_cfg.get("toxic_aggression_ticks", 3)),
    )


def _single_spec_from_config(config: dict[str, Any]) -> RunSpec:
    regimes = config.get("regimes", {})
    single_run = dict(config.get("single_run", {}))
    regime_override = dict(single_run.get("regime", {}))
    _, session_cfg, population_cfg, agents_cfg = _base_blocks(config)

    volatility = regime_override.get("volatility", _first_or_default(regimes.get("volatility"), "low"))
    toxicity = regime_override.get("toxicity", _first_or_default(regimes.get("toxicity"), 0.0))
    competition = regime_override.get("competition", _first_or_default(regimes.get("competition"), "solo"))

    configured_agents = _as_list(config.get("agents"), ["A", "B"])
    agent = str(single_run.get("agent", configured_agents[0])).upper()

    base_agent_params = dict(agents_cfg.get(agent, {}))
    agent_params = {**base_agent_params, **dict(single_run.get("agent_params", {}))}

    return RunSpec(
        config_name=str(config.get("name", "default")),
        agent=agent,
        agent_params=agent_params,
        regime=RegimeConfig(
            volatility=str(volatility),
            toxicity=float(toxicity),
            competition=str(competition),
        ),
        session_config=_session_config_from_maps(session_cfg=session_cfg, single_run=single_run),
        population_config=_population_config_from_map(population_cfg),
        agent_settings={
            "competitor_agent": str(agents_cfg.get("competitor_agent", "A")),
            "competitor_params": dict(agents_cfg.get("competitor_params", {"spread": 2.0})),
        },
    )


def _run_spec_for_condition(
    *,
    config: dict[str, Any],
    agent: str,
    volatility: str,
    toxicity: float,
    competition: str,
) -> RunSpec:
    _, session_cfg, population_cfg, agents_cfg = _base_blocks(config)
    agent_key = agent.upper()
    return RunSpec(
        config_name=str(config.get("name", "default")),
        agent=agent_key,
        agent_params=dict(agents_cfg.get(agent_key, {})),
        regime=RegimeConfig(
            volatility=str(volatility),
            toxicity=float(toxicity),
            competition=str(competition),
        ),
        session_config=SessionConfig(
            episode_steps=int(session_cfg.get("episode_steps", 200)),
            base_midprice=float(session_cfg.get("base_midprice", 100.0)),
            size=float(session_cfg.get("size", 1.0)),
            external_order_probability=float(session_cfg.get("external_order_probability", 0.7)),
            vol_window=int(session_cfg.get("vol_window", 20)),
        ),
        population_config=_population_config_from_map(population_cfg),
        agent_settings={
            "competitor_agent": str(agents_cfg.get("competitor_agent", "A")),
            "competitor_params": dict(agents_cfg.get("competitor_params", {"spread": 2.0})),
        },
    )


def _execute_run(spec: RunSpec, seed: int, output_dir: Path, config_path: Path) -> tuple[str, dict[str, float], Path]:
    set_global_seed(seed)
    session = run_bse_session(
        seed=seed,
        agent_name=spec.agent,
        agent_params=spec.agent_params,
        regime=spec.regime,
        session_config=spec.session_config,
        population_config=spec.population_config,
        agent_settings=spec.agent_settings,
    )

    run_id = build_run_id(
        config_name=spec.config_name,
        agent=spec.agent,
        volatility=spec.regime.volatility,
        toxicity=spec.regime.toxicity,
        competition=spec.regime.competition,
        seed=seed,
    )
    run_dir = output_dir / "runs" / run_id

    meta = {
        "run_id": run_id,
        "config_name": spec.config_name,
        "config_path": str(config_path),
        "runtime": "bse",
        "bse_commit": get_bse_commit(),
        "seed": int(seed),
        "agent": spec.agent,
        "agent_params": spec.agent_params,
        "regime": {
            "volatility": spec.regime.volatility,
            "toxicity": spec.regime.toxicity,
            "competition": spec.regime.competition,
        },
        "session": {
            "episode_steps": spec.session_config.episode_steps,
            "base_midprice": spec.session_config.base_midprice,
            "size": spec.session_config.size,
            "external_order_probability": spec.session_config.external_order_probability,
            "vol_window": spec.session_config.vol_window,
        },
        "population": {
            "participant_count": spec.population_config.participant_count,
            "mm_trader_id": spec.population_config.mm_trader_id,
            "counterparty_prefix": spec.population_config.counterparty_prefix,
            "toxic_activation_prob": spec.population_config.toxic_activation_prob,
            "toxic_aggression_ticks": spec.population_config.toxic_aggression_ticks,
        },
    }

    write_run_artifacts(
        run_dir=run_dir,
        tape_rows=list(session.tape_rows),
        lob_frames=list(session.lob_frames),
        mm_state_rows=list(session.mm_state_rows),
        competitor_state_rows=list(session.competitor_state_rows),
        summary=session.summary,
        meta=meta,
    )
    return run_id, session.summary, run_dir


def run_single(config_path: Path, seed: int, output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    config = load_yaml_config(config_path)
    spec = _single_spec_from_config(config)

    run_id, summary, run_dir = _execute_run(
        spec=spec,
        seed=seed,
        output_dir=output_dir,
        config_path=config_path,
    )
    print(
        f"[single] run_id={run_id} steps={int(summary['steps'])} fills={int(summary['fills'])} "
        f"final_pnl={summary['final_pnl']:.4f} output={run_dir}"
    )
    return 0


def run_grid(
    config_path: Path,
    max_runs: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> int:
    config = load_yaml_config(config_path)
    config_name = str(config.get("name", "default"))

    seeds = [int(seed) for seed in _as_list(config.get("seeds"), [42])]
    agents = [str(agent).upper() for agent in _as_list(config.get("agents"), ["A", "B"]) if str(agent).upper() in {"A", "B", "C", "C_PLUS"}]
    regimes = config.get("regimes", {})
    volatilities = [str(v) for v in _as_list(regimes.get("volatility"), ["low"])]
    toxicities = [float(t) for t in _as_list(regimes.get("toxicity"), [0.0])]
    competitions = [str(c) for c in _as_list(regimes.get("competition"), ["solo"])]

    planned_total = len(agents) * len(volatilities) * len(toxicities) * len(competitions) * len(seeds)
    executed = 0
    skipped = 0

    for agent, volatility, toxicity, competition, seed in itertools.product(
        agents,
        volatilities,
        toxicities,
        competitions,
        seeds,
    ):
        if executed >= max_runs:
            break

        # WHY summary.json as the completion marker: the artifact writer flushes
        # summary.json *last* in `write_run_artifacts`, so its presence is a
        # post-fsync guarantee that every other artefact for the run is on disk.
        # No new sentinel file needed. build_run_id is deterministic in its
        # inputs, so a re-invocation with the same config + seed list lands on
        # the same run_dir and skips cleanly — this is what makes the grid
        # CLI safely resumable across crashes.
        run_id = build_run_id(
            config_name=config_name,
            agent=agent,
            volatility=volatility,
            toxicity=float(toxicity),
            competition=competition,
            seed=int(seed),
        )
        run_dir = output_dir / "runs" / run_id
        if not force and (run_dir / "summary.json").exists():
            skipped += 1
            print(f"[grid] skip run_id={run_id} (already complete)")
            continue

        spec = _run_spec_for_condition(
            config=config,
            agent=agent,
            volatility=volatility,
            toxicity=toxicity,
            competition=competition,
        )
        run_id, summary, run_dir = _execute_run(
            spec=spec,
            seed=int(seed),
            output_dir=output_dir,
            config_path=config_path,
        )
        executed += 1
        print(
            f"[grid] {executed}/{min(planned_total, max_runs)} run_id={run_id} "
            f"final_pnl={summary['final_pnl']:.4f} output={run_dir}"
        )

    print(
        f"[grid] planned={planned_total} executed={executed} skipped={skipped} max_runs={max_runs}"
    )
    return 0


def _format_lambda_label(lam: float) -> str:
    """Stable, filesystem-safe label for a lambda value (e.g. 0.01 -> 'lam0p010000')."""
    return f"lam{lam:.6f}".replace(".", "p")


def run_pareto(
    config_path: Path,
    max_runs: int,
    output_dir: Path,
    force: bool = False,
    agents: str = "C,C_PLUS",
) -> int:
    config = load_yaml_config(config_path)
    base_name = str(config.get("name", "default"))
    pareto_cfg = dict(config.get("pareto", {}))

    lambdas = [float(x) for x in _as_list(pareto_cfg.get("lambdas"), [0.01])]
    volatilities = [str(v) for v in _as_list(pareto_cfg.get("volatilities"), ["low", "medium", "high"])]
    toxicity = float(pareto_cfg.get("toxicity", 10.0))
    competition = str(pareto_cfg.get("competition", "solo"))
    seeds = [int(seed) for seed in _as_list(config.get("seeds"), [42])]
    agent_keys = [a.strip().upper() for a in agents.split(",") if a.strip()]

    _, session_cfg, population_cfg, agents_cfg = _base_blocks(config)
    competitor_settings = {
        "competitor_agent": str(agents_cfg.get("competitor_agent", "A")),
        "competitor_params": dict(agents_cfg.get("competitor_params", {"spread": 2.0})),
    }

    planned_total = len(agent_keys) * len(lambdas) * len(volatilities) * len(seeds)
    executed = 0
    skipped = 0

    for agent_key, lam, volatility, seed in itertools.product(agent_keys, lambdas, volatilities, seeds):
        if executed >= max_runs:
            break

        config_name = f"{base_name}_pareto_{_format_lambda_label(lam)}"
        run_id = build_run_id(
            config_name=config_name,
            agent=agent_key,
            volatility=volatility,
            toxicity=toxicity,
            competition=competition,
            seed=int(seed),
        )
        run_dir = output_dir / "runs" / run_id
        if not force and (run_dir / "summary.json").exists():
            skipped += 1
            print(f"[pareto] skip run_id={run_id} (already complete)")
            continue

        base_agent_params = dict(agents_cfg.get(agent_key, {}))
        agent_params = {**base_agent_params, "lambda_penalty": float(lam)}
        spec = RunSpec(
            config_name=config_name,
            agent=agent_key,
            agent_params=agent_params,
            regime=RegimeConfig(volatility=volatility, toxicity=toxicity, competition=competition),
            session_config=SessionConfig(
                episode_steps=int(session_cfg.get("episode_steps", 200)),
                base_midprice=float(session_cfg.get("base_midprice", 100.0)),
                size=float(session_cfg.get("size", 1.0)),
                external_order_probability=float(session_cfg.get("external_order_probability", 0.7)),
                vol_window=int(session_cfg.get("vol_window", 20)),
            ),
            population_config=_population_config_from_map(population_cfg),
            agent_settings=competitor_settings,
        )
        run_id, summary, run_dir = _execute_run(
            spec=spec,
            seed=int(seed),
            output_dir=output_dir,
            config_path=config_path,
        )
        executed += 1
        print(
            f"[pareto] {executed}/{min(planned_total, max_runs)} "
            f"agent={agent_key} lambda={lam} vol={volatility} seed={seed} "
            f"final_pnl={summary['final_pnl']:.4f} output={run_dir}"
        )

    print(
        f"[pareto] planned={planned_total} executed={executed} skipped={skipped} max_runs={max_runs}"
    )
    return 0


def _regime_label(regime: dict[str, Any]) -> str:
    """Compact, filesystem-safe label for a regime dict."""
    tox = int(round(float(regime.get("toxicity", 0))))
    return f"{regime.get('volatility', 'low')}_tox{tox}_{regime.get('competition', 'solo')}"


def run_transfer(
    config_path: Path,
    max_runs: int,
    output_dir: Path,
    force: bool = False,
    agents: str = "C,C_PLUS",
) -> int:
    """Train each learner agent in a source regime, freeze, evaluate in target regime (both directions).

    For each (agent, seed) pair, runs four sessions:
      1. train_<source>: standard learning run; saves Q-table to policies/.
      2. transferred_<source>_to_<target>: loads (1)'s policy, freeze=True, evaluates in target.
      3. train_<target>: standard learning run; saves Q-table.
      4. transferred_<target>_to_<source>: loads (3)'s policy, freeze=True, evaluates in source.

    Native baselines (train and evaluate in the same regime, no transfer) are
    intentionally NOT re-run: the core grid already provides them at matched
    seeds, and the analysis notebook joins by seed × regime.
    """
    config = load_yaml_config(config_path)
    base_name = str(config.get("name", "default"))
    transfer_cfg = dict(config.get("transfer", {}))
    source_regime = dict(transfer_cfg.get("source_regime", {"volatility": "low", "toxicity": 0, "competition": "solo"}))
    target_regime = dict(transfer_cfg.get("target_regime", {"volatility": "high", "toxicity": 30, "competition": "solo"}))
    seeds = [int(seed) for seed in _as_list(config.get("seeds"), [42])]
    agent_keys = [a.strip().upper() for a in agents.split(",") if a.strip()]

    _, session_cfg, population_cfg, agents_cfg = _base_blocks(config)
    competitor_settings = {
        "competitor_agent": str(agents_cfg.get("competitor_agent", "A")),
        "competitor_params": dict(agents_cfg.get("competitor_params", {"spread": 2.0})),
    }

    src_label = _regime_label(source_regime)
    tgt_label = _regime_label(target_regime)
    policies_dir = output_dir / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)

    def _build_spec(
        *,
        agent_key: str,
        regime: dict[str, Any],
        extra_params: dict[str, Any],
        config_name: str,
    ) -> RunSpec:
        base_agent_params = dict(agents_cfg.get(agent_key, {}))
        return RunSpec(
            config_name=config_name,
            agent=agent_key,
            agent_params={**base_agent_params, **extra_params},
            regime=RegimeConfig(
                volatility=str(regime["volatility"]),
                toxicity=float(regime["toxicity"]),
                competition=str(regime["competition"]),
            ),
            session_config=SessionConfig(
                episode_steps=int(session_cfg.get("episode_steps", 200)),
                base_midprice=float(session_cfg.get("base_midprice", 100.0)),
                size=float(session_cfg.get("size", 1.0)),
                external_order_probability=float(session_cfg.get("external_order_probability", 0.7)),
                vol_window=int(session_cfg.get("vol_window", 20)),
            ),
            population_config=_population_config_from_map(population_cfg),
            agent_settings=competitor_settings,
        )

    phases = (
        ("train", source_regime, src_label, "train_calm"),
        ("transferred", target_regime, tgt_label, "transferred_calm_to_stressed"),
        ("train", target_regime, tgt_label, "train_stressed"),
        ("transferred", source_regime, src_label, "transferred_stressed_to_calm"),
    )

    planned_total = len(agent_keys) * len(phases) * len(seeds)
    executed = 0
    skipped = 0

    for agent_key in agent_keys:
        for seed in seeds:
            # Policies are per-(agent, seed) so a C-trained policy is never loaded by C+.
            src_policy = policies_dir / f"seed{seed}_{agent_key}_{src_label}.json"
            tgt_policy = policies_dir / f"seed{seed}_{agent_key}_{tgt_label}.json"

            for phase_kind, regime, regime_lbl, phase_tag in phases:
                if executed >= max_runs:
                    break

                config_name = f"{base_name}_transfer_{phase_tag}"
                run_id = build_run_id(
                    config_name=config_name,
                    agent=agent_key,
                    volatility=str(regime["volatility"]),
                    toxicity=float(regime["toxicity"]),
                    competition=str(regime["competition"]),
                    seed=int(seed),
                )
                run_dir = output_dir / "runs" / run_id
                if not force and (run_dir / "summary.json").exists():
                    skipped += 1
                    print(f"[transfer] skip run_id={run_id} (already complete)")
                    continue

                if phase_kind == "train":
                    save_target = src_policy if regime_lbl == src_label else tgt_policy
                    extra_params = {"save_policy_path": str(save_target)}
                else:
                    load_source = tgt_policy if regime_lbl == src_label else src_policy
                    if not load_source.exists():
                        raise RuntimeError(
                            f"transfer phase {phase_tag} for agent={agent_key} seed={seed} "
                            f"requires {load_source} which does not exist — run the train phase first."
                        )
                    extra_params = {"initial_policy_path": str(load_source), "freeze": True}

                spec = _build_spec(agent_key=agent_key, regime=regime, extra_params=extra_params, config_name=config_name)
                run_id, summary, run_dir = _execute_run(
                    spec=spec,
                    seed=int(seed),
                    output_dir=output_dir,
                    config_path=config_path,
                )
                executed += 1
                print(
                    f"[transfer] {executed}/{min(planned_total, max_runs)} "
                    f"agent={agent_key} phase={phase_tag} seed={seed} "
                    f"final_pnl={summary['final_pnl']:.4f} output={run_dir}"
                )

            if executed >= max_runs:
                break
        if executed >= max_runs:
            break

    print(
        f"[transfer] planned={planned_total} executed={executed} skipped={skipped} max_runs={max_runs}"
    )
    return 0


def run_metrics(
    *,
    config_path: Path,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    output_dir: Path = DEFAULT_METRICS_DIR,
    markout_horizon: int | None = None,
    inventory_threshold: float | None = None,
) -> int:
    config = load_yaml_config(config_path)
    metrics_cfg = dict(config.get("metrics", {}))

    resolved_horizon = int(
        markout_horizon
        if markout_horizon is not None
        else metrics_cfg.get("markout_horizon", DEFAULT_MARKOUT_HORIZON)
    )
    resolved_threshold = float(
        inventory_threshold
        if inventory_threshold is not None
        else metrics_cfg.get("inventory_threshold", DEFAULT_INVENTORY_THRESHOLD)
    )

    if resolved_horizon <= 0:
        print(f"[metrics] invalid markout horizon: {resolved_horizon} (must be positive)")
        return 2

    settings = MetricsSettings(
        markout_horizon=resolved_horizon,
        inventory_threshold=resolved_threshold,
    )

    try:
        report = run_metrics_pipeline(runs_dir=runs_dir, output_dir=output_dir, settings=settings)
    except ValueError as exc:
        print(f"[metrics] error: {exc}")
        return 1

    print(
        "[metrics] "
        f"runs_processed={report['runs_processed']} "
        f"groups={report['group_count']} "
        f"output={output_dir}"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "single":
        return run_single(config_path=args.config, seed=args.seed, output_dir=args.output_dir)
    if args.command == "grid":
        return run_grid(
            config_path=args.config,
            max_runs=args.max_runs,
            output_dir=args.output_dir,
            force=args.force,
        )
    if args.command == "pareto":
        return run_pareto(
            config_path=args.config,
            max_runs=args.max_runs,
            output_dir=args.output_dir,
            force=args.force,
            agents=args.agents,
        )
    if args.command == "transfer":
        return run_transfer(
            config_path=args.config,
            max_runs=args.max_runs,
            output_dir=args.output_dir,
            force=args.force,
            agents=args.agents,
        )
    if args.command == "metrics":
        return run_metrics(
            config_path=args.config,
            runs_dir=args.runs_dir,
            output_dir=args.output_dir,
            markout_horizon=args.markout_horizon,
            inventory_threshold=args.inventory_threshold,
        )

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

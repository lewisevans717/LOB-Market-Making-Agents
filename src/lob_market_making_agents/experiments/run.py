"""Experiment runner CLI (M0 scaffold)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lob_market_making_agents.agents import create_agent
from lob_market_making_agents.env import RegimeConfig, SimulatedLOBEnv
from lob_market_making_agents.env.models import AgentQuote
from lob_market_making_agents.experiments.artifacts import build_run_id, write_run_artifacts
from lob_market_making_agents.utils.config import load_yaml_config
from lob_market_making_agents.utils.seeding import set_global_seed


DEFAULT_CONFIG = Path("configs/experiment_mvp.yaml")
DEFAULT_OUTPUT_DIR = Path("results")


@dataclass(frozen=True)
class SingleRunSpec:
    config_name: str
    agent: str
    agent_params: dict[str, Any]
    regime: RegimeConfig
    episode_steps: int
    base_midprice: float
    size: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lobmm",
        description="Run LOB market-making experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single", help="Run one configured episode.")
    single.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    single.add_argument("--seed", type=int, default=42)
    single.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    grid = subparsers.add_parser("grid", help="Run a grid from experiment config.")
    grid.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    grid.add_argument("--max-runs", type=int, default=10)
    return parser


def _first_or_default(values: Any, default: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    if values is None:
        return default
    return values


def _single_spec_from_config(config: dict[str, Any]) -> SingleRunSpec:
    regimes = config.get("regimes", {})
    single_run = config.get("single_run", {})
    regime_override = single_run.get("regime", {})

    volatility = regime_override.get("volatility", _first_or_default(regimes.get("volatility"), "low"))
    toxicity = regime_override.get("toxicity", _first_or_default(regimes.get("toxicity"), 0))
    competition = regime_override.get("competition", _first_or_default(regimes.get("competition"), "solo"))
    agent = str(single_run.get("agent", _first_or_default(config.get("agents"), "A")))
    agent_params = dict(single_run.get("agent_params", {}))
    if "spread" in single_run and "spread" not in agent_params and agent.upper() == "A":
        agent_params["spread"] = float(single_run["spread"])

    return SingleRunSpec(
        config_name=str(config.get("name", "default")),
        agent=agent,
        agent_params=agent_params,
        regime=RegimeConfig(
            volatility=str(volatility),
            toxicity=float(toxicity),
            competition=str(competition),
        ),
        episode_steps=int(single_run.get("episode_steps", 200)),
        base_midprice=float(single_run.get("base_midprice", 100.0)),
        size=float(single_run.get("size", 1.0)),
    )


def _episode_summary(events: list[dict[str, Any]]) -> dict[str, float]:
    if not events:
        return {
            "steps": 0.0,
            "final_midprice": 0.0,
            "final_inventory": 0.0,
            "final_cash": 0.0,
            "final_pnl": 0.0,
            "fills": 0.0,
        }
    last = events[-1]
    fills = sum(1.0 for event in events if event["fill_side"] != "none")
    return {
        "steps": float(len(events)),
        "final_midprice": float(last["midprice"]),
        "final_inventory": float(last["inventory"]),
        "final_cash": float(last["cash"]),
        "final_pnl": float(last["pnl"]),
        "fills": float(fills),
    }


def _run_episode_with_agent(env: SimulatedLOBEnv, spec: SingleRunSpec, seed: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
    agent = create_agent(agent_name=spec.agent, params=spec.agent_params)
    agent.reset()

    state = env.reset(seed=seed, regime_config=spec.regime)
    events: list[dict[str, Any]] = []
    while True:
        decision = agent.quote(state)
        quote = AgentQuote(bid=decision.bid, ask=decision.ask, size=spec.size)
        state, event, done = env.step(quote)
        events.append(asdict(event))
        if done:
            break
    summary = _episode_summary(events)
    return events, summary


def run_single(config_path: Path, seed: int, output_dir: Path = DEFAULT_OUTPUT_DIR) -> int:
    config = load_yaml_config(config_path)
    spec = _single_spec_from_config(config)
    set_global_seed(seed)
    env = SimulatedLOBEnv(base_midprice=spec.base_midprice, episode_steps=spec.episode_steps)
    events, summary = _run_episode_with_agent(env=env, spec=spec, seed=seed)
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
        "config_path": str(config_path),
        "config_name": spec.config_name,
        "seed": seed,
        "agent": spec.agent,
        "regime": {
            "volatility": spec.regime.volatility,
            "toxicity": spec.regime.toxicity,
            "competition": spec.regime.competition,
        },
        "episode_steps": spec.episode_steps,
        "base_midprice": spec.base_midprice,
        "agent_params": spec.agent_params,
        "size": spec.size,
    }
    write_run_artifacts(run_dir=run_dir, events=events, summary=summary, meta=meta)
    print(
        f"[single] run_id={run_id} events={len(events)} output={run_dir} "
        f"final_pnl={summary['final_pnl']:.4f}"
    )
    return 0


def run_grid(config_path: Path, max_runs: int) -> int:
    config = load_yaml_config(config_path)
    seeds = config.get("seeds", [])
    planned = min(len(seeds), max_runs) if seeds else 0
    print(f"[grid] config={config_path} max_runs={max_runs} planned_runs={planned}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "single":
        return run_single(config_path=args.config, seed=args.seed, output_dir=args.output_dir)
    if args.command == "grid":
        return run_grid(config_path=args.config, max_runs=args.max_runs)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

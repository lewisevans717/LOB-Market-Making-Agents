# Implementation Plan

This plan turns the research spec in `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/README.md` into an executable development roadmap.

## 1. Scope and defaults

- Language: Python 3.11+
- Core outputs: reproducible simulations, per-run logs, aggregated metrics tables, baseline plots
- Initial scope: single-asset BSE simulation, Agents A/B/C, minimum viable matrix (1,620 runs)
- Default run mode: local parallel execution (single machine), deterministic seeds

## 2. Target repository layout (to create first)

- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/src/env/`
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/src/agents/`
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/src/metrics/`
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/src/experiments/`
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/configs/`
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/notebooks/`
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/results/` (gitignored)
- `/Users/lewis/Documents/External Work/LOB Market Maker Agents/Repo/LOB-Market-Making-Agents/tests/`

## 3. Milestones and deliverables

### M0: Project bootstrap (Day 1)

Deliverables:
- Python project scaffold (`pyproject.toml`, package init, CLI entrypoint)
- Tooling (`ruff`, `pytest`, optional `mypy`)
- Standard config loader and seed utility
- `.gitignore` updated for `results/`, caches, notebooks artifacts

Definition of done:
- `pytest` passes with at least one smoke test
- `python -m ... --help` works for experiment runner CLI

### M1: Environment adapter + event logging (Days 2-3)

Deliverables:
- BSE wrapper API in `src/env/`:
  - reset(seed, regime_config)
  - step(agent_actions)
  - episode summary
- Regime controls wired: volatility, toxicity share, competition mode
- Event schema and logging writer (CSV or Parquet)

Definition of done:
- One seeded episode runs end-to-end
- Logs include timestamp, midprice, quotes, fills, inventory, cash, pnl

### M2: Agents A and B (Days 4-5)

Deliverables:
- Agent A fixed-spread baseline
- Agent B inventory-aware skew + volatility-adaptive spread
- Shared agent interface (`quote(state) -> bid, ask, metadata`)

Definition of done:
- Both agents run in identical environment config
- Unit tests cover quote math and inventory-skew direction logic

### M3: Metrics pipeline (Days 6-7)

Deliverables:
- PnL stats: mean/median/std/p5
- Inventory risk stats: mean/var/max abs(q), threshold exposure time
- Microstructure stats: markout at horizon delta_t, fill rate
- Aggregation job from raw run logs to condition-level summary

Definition of done:
- For a 10-seed sample, summary table generates with no manual edits

### M4: Agent C contextual bandit (Days 8-10)

Deliverables:
- Discrete action grid for spread/skew
- Context features: inventory, rolling volatility, flow proxy
- Reward function: delta pnl minus lambda * q^2
- Online policy update loop (epsilon-greedy or UCB baseline)

Definition of done:
- Agent C completes episodes without exploding inventory
- Ablation flag for no inventory penalty

### M5: Experiment runner for full matrix (Days 11-12)

Deliverables:
- Grid executor over agents x vol x toxicity x competition x seed
- Deterministic run IDs and directory structure
- Failure-safe resume capability

Definition of done:
- Dry-run prints full matrix size and planned run IDs
- Sample batch executes and appends outputs safely

### M6: Analysis and reporting (Days 13-14)

Deliverables:
- Notebook(s) for result tables and plots
- Robustness comparison across regimes
- Initial Pareto view (PnL vs inventory risk)

Definition of done:
- Reproducible figure generation from `results/` only
- Ready-to-share summary section for README

## 4. Initial backlog (start here)

P0 (immediate):
1. Bootstrap Python project + lint/test config.
2. Implement env wrapper and logging schema.
3. Implement Agent A + Agent B with unit tests.
4. Implement metrics aggregation job.
5. Add experiment CLI: single run and small grid run.

P1 (next):
1. Implement Agent C bandit.
2. Add ablations (B no vol adaptation, C no inventory penalty).
3. Add full matrix launcher and resume support.

P2 (after MVP):
1. Regime-shift train/eval split tooling.
2. Lambda sweep for Pareto frontier.
3. Optional parallel/distributed execution support.

## 5. Config and data contracts

Required config files:
- `configs/regimes.yaml`: volatility/toxicity/competition definitions
- `configs/agents.yaml`: parameter sets for A/B/C
- `configs/experiment_mvp.yaml`: matrix and seeds

Run output contract:
- One folder per run ID with:
  - `events.csv` (or `events.parquet`)
  - `summary.json`
  - `meta.json` (agent, regime, seed, commit hash if available)

## 6. Testing strategy

- Unit tests:
  - quote logic, spread/skew bounds, reward calculation
- Integration tests:
  - one short deterministic episode per agent
- Regression checks:
  - seeded snapshot metrics for a small fixed batch

## 7. Risks and mitigations

- BSE integration ambiguity:
  - Mitigation: lock wrapper interface early and test with smoke episodes.
- Long runtime for 1,620 runs:
  - Mitigation: add fast mode and resume support before full sweep.
- Bandit instability:
  - Mitigation: cap inventory/skew and start with conservative epsilon schedule.

## 8. Execution order for the next development session

1. Create project scaffold and directory structure.
2. Implement env wrapper + run logger.
3. Ship Agent A/B and run a 10-seed pilot.
4. Build metrics aggregation and validate outputs.
5. Then start Agent C.

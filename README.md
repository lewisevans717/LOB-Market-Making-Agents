# Agentic Market Making in a Simulated Limit Order Book (BSE): Profit–Risk Trade-offs Under Regime Shifts

## 1) Goal and research question

### Goal
Design and evaluate multiple market-making agents in a simulated limit order book (LOB) using the **Bristol Stock Exchange (BSE)** environment. The focus is on how quoting-policy choices affect **profitability (PnL)**, **inventory risk**, and **adverse selection** across different market regimes.

### Research question (v1)
**How do different market-making policies trade off profitability and risk under changing market regimes (volatility, toxic/informed flow, competition)?**

### Hypotheses to test
- **H1:** Inventory-aware skew reduces tail inventory risk with limited PnL sacrifice relative to a fixed-spread baseline.
- **H2:** Volatility-adaptive spreads improve robustness in high-volatility regimes.
- **H3:** Policies tuned in one regime degrade under regime shift; robustness varies by design.

---

## 2) Environment and market setup

### Environment
- **Bristol Stock Exchange (BSE)**: a lightweight simulator of a continuous double auction / limit order book for one asset.

### Market participants (population/agents)
- **Market makers**: start with 1, possibly up to 3 depending on competition experiments.
- **Noise/liquidity traders**: provide background flow and liquidity.
- **Informed/toxic traders** *(regime parameter)*: directional traders designed to increase adverse selection risk for market makers.

### Regime knobs (experimental factors)
1. **Volatility:** low / medium / high (midprice noise / dynamics).
2. **Toxicity:** 0%, 10%, 30% informed flow share (fraction of informed/toxic traders).
3. **Competition:** solo MM vs MM + competitor(s) (different strategies, or same strategy with different params).

---

## 3) MM Agents

### Agent A: Fixed-spread market maker (baseline)
- Quotes around the midprice with constant spread:  
  `bid = mid − s/2`, `ask = mid + s/2`
- Provides a simple baseline for comparison.

### Agent B: Inventory-aware market maker (heuristic, Avellaneda–Stoikov-inspired)
- Maintains inventory `q`.
- **Skews** quotes to mean-revert inventory:
  - if `q > 0` (long), reduce ask and reduce bid (bias toward selling / discourage buying).
  - if `q < 0` (short), increase bid and increase ask (bias toward buying back / discourage selling).
- **Adapts spread** using a volatility proxy (e.g., rolling standard deviation of midprice).

### Agent C: Contextual bandit market maker (learning / agentic)
- Observes a compact state vector, e.g.:
  - inventory `q`
  - volatility proxy (rolling σ of midprice returns)
  - order-flow / imbalance proxy (if available; otherwise recent price move or trade sign imbalance)
- Chooses an action from a discrete grid:
  - `(spread_level, skew_level)` where spread ∈ {0.5, 1.0, 1.5, 2.0} × skew ∈ {-2, -1, 0, +1, +2}
- Reward shaping (risk-aware):
  - `reward = ΔPnL − λ · q²` (or `−λ·|q|`) to penalize unbounded inventory accumulation.

**Potential ablations:**
- Agent B without volatility adaptation.
- Agent C without the inventory penalty term (to show why risk-aware reward matters).

---

## 4) Metrics

### Profitability
- Mean and median PnL
- PnL variance / standard deviation
- Tail outcomes (e.g., 5th percentile PnL)
- Drawdown proxy (if you track equity curve)

### Risk and stability
- Inventory distribution: mean, variance, max \(|q|\)
- Time spent above an inventory threshold (exposure time in risky states)

### Microstructure / adverse selection
- **Markout:** midprice change after a fill at horizon Δt (picked off vs spread capture)
- Fill rate (trades per unit time / per quote)
- Quote cancellation rate (if modeled)

### Statistical practice
- Multiple seeds per condition (e.g., 30+)
- Confidence intervals (bootstrap is acceptable) and variance reporting

---

## 5) Experiment matrix

### Minimum viable
- Agents: A, B, C
- Volatility: 3 levels (low/med/high)
- Toxicity: 3 levels (0/10/30%)
- Competition: 2 levels (solo vs competitor(s))
- Seeds: 30 (per condition)

Total runs:  
`3 agents × 3 vol × 3 toxicity × 2 competition × 30 seeds = 1,620 runs`

### Stretch goal (if time)
- **Regime shift robustness:** tune/learn under one regime (e.g., low vol, low toxicity), evaluate under shifted regimes (e.g., high vol and/or high toxicity).
- **Pareto analysis:** sweep risk penalty `λ` to produce a PnL vs inventory-risk frontier.

---

## 6) Engineering plan

### Repo structure
- `src/env/` — BSE wrappers, configuration, market setup
- `src/agents/` — agent implementations (A/B/C)
- `src/metrics/` — PnL, inventory, markout, reporting utility functions
- `src/experiments/` — grid runner, seeding, logging, results
- `configs/` — YAML/JSON regime definitions and agent parameters
- `notebooks/` — analysis notebooks to generate figures/tables
- `README.md` — problem statement, how to run, core results plots, limitations
- `results/` — generated outputs (gitignored)

### Reproducibility checklist
- Single command to reproduce a standard experiment suite (with seeds)
- Deterministic config files for each regime
- Structured logs with schema documented

---

## 7) Running the CLI

Initialize BSE submodule (required for `env.mode: bse`):

```bash
git submodule update --init --recursive
```

Run a single seeded episode:

```bash
PYTHONPATH=src python -m lob_market_making_agents.experiments.run single --config configs/experiment_mvp.yaml --seed 42
```

Backend selection is configured in `configs/experiment_mvp.yaml`:
- `env.mode: simulated` (default)
- `env.mode: bse` (BSE-backed adapter)

Aggregate metrics from run artifacts:

```bash
PYTHONPATH=src python -m lob_market_making_agents.experiments.run metrics --config configs/experiment_mvp.yaml --runs-dir results/runs --output-dir results/metrics
```

Metrics outputs:
- `results/metrics/run_metrics.csv` (one row per run)
- `results/metrics/condition_metrics.csv` (one row per condition)
- `results/metrics/metrics_meta.json` (pipeline metadata and settings)

---

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

## 1.5) Headline results

Generated from `notebooks/01..03` over 10,000-step episodes, 30 seeds per condition × 18 regime cells (3 vol × 3 toxicity × 2 competition). Full figures in `results/figures/`, statistical tables under `results/metrics/`.

**H1 (PnL ranking).** The ordering A < B < {C, C+} holds across all 18 regime cells. Learning agents (C, C+) are the only ones with consistently positive PnL; fixed-spread A sits at a flat ~7k mean and inventory-aware B is either near zero or deeply negative depending on toxicity (see B-collapse note below). C vs C+ depends on which moment is read: pooled across all 540 runs per agent, C+ has the higher mean (613k vs 453k) but C has the higher median (143k vs 84k). At the per-cell level, Welch's t finds C significantly ahead of C+ at every toxicity ≥ 10 cell (p ranging 4e-5 to 2e-8); C+'s pooled mean advantage is driven by fat upside tails in the tox=0 + with_competitor cells (mean 2.4M–4.2M with both higher upside and worse 5th-percentile outcomes than C). Read as: C+'s TD bootstrap gives higher upside and downside variance, while the bandit C is more robust on the median.

![PnL boxplots — with competitor](results/figures/core/pnl_box_with_competitor.png)

**H2 (inventory control).** Pooled mean `max|q|` across all 18 cells: A=134, C=236, C+=465, B=1,163. Fixed-spread A and the bandit C contain inventory tightly; C+'s TD bootstrap is materially looser; B's heuristic skew rule does not contain inventory at all, and the gap is much more visible at 10k than 5k. The ranking A < C < C+ ≪ B is preserved across all three volatility levels.

**B's catastrophic loss is concentrated at toxicity=0 (new at 10k).** With no toxic flow forcing turnover, B's skew rule accumulates inventory without bound over a 10k session: pooled mean PnL ≈ −13.3M and pooled mean `max|q|` ≈ 3,090 across vol levels at tox=0 solo. At toxicity 10 and 30, B is mildly positive (8–17k mean, 18–30k median across vol). This is the cleanest "why a naive market-maker fails" signal in the dataset and only surfaces at 10k episode lengths.

**H3 (regime-shift transfer — direction inverted vs 5k).** For agent C, calm-trained policies transfer to stressed flow cleanly while stressed-trained policies degrade when redeployed to calm flow:
- Native calm 341k → transferred stressed→calm 256k (~25% loss; Wilcoxon p=0.10, median Δ ≈ +48k in favour of native)
- Native stressed 76k → transferred calm→stressed 81k (small positive surprise; Wilcoxon p=0.038, median Δ ≈ −4k)

For agent C+, the picture is dominated by TD-bootstrap brittleness. Stressed→calm transfer produces 3 catastrophic seeds out of 30 (seeds 103, 107, 127 with final PnL −13.9M, −16.2M, −15.1M and `max|q|` of 2,810 / 3,317 / 3,405); the other 27 seeds behave normally (24.0k–669.3k). Mean and median diverge wildly (transferred mean −1.30M, median +152k; native mean 334k, median 287k), so the Wilcoxon signed-rank test (p=0.0013) is the reliable headline. Agent C exhibits no comparable failure mode at any seed — the ≈10% catastrophic failure rate is specific to TD bootstrapping under regime shift. Full table at `results/metrics/transfer/transfer_penalty.csv`; non-parametric tests at `wilcoxon_tests.csv`.

![Pareto frontier — C vs C+](results/figures/pareto/frontier_overlay.png)

**Pareto frontier.** The risk-penalty knee sits in λ ∈ [0, 0.1] for both learning agents (C: 0.001 / 0 / 0.1 at low/medium/high vol; C+: 0.01 / 0.05 / 0 at low/medium/high vol). PnL is dented modestly relative to λ=0 while inventory variance is materially reduced; λ=0.5 hurts both objectives. Story is unchanged from 5k; numbers pulled from `results/metrics/pareto/knee_lambda.csv`.

Pairwise Welch's t and bootstrap CIs (for central-tendency contrasts) are in `results/metrics/core/{pairwise_tests,ci_table}.csv`; Wilcoxon signed-rank is used for the heavy-tailed transfer contrasts. These feed §6 of the report.

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

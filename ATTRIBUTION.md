# Code Attribution

This file declares the provenance of every non-trivial chunk of code in the repository, as required by the COMP3004 / COMP4105 marking rubric.

## 1. Bristol Stock Exchange (BSE)

- **Source**: <https://github.com/davecliff/BristolStockExchange> (Cliff, 2018).
- **License**: MIT (see `third_party/BristolStockExchange/LICENSE`).
- **How included**: vendored as a git submodule at `third_party/BristolStockExchange/`. The file `BSE.py` is loaded at runtime from there by `src/lob_market_making_agents/env/bse_loader.py` — it is **not modified, copied, or adapted**.
- **Symbols used unmodified**:
  - `BSE.Exchange` — the limit-order-book exchange (instantiated in `env/session.py`).
  - `BSE.Trader` — abstract base class subclassed by every market-making agent in this project (`agents/mm_a_bse.py`, `mm_b_bse.py`, `mm_c_bse.py`, `mm_cplus_bse.py`, `toxic_bse.py`).
  - `BSE.Order` — the order datatype, instantiated when our agents emit quotes.
  - `BSE.TraderGiveaway`, `BSE.TraderZIC`, `BSE.TraderShaver` — built-in noise-trader populations (configured by `env/population.py`).
- **What we did NOT take**: no built-in market-maker classes from BSE were used as a reference implementation or starting point. Agents A/B/C/C+ were designed and written from scratch against the `BSE.Trader` interface.

## 2. Code adapted from class examples or labs

**None.** The project does not reuse any code from COMP3004 / COMP4105 lab sessions, lecture notes, or worked examples. Where the agents were inspired by published literature (e.g. Avellaneda & Stoikov 2008 for Agent B's skew formula), the inspiration is a *concept*, not transcribed code, and the relevant equations were re-implemented from the published descriptions.

## 3. Original work

Every file under the following paths is original work for this submission:

- `src/lob_market_making_agents/agents/` — Agents A (fixed-spread), B (Avellaneda-Stoikov inspired inventory-aware), C (ε-greedy contextual bandit), C+ (tabular Q-learning), Toxic-flow trader, factory and module exports.
- `src/lob_market_making_agents/env/` — BSE loader, session runner, population builder, regime knobs, episode-summary writer.
- `src/lob_market_making_agents/experiments/` — `single` / `grid` / `metrics` / `pareto` / `transfer` CLI runners.
- `src/lob_market_making_agents/metrics/` — schema + aggregation pipeline.
- `src/lob_market_making_agents/utils/` — config loader, deterministic per-run RNG seeding.
- `src/lob_market_making_agents/analysis/` — notebook helpers (bootstrap CI, palette, label parsing).
- `tests/` — every smoke and unit test.
- `configs/` — every YAML experiment config.
- `notebooks/` — `01_core_results.ipynb`, `02_pareto.ipynb`, `03_regime_shift.ipynb`.
- `results/` — every figure and CSV under `figures/` and `metrics/` was produced by this codebase.

## 4. Third-party Python dependencies

Standard scientific-Python stack (not adapted, used as published packages): `numpy`, `pandas`, `matplotlib`, `scipy`, `pyyaml`, `pytest`, `jupyter`. See `pyproject.toml` for pinned versions.

## 5. AI assistance

Claude Code was used as a coding assistant during development. The breakdown of human vs AI contribution is detailed in `AI_USAGE.md` per the rubric's transparency requirement.

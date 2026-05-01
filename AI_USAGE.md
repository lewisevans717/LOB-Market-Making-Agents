# AI Usage Statement

The COMP3004 / COMP4105 marking rubric requires an honest, specific declaration of how AI tooling was used during this project. This file is that declaration.

## Tool

- **Claude Code** (Anthropic), model **Claude Opus 4.7**.
- Used as an interactive pair-programmer in the terminal, with full repository access. Every suggestion was reviewed by the author before being committed; nothing was merged blind.

## What AI was used for

### 1. Code drafting (substantial assistance)
- Initial scaffolding of the BSE session wrapper (`env/session.py`), the population builder (`env/population.py`), and the `single` / `grid` / `metrics` CLI dispatch in `experiments/run.py`.
- First-pass implementations of Agents A, B, C, and C+ from spec descriptions written by the author. Agent algorithms (Avellaneda-Stoikov-inspired skew, ε-greedy state discretisation, TD(0) bootstrap) were specified by the author; the AI translated specs into code which the author then reviewed line-by-line.
- The `pareto` and `transfer` CLI subcommands were drafted by AI from author-provided requirements.

### 2. Refactoring and review (high reliance)
- Iteration 2's generalisation of `run_pareto` / `run_transfer` to outer-loop on `--agents` (so both C and C+ could share the same sweep harness) was AI-suggested then human-approved.
- Resume-via-`summary.json`-marker logic in `run_grid` was AI-drafted from the author's "make this resumable without a sentinel file" instruction.

### 3. Tests (drafted, then reviewed)
- The smoke test suite (`tests/test_smoke.py`) was drafted by AI from the author's list of CLI surfaces to cover, then run and adjusted by the author until all 9 cases passed.

### 4. Plotting and analysis notebooks (high assistance)
- The notebook structure, plot layouts, and statistical scaffolding (bootstrap CIs, Welch's t) in `notebooks/01..03` were AI-drafted from a plan the author wrote. The author chose the metrics, hypotheses, and figure layout; the AI handled matplotlib mechanics.

### 5. Documentation (mixed)
- This file, `ATTRIBUTION.md`, and the in-code WHY comments were AI-drafted then audited and edited by the author.
- The implementation plan (`/Users/lewis/.claude/plans/...`) is a co-produced artefact: the author dictated decisions, the AI structured them.

## What AI was NOT used for

- **The research question, hypotheses, and experimental design.** The choice to test fixed → adaptive → bandit → Q-learning as a complexity gradient, the regime axes (volatility / toxicity / competition), the λ Pareto sweep, and the regime-shift transfer protocol were all author decisions.
- **The report itself.** Workstream F (the ≤4000-word coursework write-up) is being drafted by the author. AI may be used for grammar review and formatting checks during the report stage; if so, this file will be updated to reflect that.
- **Interpretation of results.** Conclusions drawn from the figures and CSVs — which hypothesis is supported, what the regime-shift gap means for the agent design, which limitations to acknowledge — are author work.
- **Choice of statistics.** Welch's t, bootstrap percentile CIs, the markout horizon, and the inventory exposure threshold were author choices motivated by the hypotheses.

## Verification level

Every commit listed in `git log` was reviewed by the author before staging. AI-drafted code was treated as a pull-request-from-a-collaborator — read for correctness, run against the test suite, and edited where the author disagreed with the design (e.g. the rejection of a proposed `bootstrap: bool` flag on `MMCBSETrader` in favour of a clean subclass for Agent C+).

## Honesty note

The author considers this declaration to be the rubric's intent: a faithful description of where AI accelerated the work and where the intellectual contribution remained the author's own.

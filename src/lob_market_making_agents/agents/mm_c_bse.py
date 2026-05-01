"""Agent C: contextual-bandit market-maker.

Selects (spread, skew) actions from a discrete grid using epsilon-greedy
exploration over a Q-table indexed by discretised (inventory, volatility,
flow-imbalance) state.  Q-values are updated with incremental-mean rewards
defined as  ΔPnL − λ·q².
"""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from lob_market_making_agents.agents.mm_a_bse import MMABSETrader


class MMCBSETrader(MMABSETrader):
    """Epsilon-greedy contextual-bandit market-maker."""

    ttype = "MM_C"

    # -- defaults (overridable via params dict) --------------------------------
    _DEFAULT_EPSILON = 0.1
    _DEFAULT_LAMBDA = 0.01
    _DEFAULT_SPREAD_LEVELS = [0.5, 1.0, 1.5, 2.0]
    _DEFAULT_SKEW_LEVELS = [-2.0, -1.0, 0.0, 1.0, 2.0]
    _DEFAULT_INV_BINS = [-3.0, -1.0, 1.0, 3.0]
    _DEFAULT_VOL_BINS = [0.3, 0.8]
    _DEFAULT_FLOW_BINS = [-0.3, 0.3]

    def __init__(self, tid: str, params: dict[str, Any], time: float = 0.0) -> None:
        super().__init__(tid=tid, params=params, time=time)
        cfg = dict(params or {})

        # Policy hyper-parameters
        self.epsilon: float = float(cfg.get("epsilon", self._DEFAULT_EPSILON))
        self.lambda_penalty: float = float(cfg.get("lambda_penalty", self._DEFAULT_LAMBDA))

        # Action grid
        spread_levels = [float(s) for s in cfg.get("spread_levels", self._DEFAULT_SPREAD_LEVELS)]
        skew_levels = [float(s) for s in cfg.get("skew_levels", self._DEFAULT_SKEW_LEVELS)]
        self.action_grid: list[tuple[float, float]] = list(product(spread_levels, skew_levels))
        self.n_actions: int = len(self.action_grid)

        # State discretisation bins (threshold arrays for np.searchsorted)
        self.inv_bins = np.asarray(cfg.get("inv_bins", self._DEFAULT_INV_BINS), dtype=float)
        self.vol_bins = np.asarray(cfg.get("vol_bins", self._DEFAULT_VOL_BINS), dtype=float)
        self.flow_bins = np.asarray(cfg.get("flow_bins", self._DEFAULT_FLOW_BINS), dtype=float)

        self.n_inv: int = len(self.inv_bins) + 1
        self.n_vol: int = len(self.vol_bins) + 1
        self.n_flow: int = len(self.flow_bins) + 1
        self.n_states: int = self.n_inv * self.n_vol * self.n_flow

        # Q-table and visit counts
        self.q_table = np.zeros((self.n_states, self.n_actions), dtype=float)
        self.action_counts = np.zeros((self.n_states, self.n_actions), dtype=float)

        # Flow imbalance (set externally by session before make_quotes)
        self.flow_imbalance: float = 0.0

        # Bookkeeping for incremental reward updates
        self.prev_pnl: float = 0.0
        self.last_state_idx: int | None = None
        self.last_action_idx: int | None = None

        # Reproducible per-agent RNG (seeded from tid hash)
        self.rng = np.random.default_rng(abs(hash(tid)) % (2**31))

        # Transfer-learning support: optionally warm-start the Q-table from a
        # previously-saved policy and/or freeze updates so the session evaluates
        # a fixed policy under regime shift (used by the `transfer` CLI).
        self.freeze: bool = bool(cfg.get("freeze", False))
        self.save_policy_path: str | None = cfg.get("save_policy_path") or None
        initial_path = cfg.get("initial_policy_path") or None
        if initial_path is not None:
            self.load_policy(Path(initial_path))
        if self.freeze:
            self.epsilon = 0.0

    # -- policy persistence (used by the regime-shift transfer experiment) -----

    def save_policy(self, path: Path) -> None:
        """Write Q-table, visit counts, and discretisation metadata to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table.tolist(),
            "action_counts": self.action_counts.tolist(),
            "shape": [self.n_states, self.n_actions],
            "inv_bins": self.inv_bins.tolist(),
            "vol_bins": self.vol_bins.tolist(),
            "flow_bins": self.flow_bins.tolist(),
            "action_grid": [list(a) for a in self.action_grid],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def load_policy(self, path: Path) -> None:
        """Restore Q-table and visit counts from a saved policy.

        The state/action grid in the file must match this agent's grid; we
        verify shape rather than silently accepting a different discretisation.
        """
        with Path(path).open(encoding="utf-8") as fh:
            payload = json.load(fh)
        q = np.asarray(payload["q_table"], dtype=float)
        c = np.asarray(payload["action_counts"], dtype=float)
        if q.shape != (self.n_states, self.n_actions):
            raise ValueError(
                f"Q-table shape mismatch: file {q.shape} vs agent {(self.n_states, self.n_actions)}"
            )
        self.q_table = q
        self.action_counts = c

    def finalize(self) -> None:
        """Called by the session at the end of an episode; persists the policy if configured."""
        if self.save_policy_path:
            self.save_policy(Path(self.save_policy_path))

    # -- internals -------------------------------------------------------------

    def _discretize_state(self, vol_proxy: float) -> int:
        """Map continuous (inventory, vol, flow) to a flat state index.

        WHY discretise: a tabular Q-table is the simplest learner that gives an
        interpretable visit-count diagnostic for the report (we can show which
        cells the agent actually explored). Function approximation would hide
        that signal. Bin counts are kept small (5×3×3=45 states × 20 actions =
        900 cells) so the bandit converges within a 5,000-step episode.
        """
        inv_bin = int(np.searchsorted(self.inv_bins, self.inventory))
        vol_bin = int(np.searchsorted(self.vol_bins, vol_proxy))
        flow_bin = int(np.searchsorted(self.flow_bins, self.flow_imbalance))
        return inv_bin * self.n_vol * self.n_flow + vol_bin * self.n_flow + flow_bin

    def _select_action(self, state_idx: int) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state_idx]))

    def _update_q(self, reward: float) -> None:
        """Incremental mean update for the last (state, action) pair."""
        if self.last_state_idx is None or self.freeze:
            return
        s, a = self.last_state_idx, self.last_action_idx
        self.action_counts[s, a] += 1
        n = self.action_counts[s, a]
        self.q_table[s, a] += (reward - self.q_table[s, a]) / n

    # -- quote interface (called by session loop) ------------------------------

    def quote_prices(self, midprice: float, volatility_proxy: float = 0.0) -> tuple[int, int]:
        """Select spread/skew via bandit policy and return (bid, ask)."""
        # 1. Compute reward from previous step and update Q
        # WHY q² (not |q|): squaring penalises large positions super-linearly so
        # the bandit treats inventory risk as a variance cost rather than a
        # linear holding fee — matches the standard market-making utility in
        # Cartea/Jaimungal/Penalva (2015) and produces a smoother gradient for
        # ε-greedy to climb.
        current_pnl = self.pnl(midprice)
        reward = (current_pnl - self.prev_pnl) - self.lambda_penalty * (self.inventory ** 2)
        self._update_q(reward)
        self.prev_pnl = current_pnl

        # 2. Observe current state
        state_idx = self._discretize_state(volatility_proxy)

        # 3. Choose action
        action_idx = self._select_action(state_idx)
        self.last_state_idx = state_idx
        self.last_action_idx = action_idx

        # 4. Decode action → (spread, skew) → (bid, ask)
        spread, skew = self.action_grid[action_idx]
        center = midprice + skew
        half = spread / 2.0
        bid = int(round(center - half))
        ask = int(round(center + half))
        if ask <= bid:
            ask = bid + 1
        return bid, ask

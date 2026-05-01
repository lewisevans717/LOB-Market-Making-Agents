"""Agent C+: tabular Q-learning extension of the contextual-bandit market-maker.

Replaces ``MMCBSETrader``'s incremental-mean Q update with a TD(0) bootstrap so
the agent can perform multi-step credit assignment::

    Q[s,a] <- Q[s,a] + (1/n) * (r + gamma * max_a' Q[s',a'] - Q[s,a])

State discretisation, e-greedy selection, the action grid, freeze mode, and
``save_policy`` / ``load_policy`` / ``finalize`` are inherited unchanged so the
*only* algorithmic difference between Agent C and Agent C+ is the bootstrap
term. This makes the bandit-vs-Q comparison clean.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lob_market_making_agents.agents.mm_c_bse import MMCBSETrader


class MMCPlusBSETrader(MMCBSETrader):
    """Tabular Q-learning market-maker (TD(0) bootstrap, sample-mean step size)."""

    ttype = "MM_C_PLUS"

    _DEFAULT_GAMMA = 0.95

    def __init__(self, tid: str, params: dict[str, Any], time: float = 0.0) -> None:
        super().__init__(tid=tid, params=params, time=time)
        cfg = dict(params or {})
        self.gamma: float = float(cfg.get("gamma", self._DEFAULT_GAMMA))

    # -- override the quote loop to compute s' BEFORE updating Q --------------

    def quote_prices(self, midprice: float, volatility_proxy: float = 0.0) -> tuple[int, int]:
        """Q-learning variant: observe s' first so the TD target can use max_a' Q[s',a']."""
        # Reward earned by the previous step's action.
        current_pnl = self.pnl(midprice)
        reward = (current_pnl - self.prev_pnl) - self.lambda_penalty * (self.inventory ** 2)
        self.prev_pnl = current_pnl

        # Observe the new state s' BEFORE the Q update.
        next_state_idx = self._discretize_state(volatility_proxy)
        self._td_update(reward, next_state_idx)

        # Choose the action for s'.
        action_idx = self._select_action(next_state_idx)
        self.last_state_idx = next_state_idx
        self.last_action_idx = action_idx

        spread, skew = self.action_grid[action_idx]
        center = midprice + skew
        half = spread / 2.0
        bid = int(round(center - half))
        ask = int(round(center + half))
        if ask <= bid:
            ask = bid + 1
        return bid, ask

    def _td_update(self, reward: float, next_state_idx: int) -> None:
        """TD(0) update with sample-mean step size (1/n)."""
        if self.last_state_idx is None or self.freeze:
            return
        s, a = self.last_state_idx, self.last_action_idx
        self.action_counts[s, a] += 1
        n = self.action_counts[s, a]
        target = reward + self.gamma * float(np.max(self.q_table[next_state_idx]))
        self.q_table[s, a] += (target - self.q_table[s, a]) / n

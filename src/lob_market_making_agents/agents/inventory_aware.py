"""Agent B: inventory-aware skew with volatility-adaptive spread."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lob_market_making_agents.agents.base import QuoteDecision
from lob_market_making_agents.env.models import MarketState


@dataclass
class InventoryAwareAgent:
    """Avellaneda-Stoikov-inspired heuristic policy."""

    base_spread: float = 1.0
    inventory_skew_per_unit: float = 0.05
    max_skew: float = 0.75
    vol_window: int = 20
    vol_spread_multiplier: float = 30.0
    min_spread: float = 0.25
    max_spread: float = 4.0
    _midprices: deque[float] = field(default_factory=deque, init=False, repr=False)

    def reset(self) -> None:
        self._midprices.clear()

    def quote(self, state: MarketState) -> QuoteDecision:
        self._midprices.append(float(state.midprice))
        max_points = max(self.vol_window + 1, 2)
        while len(self._midprices) > max_points:
            self._midprices.popleft()

        sigma = self._volatility_proxy()
        spread = self._adaptive_spread(sigma=sigma)

        raw_skew = float(state.inventory) * float(self.inventory_skew_per_unit)
        skew = float(np.clip(raw_skew, -self.max_skew, self.max_skew))
        center = float(state.midprice) - skew

        bid = center - spread / 2.0
        ask = center + spread / 2.0
        if bid >= ask:
            ask = bid + 1e-6

        return QuoteDecision(
            bid=bid,
            ask=ask,
            metadata={
                "spread": spread,
                "skew": skew,
                "volatility_proxy": sigma,
            },
        )

    def _volatility_proxy(self) -> float:
        prices = np.asarray(self._midprices, dtype=float)
        if prices.size < 3:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        if returns.size == 0:
            return 0.0
        return float(np.std(returns))

    def _adaptive_spread(self, sigma: float) -> float:
        raw = float(self.base_spread) * (1.0 + float(self.vol_spread_multiplier) * sigma)
        return float(np.clip(raw, self.min_spread, self.max_spread))


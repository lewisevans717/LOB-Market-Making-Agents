"""Agent A: fixed-spread market maker."""

from __future__ import annotations

from dataclasses import dataclass

from lob_market_making_agents.agents.base import QuoteDecision
from lob_market_making_agents.env.models import MarketState


@dataclass
class FixedSpreadAgent:
    """Quotes a constant spread around the current midprice."""

    spread: float = 1.0

    def reset(self) -> None:
        return None

    def quote(self, state: MarketState) -> QuoteDecision:
        spread = max(float(self.spread), 1e-6)
        bid = state.midprice - spread / 2.0
        ask = state.midprice + spread / 2.0
        return QuoteDecision(bid=bid, ask=ask, metadata={"spread": spread, "skew": 0.0})


"""Shared interfaces for market-making agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lob_market_making_agents.env.models import MarketState


@dataclass(frozen=True)
class QuoteDecision:
    """Agent quote output with optional diagnostics."""

    bid: float
    ask: float
    metadata: dict[str, Any]


class MarketMakingAgent(Protocol):
    """Minimal quote interface shared by all agents."""

    def reset(self) -> None:
        """Reset any episode-local state."""

    def quote(self, state: MarketState) -> QuoteDecision:
        """Return a quote for the current state."""


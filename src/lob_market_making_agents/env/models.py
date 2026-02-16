"""Typed models for the simulation wrapper and run logs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegimeConfig:
    """Top-level regime controls used by the simulator."""

    volatility: str
    toxicity: float
    competition: str

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RegimeConfig":
        return cls(
            volatility=str(data.get("volatility", "low")),
            toxicity=float(data.get("toxicity", 0.0)),
            competition=str(data.get("competition", "solo")),
        )


@dataclass(frozen=True)
class AgentQuote:
    """Single market-making quote."""

    bid: float
    ask: float
    size: float = 1.0


@dataclass(frozen=True)
class MarketState:
    """State exposed to policy code."""

    timestamp: int
    midprice: float
    inventory: float
    cash: float
    pnl: float


@dataclass(frozen=True)
class EventRecord:
    """Per-step log row."""

    timestamp: int
    midprice: float
    bid: float
    ask: float
    fill_side: str
    fill_qty: float
    fill_price: float | None
    inventory: float
    cash: float
    pnl: float


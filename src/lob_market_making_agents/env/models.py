"""Shared typed models for BSE-first runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegimeConfig:
    """Top-level market regime controls exposed by experiment config."""

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
class SessionConfig:
    """BSE session controls."""

    episode_steps: int = 500
    base_midprice: float = 100.0
    size: float = 1.0
    external_order_probability: float = 0.7
    vol_window: int = 20


@dataclass(frozen=True)
class PopulationConfig:
    """Participant population controls."""

    participant_count: int = 8
    mm_trader_id: str = "MM0"
    counterparty_prefix: str = "CP"
    toxic_activation_prob: float = 1.0
    toxic_aggression_ticks: int = 3

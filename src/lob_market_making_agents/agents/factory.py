"""Factory for constructing configured agents."""

from __future__ import annotations

from typing import Any

from lob_market_making_agents.agents.base import MarketMakingAgent
from lob_market_making_agents.agents.fixed_spread import FixedSpreadAgent
from lob_market_making_agents.agents.inventory_aware import InventoryAwareAgent


def create_agent(agent_name: str, params: dict[str, Any] | None = None) -> MarketMakingAgent:
    """Create an agent by symbolic name."""
    key = agent_name.strip().upper()
    cfg = dict(params or {})

    if key == "A":
        spread = float(cfg.get("spread", 1.0))
        return FixedSpreadAgent(spread=spread)

    if key == "B":
        return InventoryAwareAgent(
            base_spread=float(cfg.get("base_spread", 1.0)),
            inventory_skew_per_unit=float(cfg.get("inventory_skew_per_unit", 0.05)),
            max_skew=float(cfg.get("max_skew", 0.75)),
            vol_window=int(cfg.get("vol_window", 20)),
            vol_spread_multiplier=float(cfg.get("vol_spread_multiplier", 30.0)),
            min_spread=float(cfg.get("min_spread", 0.25)),
            max_spread=float(cfg.get("max_spread", 4.0)),
        )

    raise ValueError(f"Unsupported agent '{agent_name}'. Implemented agents: A, B.")


"""Factories for native BSE market-maker trader classes."""

from __future__ import annotations

from typing import Any

from lob_market_making_agents.agents.mm_a_bse import MMABSETrader
from lob_market_making_agents.agents.mm_b_bse import MMBBSETrader
from lob_market_making_agents.agents.mm_c_bse import MMCBSETrader


def create_mm_trader(agent_name: str, tid: str, params: dict[str, Any] | None = None, time: float = 0.0) -> MMABSETrader:
    """Create the requested MM trader implementation."""
    key = agent_name.strip().upper()
    cfg = dict(params or {})
    if key == "A":
        return MMABSETrader(tid=tid, params=cfg, time=time)
    if key == "B":
        return MMBBSETrader(tid=tid, params=cfg, time=time)
    if key == "C":
        return MMCBSETrader(tid=tid, params=cfg, time=time)
    raise ValueError(f"Unsupported agent '{agent_name}'. Implemented agents: A, B, C.")

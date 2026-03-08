"""Custom toxic/informed participant for BSE sessions."""

from __future__ import annotations

from typing import Any

from lob_market_making_agents.env.bse_loader import load_bse_module


BSE = load_bse_module()
TraderBase = BSE.Trader
Order = BSE.Order
SYS_MIN = int(getattr(BSE, "bse_sys_minprice", 1))
SYS_MAX = int(getattr(BSE, "bse_sys_maxprice", 500))


class ToxicBSETrader(TraderBase):
    """Toxic flow trader that tends to submit marketable directional orders."""

    def __init__(self, tid: str, params: dict[str, Any] | None = None, time: float = 0.0) -> None:
        super().__init__(ttype="TOXIC", tid=tid, balance=0.0, params=params or {}, time=time)
        cfg = dict(params or {})
        self.activation_prob = float(cfg.get("activation_prob", 1.0))
        self.aggression_ticks = int(cfg.get("aggression_ticks", 2))

    def getorder(self, time: float, countdown: float, lob: dict[str, Any]) -> Any | None:
        """Unused by the session runner; kept for BSE trader compatibility."""
        if countdown < 0:
            raise ValueError("countdown cannot be negative")
        if lob is None:
            return None
        return None

    def make_aggressive_order(self, *, time: float, qid: int, side: str, reference_mid: float) -> Any:
        """Create an aggressive order to hit/lift existing top-of-book quotes."""
        mid = int(round(reference_mid))
        if side == "buy":
            price = min(SYS_MAX, mid + self.aggression_ticks + 2)
            return Order(self.tid, "Bid", price, 1, time, qid)
        price = max(SYS_MIN, mid - self.aggression_ticks - 2)
        return Order(self.tid, "Ask", price, 1, time, qid)

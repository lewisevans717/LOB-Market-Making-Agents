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
    """Aggressive liquidity-taking participant used to model "toxic" order flow.

    "Toxic" here means *aggressive*, not *informed*: this trader has no oracle
    access to future midprices. When activated by the session runner it submits
    marketable orders that lift the ask or hit the bid via
    ``make_aggressive_order``, anchored to a reference midprice rather than to
    any forward-looking signal. The adverse-selection pressure it imposes on
    the market maker arises from order-flow imbalance and price impact, not
    from directional knowledge of where the price is heading. This is a
    deliberate simplification of classical informed-trader models and is
    documented as such in the report's Limitations section.
    """

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

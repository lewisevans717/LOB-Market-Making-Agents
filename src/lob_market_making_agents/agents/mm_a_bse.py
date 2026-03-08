"""Native BSE market-maker agent A (fixed spread)."""

from __future__ import annotations

from typing import Any

from lob_market_making_agents.env.bse_loader import load_bse_module


BSE = load_bse_module()
TraderBase = BSE.Trader
Order = BSE.Order


class MMABSETrader(TraderBase):
    """Fixed-spread market-maker implemented as a BSE trader class."""

    def __init__(self, tid: str, params: dict[str, Any] | None = None, time: float = 0.0) -> None:
        super().__init__(ttype="MM_A", tid=tid, balance=0.0, params=params or {}, time=time)
        cfg = dict(params or {})
        self.spread = float(cfg.get("spread", 2.0))

        self.inventory = 0.0
        self.cash = 0.0

        self.last_bid_order: Any | None = None
        self.last_ask_order: Any | None = None

    def getorder(self, time: float, countdown: float, lob: dict[str, Any]) -> Any | None:
        """Unused by the session runner; kept for BSE trader compatibility."""
        if countdown < 0:
            raise ValueError("countdown cannot be negative")
        if lob is None:
            return None
        return None

    def quote_prices(self, midprice: float, volatility_proxy: float = 0.0) -> tuple[int, int]:
        """Return fixed bid/ask around midprice."""
        if volatility_proxy < 0:
            raise ValueError("volatility_proxy must be non-negative")
        half = self.spread / 2.0
        bid = int(round(midprice - half))
        ask = int(round(midprice + half))
        if ask <= bid:
            ask = bid + 1
        return bid, ask

    def make_quotes(
        self,
        *,
        time: float,
        qid: int,
        midprice: float,
        volatility_proxy: float,
    ) -> tuple[Any, Any]:
        """Create one bid and one ask BSE order for this trader."""
        bid_px, ask_px = self.quote_prices(midprice=midprice, volatility_proxy=volatility_proxy)
        bid_order = Order(self.tid, "Bid", bid_px, 1, time, qid)
        ask_order = Order(self.tid, "Ask", ask_px, 1, time, qid)
        self.last_bid_order = bid_order
        self.last_ask_order = ask_order
        return bid_order, ask_order

    def on_fill(self, fill_side: str, fill_price: float, fill_qty: float) -> None:
        """Apply inventory/cash updates for a fill against this MM."""
        if fill_side == "bid":
            self.inventory += fill_qty
            self.cash -= fill_qty * float(fill_price)
        elif fill_side == "ask":
            self.inventory -= fill_qty
            self.cash += fill_qty * float(fill_price)

    def pnl(self, midprice: float) -> float:
        """Mark-to-market PnL."""
        return self.cash + (self.inventory * float(midprice))

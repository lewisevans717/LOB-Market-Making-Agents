"""Native BSE market-maker agent B (inventory-aware + volatility-adaptive)."""

from __future__ import annotations

from typing import Any

from lob_market_making_agents.agents.mm_a_bse import MMABSETrader


class MMBBSETrader(MMABSETrader):
    """Inventory-aware MM that skews center and adapts spread by volatility proxy."""

    def __init__(self, tid: str, params: dict[str, Any] | None = None, time: float = 0.0) -> None:
        super().__init__(tid=tid, params=params or {}, time=time)
        cfg = dict(params or {})

        self.base_spread = float(cfg.get("base_spread", cfg.get("spread", 2.0)))
        self.inventory_skew_per_unit = float(cfg.get("inventory_skew_per_unit", 0.08))
        self.max_skew = float(cfg.get("max_skew", 1.5))
        self.vol_spread_multiplier = float(cfg.get("vol_spread_multiplier", 20.0))
        self.min_spread = float(cfg.get("min_spread", 1.0))
        self.max_spread = float(cfg.get("max_spread", 8.0))

    def quote_prices(self, midprice: float, volatility_proxy: float = 0.0) -> tuple[int, int]:
        # WHY: spread widens linearly with the volatility proxy (Avellaneda-Stoikov 2008
        # gives an exponential reservation-price model; we use a linear proxy because BSE
        # ticks are discrete and the closed-form γ·σ² term has no observable counterpart
        # without an analytic price process).
        dynamic_spread = self.base_spread + (self.vol_spread_multiplier * max(volatility_proxy, 0.0))
        dynamic_spread = min(max(dynamic_spread, self.min_spread), self.max_spread)

        # WHY: inventory skew leans the quote *against* the current position so fills
        # reduce |q|. Sign is negative because long inventory (q>0) should drag the
        # midprice down, encouraging asks and discouraging bids.
        raw_skew = -self.inventory * self.inventory_skew_per_unit
        skew = min(max(raw_skew, -self.max_skew), self.max_skew)
        center = float(midprice) + skew

        half = dynamic_spread / 2.0
        bid = int(round(center - half))
        ask = int(round(center + half))
        if ask <= bid:
            ask = bid + 1
        return bid, ask

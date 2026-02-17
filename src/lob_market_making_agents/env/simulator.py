"""Deterministic lightweight LOB simulator wrapper for M1 development."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from lob_market_making_agents.env.models import AgentQuote, EventRecord, MarketState, RegimeConfig


VOLATILITY_SCALE = {
    "low": 0.05,
    "medium": 0.12,
    "high": 0.25,
}

COMPETITION_PENALTY = {
    "solo": 0.0,
    "with_competitor": 0.08,
}


class SimulatedLOBEnv:
    """Small adapter-style environment with reset/step semantics."""

    def __init__(self, base_midprice: float = 100.0, episode_steps: int = 200) -> None:
        self.base_midprice = float(base_midprice)
        self.episode_steps = int(episode_steps)
        self._rng: np.random.Generator | None = None
        self._regime = RegimeConfig(volatility="low", toxicity=0.0, competition="solo")
        self._state = MarketState(timestamp=0, midprice=self.base_midprice, inventory=0.0, cash=0.0, pnl=0.0)
        self._done = False

    def reset(self, seed: int, regime_config: RegimeConfig) -> MarketState:
        """Initialize episode state."""
        self._rng = np.random.default_rng(seed)
        self._regime = regime_config
        self._state = MarketState(timestamp=0, midprice=self.base_midprice, inventory=0.0, cash=0.0, pnl=0.0)
        self._done = False
        return self._state

    def step(self, quote: AgentQuote) -> tuple[MarketState, EventRecord, bool]:
        """Advance one step from a quote decision."""
        if self._done:
            raise RuntimeError("Episode already completed; call reset() before stepping again.")
        if self._rng is None:
            raise RuntimeError("Environment not initialized; call reset() before step().")
        if quote.bid >= quote.ask:
            raise ValueError(f"Invalid quote: bid={quote.bid} must be less than ask={quote.ask}.")

        fill_side = "none"
        fill_qty = 0.0
        fill_price: float | None = None

        toxicity = min(max(self._regime.toxicity / 100.0, 0.0), 1.0)
        informed = bool(self._rng.random() < toxicity)

        if self._rng.random() < self._fill_probability():
            fill_side = self._sample_fill_side(informed=informed)
            fill_qty = quote.size
            fill_price = quote.ask if fill_side == "ask" else quote.bid

        inventory = self._state.inventory
        cash = self._state.cash
        if fill_side != "none":
            if fill_price is None:
                raise RuntimeError("Missing fill_price for fill.")
            if fill_side == "bid":
                inventory += fill_qty
                cash -= fill_qty * fill_price
            elif fill_side == "ask":
                inventory -= fill_qty
                cash += fill_qty * fill_price

        midprice = self._next_midprice(fill_side=fill_side, informed=informed)
        pnl = cash + inventory * midprice
        timestamp = self._state.timestamp + 1
        self._done = timestamp >= self.episode_steps

        self._state = MarketState(
            timestamp=timestamp,
            midprice=midprice,
            inventory=inventory,
            cash=cash,
            pnl=pnl,
        )
        event = EventRecord(
            timestamp=timestamp,
            midprice=midprice,
            bid=quote.bid,
            ask=quote.ask,
            fill_side=fill_side,
            fill_qty=fill_qty,
            fill_price=fill_price,
            inventory=inventory,
            cash=cash,
            pnl=pnl,
        )
        return self._state, event, self._done

    def run_episode(
        self,
        seed: int,
        regime_config: RegimeConfig,
        spread: float = 1.0,
        size: float = 1.0,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        """Run a full episode using a fixed-spread quote policy."""
        state = self.reset(seed=seed, regime_config=regime_config)
        events: list[dict[str, Any]] = []
        while True:
            quote = AgentQuote(
                bid=state.midprice - spread / 2.0,
                ask=state.midprice + spread / 2.0,
                size=size,
            )
            state, event, done = self.step(quote)
            events.append(asdict(event))
            if done:
                break
        summary = self._episode_summary(events)
        return events, summary

    def _fill_probability(self) -> float:
        competition_penalty = COMPETITION_PENALTY.get(self._regime.competition, 0.0)
        base = 0.28 - competition_penalty
        return float(min(max(base, 0.05), 0.9))

    def _sample_fill_side(self, informed: bool) -> str:
        if self._rng is None:
            raise RuntimeError("RNG unavailable.")
        if not informed:
            return "bid" if self._rng.random() < 0.5 else "ask"
        return "ask" if self._rng.random() < 0.5 else "bid"

    def _next_midprice(self, fill_side: str, informed: bool) -> float:
        if self._rng is None:
            raise RuntimeError("RNG unavailable.")
        sigma = VOLATILITY_SCALE.get(self._regime.volatility, VOLATILITY_SCALE["low"])
        noise = self._rng.normal(loc=0.0, scale=sigma)
        adverse = 0.0
        if informed and fill_side == "ask":
            adverse = sigma * 0.8
        elif informed and fill_side == "bid":
            adverse = -sigma * 0.8
        new_mid = self._state.midprice + noise + adverse
        return float(max(new_mid, 0.01))

    @staticmethod
    def _episode_summary(events: list[dict[str, Any]]) -> dict[str, float]:
        if not events:
            return {
                "steps": 0.0,
                "final_midprice": 0.0,
                "final_inventory": 0.0,
                "final_cash": 0.0,
                "final_pnl": 0.0,
                "fills": 0.0,
            }
        last = events[-1]
        fills = sum(1.0 for event in events if event["fill_side"] != "none")
        return {
            "steps": float(len(events)),
            "final_midprice": float(last["midprice"]),
            "final_inventory": float(last["inventory"]),
            "final_cash": float(last["cash"]),
            "final_pnl": float(last["pnl"]),
            "fills": float(fills),
        }

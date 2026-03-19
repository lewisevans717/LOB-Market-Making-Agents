"""Regime-to-population mapping and external participant order generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from lob_market_making_agents.agents import ToxicBSETrader, create_mm_trader
from lob_market_making_agents.env.bse_loader import load_bse_module
from lob_market_making_agents.env.models import PopulationConfig, RegimeConfig


BSE = load_bse_module()
Order = BSE.Order
SYS_MIN = int(getattr(BSE, "bse_sys_minprice", 1))
SYS_MAX = int(getattr(BSE, "bse_sys_maxprice", 500))


@dataclass
class ExternalParticipant:
    """One non-MM participant in the market session."""

    tid: str
    kind: str
    profile: str
    trader: Any


@dataclass(frozen=True)
class PopulationBundle:
    """Participants generated for one run."""

    competitor_mms: tuple[Any, ...]
    external_participants: tuple[ExternalParticipant, ...]


NOISE_PROFILES = ("GVWY", "ZIC", "SHVR")


def build_population(
    *,
    regime: RegimeConfig,
    population_config: PopulationConfig,
    competitor_agent: str = "A",
    competitor_params: dict[str, Any] | None = None,
) -> PopulationBundle:
    """Create competitor MM traders and external participant pool."""
    competitor_mms: list[Any] = []
    if regime.competition == "with_competitor":
        competitor_tid = f"{population_config.mm_trader_id}_C0"
        competitor_mms.append(
            create_mm_trader(
                agent_name=competitor_agent,
                tid=competitor_tid,
                params=competitor_params or {"spread": 2.0},
                time=0.0,
            )
        )

    participant_count = max(1, int(population_config.participant_count))
    toxic_ratio = min(max(float(regime.toxicity), 0.0), 100.0) / 100.0
    toxic_count = int(round(participant_count * toxic_ratio))
    toxic_count = min(max(toxic_count, 0), participant_count)
    noise_count = participant_count - toxic_count

    external_participants: list[ExternalParticipant] = []
    prefix = population_config.counterparty_prefix
    for idx in range(noise_count):
        tid = f"{prefix}N{idx}"
        profile = NOISE_PROFILES[idx % len(NOISE_PROFILES)]
        if profile == "GVWY":
            trader = BSE.TraderGiveaway("GVWY", tid, 0.0, {}, 0.0)
        elif profile == "ZIC":
            trader = BSE.TraderZIC("ZIC", tid, 0.0, {}, 0.0)
        else:
            trader = BSE.TraderShaver("SHVR", tid, 0.0, {}, 0.0)
        external_participants.append(
            ExternalParticipant(tid=tid, kind="noise", profile=profile, trader=trader)
        )

    for idx in range(toxic_count):
        tid = f"{prefix}T{idx}"
        trader = ToxicBSETrader(
            tid=tid,
            params={
                "activation_prob": population_config.toxic_activation_prob,
                "aggression_ticks": population_config.toxic_aggression_ticks,
            },
            time=0.0,
        )
        external_participants.append(
            ExternalParticipant(tid=tid, kind="toxic", profile="TOXIC", trader=trader)
        )

    return PopulationBundle(
        competitor_mms=tuple(competitor_mms),
        external_participants=tuple(external_participants),
    )


def build_external_order(
    *,
    participant: ExternalParticipant,
    time: float,
    qid: int,
    lob: dict[str, Any],
    reference_mid: float,
    rng: np.random.Generator,
) -> tuple[Any | None, str | None]:
    """Create one external order and return order + flow side (buy/sell)."""
    if participant.kind == "toxic":
        if rng.random() > float(getattr(participant.trader, "activation_prob", 1.0)):
            return None, None
        side = "buy" if rng.random() < 0.5 else "sell"
        order = participant.trader.make_aggressive_order(
            time=time,
            qid=qid,
            side=side,
            reference_mid=reference_mid,
        )
        return order, side

    side = "buy" if rng.random() < 0.5 else "sell"
    mid = int(round(reference_mid))

    if side == "buy":
        limit = min(SYS_MAX - 1, mid + int(rng.integers(2, 8)))
        if limit < SYS_MIN:
            return None, None
        assignment = Order(participant.tid, "Bid", limit, 1, time, qid)
    else:
        limit = max(SYS_MIN + 1, mid - int(rng.integers(2, 8)))
        if limit > SYS_MAX:
            return None, None
        assignment = Order(participant.tid, "Ask", limit, 1, time, qid)

    participant.trader.orders = [assignment]
    order = participant.trader.getorder(time, 0.0, lob)

    if order is None:
        return None, None

    # Design choice: noise orders are forced to be marketable (crossing the best
    # counterparty price) so that external flow reliably generates fills for the MM.
    # This inflates fill rate compared to a passive limit-order market, but is
    # intentional — it ensures enough data for metrics and Agent C learning.
    best_bid = lob["bids"]["best"]
    best_ask = lob["asks"]["best"]
    if side == "buy" and best_ask is not None:
        order.price = max(int(order.price), int(best_ask))
    elif side == "sell" and best_bid is not None:
        order.price = min(int(order.price), int(best_bid))

    return order, side

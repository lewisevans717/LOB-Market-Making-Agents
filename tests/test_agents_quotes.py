from lob_market_making_agents.agents.fixed_spread import FixedSpreadAgent
from lob_market_making_agents.agents.inventory_aware import InventoryAwareAgent
from lob_market_making_agents.env.models import MarketState


def _state(*, midprice: float = 100.0, inventory: float = 0.0, t: int = 0) -> MarketState:
    return MarketState(
        timestamp=t,
        midprice=midprice,
        inventory=inventory,
        cash=0.0,
        pnl=0.0,
    )


def test_fixed_spread_quote_math() -> None:
    agent = FixedSpreadAgent(spread=2.0)
    decision = agent.quote(_state(midprice=100.0))
    assert decision.bid == 99.0
    assert decision.ask == 101.0
    assert decision.metadata["spread"] == 2.0


def test_inventory_skew_direction() -> None:
    agent = InventoryAwareAgent(
        base_spread=1.0,
        inventory_skew_per_unit=0.2,
        max_skew=10.0,
        vol_spread_multiplier=0.0,
    )
    flat = agent.quote(_state(midprice=100.0, inventory=0.0, t=1))
    long_inv = agent.quote(_state(midprice=100.0, inventory=3.0, t=2))
    short_inv = agent.quote(_state(midprice=100.0, inventory=-3.0, t=3))

    assert long_inv.bid < flat.bid
    assert long_inv.ask < flat.ask
    assert short_inv.bid > flat.bid
    assert short_inv.ask > flat.ask


def test_inventory_agent_spread_widens_with_volatility() -> None:
    agent = InventoryAwareAgent(
        base_spread=1.0,
        vol_window=10,
        vol_spread_multiplier=200.0,
        min_spread=0.25,
        max_spread=4.0,
    )
    # Warm up with stable prices (near-zero volatility proxy).
    for idx in range(1, 6):
        decision = agent.quote(_state(midprice=100.0, t=idx))
    calm_spread = float(decision.metadata["spread"])

    for idx, mid in enumerate([100.0, 101.2, 99.1, 101.8, 98.7], start=6):
        decision = agent.quote(_state(midprice=mid, t=idx))
    volatile_spread = float(decision.metadata["spread"])

    assert volatile_spread > calm_spread


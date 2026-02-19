from lob_market_making_agents.metrics.pipeline import compute_run_metrics
from lob_market_making_agents.metrics.schema import MetricsSettings, RunContext


def _context() -> RunContext:
    return RunContext(
        run_id="run_x",
        config_name="cfg",
        agent="A",
        volatility="low",
        toxicity=0.0,
        competition="solo",
        seed=7,
    )


def test_markout_sign_bid_and_ask() -> None:
    events = [
        {
            "timestamp": 1,
            "midprice": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "fill_side": "bid",
            "fill_qty": 1.0,
            "fill_price": 100.0,
            "inventory": 1.0,
            "cash": -100.0,
            "pnl": 0.0,
        },
        {
            "timestamp": 2,
            "midprice": 101.0,
            "bid": 100.5,
            "ask": 101.5,
            "fill_side": "ask",
            "fill_qty": 1.0,
            "fill_price": 102.0,
            "inventory": 0.0,
            "cash": 2.0,
            "pnl": 2.0,
        },
        {
            "timestamp": 3,
            "midprice": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "fill_side": "none",
            "fill_qty": 0.0,
            "fill_price": None,
            "inventory": 0.0,
            "cash": 2.0,
            "pnl": 2.0,
        },
    ]
    summary = {
        "final_midprice": 100.0,
        "final_inventory": 0.0,
        "final_cash": 2.0,
        "final_pnl": 2.0,
    }
    settings = MetricsSettings(markout_horizon=1, inventory_threshold=3.0)
    row = compute_run_metrics(events=events, summary=summary, context=_context(), settings=settings)

    # markouts: bid -> 101-100=1, ask -> 102-100=2
    assert row["markout_count_h1"] == 2.0
    assert row["markout_mean_h1"] == 1.5


def test_markout_horizon_clipping() -> None:
    events = [
        {
            "timestamp": 1,
            "midprice": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "fill_side": "none",
            "fill_qty": 0.0,
            "fill_price": None,
            "inventory": 0.0,
            "cash": 0.0,
            "pnl": 0.0,
        },
        {
            "timestamp": 2,
            "midprice": 100.1,
            "bid": 99.6,
            "ask": 100.6,
            "fill_side": "ask",
            "fill_qty": 1.0,
            "fill_price": 100.6,
            "inventory": -1.0,
            "cash": 100.6,
            "pnl": 0.5,
        },
    ]
    summary = {"final_midprice": 100.1, "final_inventory": -1.0, "final_cash": 100.6, "final_pnl": 0.5}
    settings = MetricsSettings(markout_horizon=5, inventory_threshold=3.0)
    row = compute_run_metrics(events=events, summary=summary, context=_context(), settings=settings)

    assert row["markout_count_h5"] == 0.0
    assert row["markout_mean_h5"] == 0.0
    assert row["markout_std_h5"] == 0.0
    assert row["markout_p5_h5"] == 0.0


def test_inventory_exposure_fraction_and_fill_rate() -> None:
    events = [
        {
            "timestamp": 1,
            "midprice": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "fill_side": "none",
            "fill_qty": 0.0,
            "fill_price": None,
            "inventory": 0.0,
            "cash": 0.0,
            "pnl": 0.0,
        },
        {
            "timestamp": 2,
            "midprice": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "fill_side": "bid",
            "fill_qty": 1.0,
            "fill_price": 99.5,
            "inventory": 4.0,
            "cash": -398.0,
            "pnl": 2.0,
        },
        {
            "timestamp": 3,
            "midprice": 99.9,
            "bid": 99.4,
            "ask": 100.4,
            "fill_side": "ask",
            "fill_qty": 1.0,
            "fill_price": 100.4,
            "inventory": -5.0,
            "cash": 104.0,
            "pnl": -395.5,
        },
        {
            "timestamp": 4,
            "midprice": 100.2,
            "bid": 99.7,
            "ask": 100.7,
            "fill_side": "none",
            "fill_qty": 0.0,
            "fill_price": None,
            "inventory": 2.0,
            "cash": 104.0,
            "pnl": 304.4,
        },
    ]
    summary = {"final_midprice": 100.2, "final_inventory": 2.0, "final_cash": 104.0, "final_pnl": 304.4}
    settings = MetricsSettings(markout_horizon=1, inventory_threshold=3.0)
    row = compute_run_metrics(events=events, summary=summary, context=_context(), settings=settings)

    assert row["fill_rate"] == 0.5
    # |inventory| > 3 for 2 of 4 rows (4 and -5).
    assert row["inventory_exposure_frac_abs_gt_threshold"] == 0.5


import numpy as np
import pandas as pd

from testing.engine.execution_config import ExecutionConfig
from testing.engine.leg import Leg
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import (
    PositionSimulator,
    check_exit,
    entry_value_by_right,
    mark_to_market,
    net_credit,
)


def _leg(optionid, strike, delta, bid, offer, right, side, exdate="2020-02-15"):
    return Leg(
        optionid=optionid, exdate=pd.Timestamp(exdate), strike_price=strike, delta=delta,
        best_bid=bid, best_offer=offer, right=right, side=side,
    )


def _trade(entry_date="2020-01-15", exdate="2020-02-15"):
    short_put = _leg("short", 95, -0.20, bid=3.0, offer=3.2, right="P", side="short", exdate=exdate)
    long_put = _leg("long", 90, -0.10, bid=1.0, offer=1.2, right="P", side="long", exdate=exdate)
    return PositionSpec(
        entry_date=pd.Timestamp(entry_date),
        legs=[short_put, long_put],
    )


def _leg_row(date, optionid, bid, offer):
    return {"date": pd.Timestamp(date), "optionid": optionid, "best_bid": bid, "best_offer": offer}


def _config(**overrides):
    defaults = dict(profit_target_pct=0.50, stop_loss_multiple=2.0, exit_dte=21)
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def test_entry_credit_uses_short_bid_minus_long_offer():
    trade = _trade()
    # entry_credit = 3.0 - 1.2 = 1.8
    history = pd.DataFrame([
        _leg_row("2020-01-16", "short", bid=2.9, offer=3.0),
        _leg_row("2020-01-16", "long", bid=1.0, offer=1.1),
    ])
    result = PositionSimulator().simulate(trade, history, _config())
    assert result.entry_credit == 1.8


def test_exits_at_profit_target_using_offer_minus_bid_to_close():
    trade = _trade()
    # entry_credit = 1.8; profit target = 50% -> need profit_so_far >= 0.9
    # cost_to_close = short.best_offer - long.best_bid
    history = pd.DataFrame([
        # day 1: cost_to_close = 1.0 - 0.2 = 0.8 -> profit 1.0, target met
        _leg_row("2020-01-20", "short", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long", bid=0.2, offer=0.3),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "profit_target"
    assert result.exit_date == pd.Timestamp("2020-01-20")
    assert result.exit_debit == 0.8
    assert result.pnl == 1.0


def test_exits_at_hard_stop_when_loss_exceeds_multiple():
    trade = _trade()
    # entry_credit = 1.8; stop multiple 2.0 -> stop when loss >= 3.6, i.e. cost_to_close >= 5.4
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=5.0, offer=5.5),  # cost_to_close = 5.5 - 0.1 = 5.4
        _leg_row("2020-01-20", "long", bid=0.1, offer=0.2),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "hard_stop"
    assert result.exit_debit == 5.4
    assert result.pnl == -(5.4 - 1.8)


def test_max_loss_basis_does_not_false_stop_where_entry_credit_basis_would():
    # Same trade/history as test_exits_at_hard_stop_when_loss_exceeds_multiple:
    # loss=3.6 exactly clears the entry_credit-basis threshold (2.0 * 1.8=3.6)
    # but not the max_loss-basis one (2.0 * max_loss_at_expiration(legs, 1.8)=3.2 -> 6.4).
    trade = _trade()
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=5.0, offer=5.5),
        _leg_row("2020-01-20", "long", bid=0.1, offer=0.2),
    ])
    result = PositionSimulator().simulate(trade, history, _config(exit_trigger_basis="max_loss"))

    assert result.exit_reason != "hard_stop"


def test_max_loss_basis_fires_hard_stop_once_loss_exceeds_max_loss_multiple():
    trade = _trade()
    # max_loss_at_expiration(legs, 1.8) = 3.2 -> stop fires at loss >= 6.4,
    # i.e. cost_to_close >= 8.2
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=8.0, offer=8.5),
        _leg_row("2020-01-20", "long", bid=0.1, offer=0.2),  # cost_to_close = 8.5 - 0.1 = 8.4
    ])
    result = PositionSimulator().simulate(trade, history, _config(exit_trigger_basis="max_loss"))

    assert result.exit_reason == "hard_stop"


def test_max_loss_basis_does_not_false_profit_target_where_entry_credit_basis_would():
    # profit_so_far=1.0 clears the entry_credit-basis target (0.5 * 1.8=0.9)
    # but not the max_loss-basis one (0.5 * 3.2=1.6).
    trade = _trade()
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=0.7, offer=0.8),
        _leg_row("2020-01-20", "long", bid=0.0, offer=0.1),  # cost_to_close = 0.8 - 0.0 = 0.8, profit 1.0
    ])
    result = PositionSimulator().simulate(trade, history, _config(exit_trigger_basis="max_loss"))

    assert result.exit_reason != "profit_target"


def test_default_exit_trigger_basis_is_entry_credit_unchanged():
    assert _config().exit_trigger_basis == "entry_credit"


def test_exits_at_dte_stop_when_neither_target_nor_stop_fires():
    trade = _trade(entry_date="2020-01-15", exdate="2020-02-15")
    # 21 DTE from 2020-02-15 is 2020-01-25
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=2.95, offer=3.05),  # tiny move, no trigger, dte=26
        _leg_row("2020-01-20", "long", bid=0.95, offer=1.05),
        _leg_row("2020-01-25", "short", bid=2.9, offer=3.0),  # dte=21 -> stop
        _leg_row("2020-01-25", "long", bid=0.9, offer=1.0),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "dte_stop"
    assert result.exit_date == pd.Timestamp("2020-01-25")


def test_profit_target_takes_precedence_over_dte_stop_on_same_day():
    trade = _trade(entry_date="2020-01-15", exdate="2020-02-15")
    # On 2020-01-25 (dte=21, would trigger dte_stop) the position has ALSO already hit profit target.
    history = pd.DataFrame([
        _leg_row("2020-01-25", "short", bid=0.9, offer=1.0),  # cost_to_close = 1.0 - 0.2 = 0.8, profit 1.0 >= 0.9
        _leg_row("2020-01-25", "long", bid=0.2, offer=0.3),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "profit_target"


def test_data_exhausted_fallback_when_no_trigger_fires():
    trade = _trade(entry_date="2020-01-15", exdate="2020-02-15")
    history = pd.DataFrame([
        # small move, no trigger, far from dte stop
        _leg_row("2020-01-16", "short", bid=2.95, offer=3.05),
        _leg_row("2020-01-16", "long", bid=0.95, offer=1.05),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "data_exhausted"
    assert result.exit_date == pd.Timestamp("2020-01-16")


def _iron_condor_trade(entry_date="2020-01-15", exdate="2020-02-15"):
    short_put = _leg("short_put", 95, -0.20, bid=3.0, offer=3.2, right="P", side="short", exdate=exdate)
    long_put = _leg("long_put", 90, -0.10, bid=1.0, offer=1.2, right="P", side="long", exdate=exdate)
    short_call = _leg("short_call", 105, 0.30, bid=2.0, offer=2.2, right="C", side="short", exdate=exdate)
    long_call = _leg("long_call", 110, 0.20, bid=0.8, offer=1.0, right="C", side="long", exdate=exdate)
    return PositionSpec(
        entry_date=pd.Timestamp(entry_date),
        legs=[short_put, long_put, short_call, long_call],
    )


def test_iron_condor_entry_credit_combines_both_spreads():
    trade = _iron_condor_trade()
    # put credit: 3.0 - 1.2 = 1.8; call credit: 2.0 - 1.0 = 1.0; combined = 2.8
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short_put", bid=2.9, offer=3.0),
        _leg_row("2020-01-20", "long_put", bid=1.0, offer=1.1),
        _leg_row("2020-01-20", "short_call", bid=1.9, offer=2.0),
        _leg_row("2020-01-20", "long_call", bid=0.7, offer=0.9),
    ])
    result = PositionSimulator().simulate(trade, history, _config())
    assert result.entry_credit == 2.8


def test_iron_condor_exit_uses_combined_pnl_with_put_call_breakdown():
    trade = _iron_condor_trade()
    # entry_credit = 2.8 (put 1.8 + call 1.0); profit target 50% -> need profit_so_far >= 1.4
    # put cost_to_close = 1.0 - 0.2 = 0.8 -> put_pnl = 1.0
    # call cost_to_close = 0.5 - 0.1 = 0.4 -> call_pnl = 0.6
    # combined cost_to_close = 1.2 -> profit_so_far = 1.6 >= 1.4 -> profit_target
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short_put", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long_put", bid=0.2, offer=0.3),
        _leg_row("2020-01-20", "short_call", bid=0.4, offer=0.5),
        _leg_row("2020-01-20", "long_call", bid=0.1, offer=0.2),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "profit_target"
    assert np.isclose(result.put_pnl, 1.0)
    assert np.isclose(result.call_pnl, 0.6)
    assert np.isclose(result.pnl, 1.6)


def test_iron_condor_requires_all_four_legs_present_on_a_check_date():
    trade = _iron_condor_trade()
    history = pd.DataFrame([
        # 2020-01-20 is missing the call legs entirely -> not a valid check date
        _leg_row("2020-01-20", "short_put", bid=2.9, offer=3.0),
        _leg_row("2020-01-20", "long_put", bid=1.0, offer=1.1),
        # 2020-01-25 has all four legs
        _leg_row("2020-01-25", "short_put", bid=0.9, offer=1.0),
        _leg_row("2020-01-25", "long_put", bid=0.2, offer=0.3),
        _leg_row("2020-01-25", "short_call", bid=0.4, offer=0.5),
        _leg_row("2020-01-25", "long_call", bid=0.1, offer=0.2),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_date == pd.Timestamp("2020-01-25")


def test_put_only_trade_has_no_call_pnl_breakdown():
    trade = _trade()
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long", bid=0.2, offer=0.3),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.call_pnl is None
    assert result.put_pnl == result.pnl


def test_mid_fill_mode_uses_midpoint_for_both_legs_at_entry():
    trade = _trade()  # short: bid=3.0 offer=3.2 -> mid=3.1; long: bid=1.0 offer=1.2 -> mid=1.1
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=2.9, offer=3.0),
        _leg_row("2020-01-20", "long", bid=1.0, offer=1.1),
    ])
    result = PositionSimulator().simulate(trade, history, _config(fill_mode="mid"))
    assert np.isclose(result.entry_credit, 3.1 - 1.1)


def test_mid_fill_mode_uses_midpoint_for_both_legs_at_exit():
    trade = _trade()
    history = pd.DataFrame([
        # short mid = (0.9+1.0)/2=0.95; long mid = (0.2+0.3)/2=0.25 -> cost_to_close = 0.70
        _leg_row("2020-01-20", "short", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long", bid=0.2, offer=0.3),
    ])
    result = PositionSimulator().simulate(trade, history, _config(fill_mode="mid", profit_target_pct=10.0, stop_loss_multiple=10.0))
    assert np.isclose(result.exit_debit, 0.70)


def test_bid_ask_mode_is_unchanged_default():
    trade = _trade()
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long", bid=0.2, offer=0.3),
    ])
    bid_ask_result = PositionSimulator().simulate(trade, history, _config(fill_mode="bid_ask"))
    default_result = PositionSimulator().simulate(trade, history, _config())
    assert bid_ask_result == default_result


def test_reprice_at_exit_holds_timing_fixed_and_only_changes_fill_price():
    trade = _trade()
    history = pd.DataFrame([
        # bid_ask cost_to_close = 1.0 - 0.2 = 0.8; mid cost_to_close = 0.95 - 0.25 = 0.70
        _leg_row("2020-01-20", "short", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long", bid=0.2, offer=0.3),
    ])
    bid_ask_result = PositionSimulator().simulate(trade, history, _config())
    repriced = PositionSimulator().reprice_at_exit(
        trade, history, bid_ask_result.exit_date, bid_ask_result.exit_reason, fill_mode="mid"
    )

    assert repriced.exit_date == bid_ask_result.exit_date
    assert repriced.exit_reason == bid_ask_result.exit_reason
    assert np.isclose(repriced.exit_debit, 0.70)
    assert repriced.exit_debit < bid_ask_result.exit_debit  # mid is more favorable to the seller here


def test_reprice_at_exit_handles_iron_condor_legs():
    """Mutation-testing gap (2026-07-12): reprice_at_exit's own
    `entry_credit = (P or 0.0) + (C or 0.0)` line survived being flipped to
    `-`/`and` because this test only checked put_pnl/call_pnl are non-None,
    never their actual value or entry_credit/pnl -- unlike simulate()'s
    identical line, which IS pinned by test_iron_condor_entry_credit_combines_both_spreads.
    Now asserts real numbers so the combined-credit arithmetic is load-bearing here too."""
    trade = _iron_condor_trade()
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short_put", bid=0.9, offer=1.0),
        _leg_row("2020-01-20", "long_put", bid=0.2, offer=0.3),
        _leg_row("2020-01-20", "short_call", bid=0.4, offer=0.5),
        _leg_row("2020-01-20", "long_call", bid=0.1, offer=0.2),
    ])
    repriced = PositionSimulator().reprice_at_exit(trade, history, "2020-01-20", "profit_target", fill_mode="mid")

    # mid fill mode: put entry credit = 3.1-1.1=2.0, call entry credit = 2.1-0.9=1.2 -> combined 3.2
    # put cost_to_close = 0.95-0.25=0.70 -> put_pnl=1.30; call cost_to_close=0.45-0.15=0.30 -> call_pnl=0.90
    assert np.isclose(repriced.entry_credit, 3.2)
    assert np.isclose(repriced.put_pnl, 1.30)
    assert np.isclose(repriced.call_pnl, 0.90)
    assert np.isclose(repriced.pnl, 2.20)  # additive combination, not put_pnl - call_pnl


def test_check_exit_profit_target_boundary_is_inclusive():
    """profit_so_far exactly equal to the trigger must fire -- mutation
    testing found `>=` silently survives as `>` here (2026-07-12). Uses
    integer-valued inputs so the boundary lands bit-exact in float64."""
    trade = _trade()
    config = _config(profit_target_pct=0.50)
    entry_credit = 2.0  # trigger_basis = abs(2.0) = 2.0, threshold = 0.50*2.0 = 1.0
    cost_to_close = entry_credit - 1.0  # profit_so_far = entry_credit - cost_to_close == 1.0 exactly
    result = check_exit(trade, pd.Timestamp("2020-01-25"), cost_to_close=cost_to_close, entry_credit=entry_credit,
                         dte_remaining=30, put_pnl=1.0, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "profit_target"


def test_check_exit_hard_stop_fields_are_not_shifted_for_a_mixed_put_call_trade():
    """Mutation testing found the hard_stop TradeResult call survives losing
    its `-loss` positional entirely (2026-07-12) -- silently SHIFTS every
    later field over by one instead of crashing (pnl gets put_pnl's value,
    put_pnl gets call_pnl's, call_pnl defaults to None) rather than raising,
    since put_pnl/call_pnl both have defaults. For a put-only trade pnl and
    put_pnl are mathematically identical (equivalent mutant, unkillable) --
    only a mixed put+call trade where all three values genuinely differ can
    catch the shift."""
    trade = _iron_condor_trade()
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short_put", bid=5.0, offer=5.2),
        _leg_row("2020-01-20", "long_put", bid=0.2, offer=0.3),
        _leg_row("2020-01-20", "short_call", bid=3.5, offer=3.7),
        _leg_row("2020-01-20", "long_call", bid=0.2, offer=0.3),
    ])
    # entry_credit = 1.8(put)+1.0(call) = 2.8; put cost_to_close=5.2-0.2=5.0,
    # call cost_to_close=3.7-0.2=3.5 -> combined=8.5, loss=5.7 >= 2.0*2.8=5.6 -> hard_stop
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "hard_stop"
    assert np.isclose(result.pnl, -5.7)
    assert np.isclose(result.put_pnl, 1.8 - 5.0)
    assert np.isclose(result.call_pnl, 1.0 - 3.5)
    # the three values must be genuinely distinct for this test to have teeth
    assert len({round(result.pnl, 2), round(result.put_pnl, 2), round(result.call_pnl, 2)}) == 3


def test_simulate_excludes_entry_date_itself_from_check_dates():
    """`d > trade.entry_date` (strict) survived being loosened to `>=`
    (2026-07-12) -- checking exit conditions using the entry day's own
    pricing would mean "entered and immediately exited," not a genuine
    price-has-moved check. A leg_history row dated exactly on entry_date
    with numbers that would obviously fire hard_stop must be ignored; the
    real check date's (milder) numbers should decide the outcome instead."""
    trade = _trade(entry_date="2020-01-15", exdate="2020-02-15")
    history = pd.DataFrame([
        # on entry_date itself: would be an enormous, obvious hard_stop if checked
        _leg_row("2020-01-15", "short", bid=50.0, offer=55.0),
        _leg_row("2020-01-15", "long", bid=0.1, offer=0.2),
        # the real, later check date: mild move, no trigger, far from dte_stop
        _leg_row("2020-01-20", "short", bid=2.95, offer=3.05),
        _leg_row("2020-01-20", "long", bid=0.95, offer=1.05),
    ])
    result = PositionSimulator().simulate(trade, history, _config())

    assert result.exit_reason == "data_exhausted"
    assert result.exit_date == pd.Timestamp("2020-01-20")


def test_partial_spread_capture_interpolates_between_bid_ask_and_mid():
    trade = _trade()  # short: bid=3.0 offer=3.2; long: bid=1.0 offer=1.2
    history = pd.DataFrame([
        _leg_row("2020-01-20", "short", bid=2.9, offer=3.0),
        _leg_row("2020-01-20", "long", bid=1.0, offer=1.1),
    ])
    zero_capture = PositionSimulator().simulate(trade, history, _config(fill_mode=0.0))
    half_capture = PositionSimulator().simulate(trade, history, _config(fill_mode=0.5))
    full_capture = PositionSimulator().simulate(trade, history, _config(fill_mode=1.0))

    bid_ask = PositionSimulator().simulate(trade, history, _config(fill_mode="bid_ask"))
    mid = PositionSimulator().simulate(trade, history, _config(fill_mode="mid"))

    assert np.isclose(zero_capture.entry_credit, bid_ask.entry_credit)
    assert np.isclose(full_capture.entry_credit, mid.entry_credit)
    # 0.5 capture should land exactly halfway between the two extremes
    assert np.isclose(half_capture.entry_credit, (zero_capture.entry_credit + full_capture.entry_credit) / 2)


def test_no_history_at_all_falls_back_to_entry_date_with_zero_pnl():
    trade = _trade()
    empty = pd.DataFrame(columns=["date", "optionid", "best_bid", "best_offer"])
    result = PositionSimulator().simulate(trade, empty, _config())

    assert result.exit_reason == "data_exhausted"
    assert result.exit_date == trade.entry_date
    assert result.pnl == 0.0


def test_check_exit_returns_none_when_nothing_fires():
    trade = _trade()
    result = check_exit(trade, pd.Timestamp("2020-01-20"), cost_to_close=1.7, entry_credit=1.8,
                         dte_remaining=26, put_pnl=0.1, call_pnl=None, config=_config())
    assert result is None


def test_check_exit_fires_profit_target():
    trade = _trade()
    result = check_exit(trade, pd.Timestamp("2020-01-20"), cost_to_close=0.8, entry_credit=1.8,
                         dte_remaining=26, put_pnl=1.0, call_pnl=None, config=_config())
    assert result.exit_reason == "profit_target"
    assert result.pnl == 1.0


def test_check_exit_fires_hard_stop():
    trade = _trade()
    result = check_exit(trade, pd.Timestamp("2020-01-20"), cost_to_close=5.4, entry_credit=1.8,
                         dte_remaining=26, put_pnl=-3.6, call_pnl=None, config=_config())
    assert result.exit_reason == "hard_stop"
    assert np.isclose(result.pnl, -3.6)


def test_check_exit_fires_dte_stop():
    trade = _trade()
    result = check_exit(trade, pd.Timestamp("2020-01-25"), cost_to_close=2.9, entry_credit=1.8,
                         dte_remaining=21, put_pnl=-1.1, call_pnl=None, config=_config())
    assert result.exit_reason == "dte_stop"


def test_check_exit_profit_target_takes_precedence_over_dte_stop_same_day():
    trade = _trade()
    result = check_exit(trade, pd.Timestamp("2020-01-25"), cost_to_close=0.8, entry_credit=1.8,
                         dte_remaining=21, put_pnl=1.0, call_pnl=None, config=_config())
    assert result.exit_reason == "profit_target"


# A same-strike calendar spread (sell near, buy far) is a net debit, so
# entry_credit is negative -- these confirm check_exit uses abs(entry_credit)
# and doesn't silently invert for a debit trade (the bug found and fixed
# while designing the calendar spread strategy).

def test_check_exit_profit_target_fires_correctly_for_a_debit_trade():
    trade = _trade()
    # entry_credit=-3 (paid $3 debit); cost_to_close=-7 means proceeds of $7
    # at close -> profit_so_far = -3 - (-7) = 4, well above 50% of the $3 paid (1.5).
    result = check_exit(trade, pd.Timestamp("2020-01-20"), cost_to_close=-7.0, entry_credit=-3.0,
                         dte_remaining=26, put_pnl=4.0, call_pnl=None, config=_config())
    assert result.exit_reason == "profit_target"
    assert np.isclose(result.pnl, 4.0)


def test_check_exit_hard_stop_fires_correctly_for_a_debit_trade():
    trade = _trade()
    # entry_credit=-3; cost_to_close=3 (now costs $3 more on top of the
    # original debit) -> loss = 3 - (-3) = 6 = 2.0x the $3 paid -> hard stop.
    result = check_exit(trade, pd.Timestamp("2020-01-20"), cost_to_close=3.0, entry_credit=-3.0,
                         dte_remaining=26, put_pnl=-6.0, call_pnl=None, config=_config())
    assert result.exit_reason == "hard_stop"
    assert np.isclose(result.pnl, -6.0)


def test_check_exit_does_not_fire_on_a_small_debit_trade_gain():
    trade = _trade()
    # entry_credit=-3; cost_to_close=-3.5 -> profit_so_far=0.5, a small real
    # gain, well under the 1.5 profit-target threshold (50% of $3) and well
    # under the 6.0 hard-stop threshold. The un-abs'd formula this replaced
    # would have incorrectly fired profit_target here (0.5 >= 0.5*-3=-1.5).
    result = check_exit(trade, pd.Timestamp("2020-01-20"), cost_to_close=-3.5, entry_credit=-3.0,
                         dte_remaining=26, put_pnl=0.5, call_pnl=None, config=_config())
    assert result is None


def test_mark_to_market_put_only_matches_simulate():
    trade = _trade()
    quotes = {"short": (0.9, 1.0), "long": (0.2, 0.3)}
    entry_credit_by_right = {"P": 1.8, "C": None}
    cost_to_close, put_pnl, call_pnl = mark_to_market(trade, quotes, entry_credit_by_right, "bid_ask")

    history = pd.DataFrame([_leg_row("2020-01-20", "short", 0.9, 1.0), _leg_row("2020-01-20", "long", 0.2, 0.3)])
    simulated = PositionSimulator().simulate(trade, history, _config())

    assert np.isclose(cost_to_close, simulated.exit_debit)
    assert np.isclose(put_pnl, simulated.put_pnl)
    assert call_pnl is None
    assert simulated.call_pnl is None


def test_mark_to_market_iron_condor_matches_simulate_breakdown():
    trade = _iron_condor_trade()
    quotes = {
        "short_put": (0.9, 1.0), "long_put": (0.2, 0.3),
        "short_call": (0.4, 0.5), "long_call": (0.1, 0.2),
    }
    entry_credit_by_right = {"P": 1.8, "C": 1.0}
    cost_to_close, put_pnl, call_pnl = mark_to_market(trade, quotes, entry_credit_by_right, "bid_ask")

    history = pd.DataFrame([
        _leg_row("2020-01-20", "short_put", 0.9, 1.0), _leg_row("2020-01-20", "long_put", 0.2, 0.3),
        _leg_row("2020-01-20", "short_call", 0.4, 0.5), _leg_row("2020-01-20", "long_call", 0.1, 0.2),
    ])
    simulated = PositionSimulator().simulate(trade, history, _config())

    assert np.isclose(cost_to_close, simulated.exit_debit)
    assert np.isclose(put_pnl, simulated.put_pnl)
    assert np.isclose(call_pnl, simulated.call_pnl)


def test_entry_value_by_right_matches_simulate_entry_credit_for_a_put_spread():
    trade = _trade()  # short bid=3.0/offer=3.2, long bid=1.0/offer=1.2
    history = pd.DataFrame([_leg_row("2020-01-20", "short", 0.9, 1.0), _leg_row("2020-01-20", "long", 0.2, 0.3)])
    simulated = PositionSimulator().simulate(trade, history, _config())

    estimate = entry_value_by_right(trade, capture=0.0)

    assert np.isclose(estimate["P"], simulated.entry_credit)
    assert estimate["C"] is None


def test_entry_value_by_right_is_negative_for_a_net_debit_calendar():
    near_leg = _leg("near", 100, -0.50, bid=4.0, offer=4.2, right="P", side="short")
    far_leg = _leg("far", 100, -0.45, bid=6.0, offer=6.2, right="P", side="long")
    calendar = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[near_leg, far_leg])

    # bid_ask (capture=0.0): sell near at bid (4.0), buy far at offer (6.2) -> net = -2.2 (a debit)
    estimate = entry_value_by_right(calendar, capture=0.0)

    assert np.isclose(estimate["P"], -2.2)


def test_net_credit_default_weight_is_unchanged_from_pre_weight_behavior():
    legs = [_leg("short", 95, -0.20, bid=3.0, offer=3.2, right="P", side="short")]
    assert net_credit(legs, lambda leg: leg.best_bid) == 3.0


def test_net_credit_scales_by_leg_weight():
    weighted_leg = Leg(
        optionid="w", exdate=pd.Timestamp("2020-02-15"), strike_price=95, delta=-0.20,
        best_bid=3.0, best_offer=3.2, right="P", side="short", weight=10.0,
    )
    assert net_credit([weighted_leg], lambda leg: leg.best_bid) == 30.0

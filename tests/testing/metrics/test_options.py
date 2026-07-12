import numpy as np
import pandas as pd
import pytest

from testing.engine.leg import Leg
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import TradeResult
from testing.engine.sizing import SizedTradeResult
from testing.metrics.options import (
    GREEK_COLUMNS,
    greeks_history_from_chain,
    portfolio_greek_exposure,
    position_greek_exposure,
)


def _leg(optionid, right, side, exdate="2020-02-15"):
    # strike_price/best_bid/best_offer are irrelevant to Greek exposure --
    # held at placeholder values; only optionid/right/side/exdate matter here.
    return Leg(
        optionid=optionid, exdate=pd.Timestamp(exdate), strike_price=100.0,
        delta=0.0, best_bid=1.0, best_offer=1.1, right=right, side=side,
    )


def _greek_row(date, optionid, delta, gamma, vega, theta):
    return {
        "date": pd.Timestamp(date), "optionid": optionid,
        "delta": delta, "gamma": gamma, "vega": vega, "theta": theta,
    }


def _vertical_spread(entry_date="2020-01-15", exdate="2020-02-15"):
    # A put credit spread: short the -0.20 delta put, long the -0.10 delta put
    # -- the exact structure vrp_harvester's default put_credit_spread mode
    # builds (VRPConfig's short_put_delta=-0.20, long_put_delta=-0.10).
    short_put = _leg("short_put", right="P", side="short", exdate=exdate)
    long_put = _leg("long_put", right="P", side="long", exdate=exdate)
    return PositionSpec(entry_date=pd.Timestamp(entry_date), legs=[short_put, long_put])


def _iron_condor(entry_date="2020-01-15", exdate="2020-02-15"):
    short_put = _leg("short_put", right="P", side="short", exdate=exdate)
    long_put = _leg("long_put", right="P", side="long", exdate=exdate)
    short_call = _leg("short_call", right="C", side="short", exdate=exdate)
    long_call = _leg("long_call", right="C", side="long", exdate=exdate)
    return PositionSpec(entry_date=pd.Timestamp(entry_date), legs=[short_put, long_put, short_call, long_call])


def test_vertical_spread_aggregates_signed_greeks_across_both_legs():
    trade = _vertical_spread()
    # Known per-share Greeks on the entry day:
    # short put: delta=-0.20, gamma=0.01, vega=0.05, theta=-0.02
    # long put:  delta=-0.10, gamma=0.008, vega=0.03, theta=-0.015
    history = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
    ])
    result = position_greek_exposure(trade, exit_date="2020-01-15", greeks_history=history, contracts=1, contract_multiplier=1)

    day = result.loc[pd.Timestamp("2020-01-15")]
    # short put contributes -1 * its own greek (short = -1 sign); long put
    # contributes +1 * its own greek (long = +1 sign).
    assert np.isclose(day["delta"], -1 * -0.20 + 1 * -0.10)  # 0.20 - 0.10 = 0.10
    assert np.isclose(day["gamma"], -1 * 0.01 + 1 * 0.008)  # -0.01 + 0.008 = -0.002
    assert np.isclose(day["vega"], -1 * 0.05 + 1 * 0.03)  # -0.05 + 0.03 = -0.02
    assert np.isclose(day["theta"], -1 * -0.02 + 1 * -0.015)  # 0.02 - 0.015 = 0.005


def test_vertical_spread_scales_by_contracts_and_contract_multiplier():
    trade = _vertical_spread()
    history = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
    ])
    per_contract = position_greek_exposure(trade, "2020-01-15", history, contracts=1, contract_multiplier=1)
    scaled = position_greek_exposure(trade, "2020-01-15", history, contracts=3, contract_multiplier=100)

    day_per_contract = per_contract.loc[pd.Timestamp("2020-01-15")]
    day_scaled = scaled.loc[pd.Timestamp("2020-01-15")]
    for col in GREEK_COLUMNS:
        assert np.isclose(day_scaled[col], day_per_contract[col] * 300)


def test_iron_condor_aggregates_all_four_legs():
    trade = _iron_condor()
    history = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
        _greek_row("2020-01-15", "short_call", 0.30, 0.012, 0.06, -0.025),
        _greek_row("2020-01-15", "long_call", 0.20, 0.009, 0.04, -0.018),
    ])
    result = position_greek_exposure(trade, "2020-01-15", history, contracts=1, contract_multiplier=1)
    day = result.loc[pd.Timestamp("2020-01-15")]

    expected_delta = (-1 * -0.20) + (1 * -0.10) + (-1 * 0.30) + (1 * 0.20)  # 0.20 -0.10 -0.30 +0.20 = 0.0
    expected_gamma = (-1 * 0.01) + (1 * 0.008) + (-1 * 0.012) + (1 * 0.009)
    expected_vega = (-1 * 0.05) + (1 * 0.03) + (-1 * 0.06) + (1 * 0.04)
    expected_theta = (-1 * -0.02) + (1 * -0.015) + (-1 * -0.025) + (1 * -0.018)

    assert np.isclose(day["delta"], expected_delta)
    assert np.isclose(day["gamma"], expected_gamma)
    assert np.isclose(day["vega"], expected_vega)
    assert np.isclose(day["theta"], expected_theta)


def test_position_exposure_is_a_time_series_across_the_holding_period():
    trade = _vertical_spread(entry_date="2020-01-15", exdate="2020-02-15")
    history = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
        # Delta moves as spot/vol change day to day -- a real time series,
        # not a single entry-day snapshot repeated.
        _greek_row("2020-01-20", "short_put", -0.35, 0.015, 0.06, -0.03),
        _greek_row("2020-01-20", "long_put", -0.18, 0.011, 0.035, -0.02),
    ])
    result = position_greek_exposure(trade, "2020-01-20", history, contracts=1, contract_multiplier=1)

    assert list(result.index) == [pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-20")]
    assert np.isclose(result.loc[pd.Timestamp("2020-01-15"), "delta"], 0.10)
    assert np.isclose(result.loc[pd.Timestamp("2020-01-20"), "delta"], -1 * -0.35 + 1 * -0.18)  # 0.17


def test_position_exposure_excludes_rows_outside_the_holding_window():
    trade = _vertical_spread(entry_date="2020-01-15", exdate="2020-02-15")
    history = pd.DataFrame([
        _greek_row("2020-01-10", "short_put", -0.15, 0.01, 0.05, -0.02),  # before entry -- excluded
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
        _greek_row("2020-03-01", "short_put", -0.50, 0.02, 0.08, -0.04),  # after exit -- excluded
    ])
    result = position_greek_exposure(trade, exit_date="2020-01-15", greeks_history=history)

    assert list(result.index) == [pd.Timestamp("2020-01-15")]


def test_position_exposure_empty_history_returns_empty_frame_with_greek_columns():
    trade = _vertical_spread()
    empty = pd.DataFrame(columns=["date", "optionid", "delta", "gamma", "vega", "theta"])
    result = position_greek_exposure(trade, "2020-01-15", empty)

    assert result.empty
    assert list(result.columns) == GREEK_COLUMNS


def _trade_result(entry_date, exit_date, exit_reason="dte_stop"):
    return TradeResult(
        entry_date=pd.Timestamp(entry_date), exit_date=pd.Timestamp(exit_date),
        exit_reason=exit_reason, entry_credit=1.8, exit_debit=1.0, pnl=0.8,
    )


def test_portfolio_exposure_sums_two_concurrent_positions_on_overlapping_dates():
    trade_a = _vertical_spread(entry_date="2020-01-15", exdate="2020-02-15")
    trade_b = PositionSpec(
        entry_date=pd.Timestamp("2020-01-16"),
        legs=[_leg("b_short_put", right="P", side="short", exdate="2020-02-20"),
              _leg("b_long_put", right="P", side="long", exdate="2020-02-20")],
    )
    result_a = _trade_result("2020-01-15", "2020-01-20")
    result_b = _trade_result("2020-01-16", "2020-01-22")

    sized_a = SizedTradeResult(trade_result=result_a, contracts=1, pnl_dollars=80.0)
    sized_b = SizedTradeResult(trade_result=result_b, contracts=1, pnl_dollars=80.0)

    history = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
        _greek_row("2020-01-16", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-16", "long_put", -0.10, 0.008, 0.03, -0.015),
        _greek_row("2020-01-16", "b_short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-16", "b_long_put", -0.10, 0.008, 0.03, -0.015),
    ])

    portfolio = portfolio_greek_exposure(
        trades=[(trade_a, result_a), (trade_b, result_b)],
        sized_trades=[sized_a, sized_b],
        greeks_history=history,
        contract_multiplier=1,
    )

    # 2020-01-15: only trade_a is open -> delta = 0.10
    assert np.isclose(portfolio.loc[pd.Timestamp("2020-01-15"), "delta"], 0.10)
    # 2020-01-16: both trade_a and trade_b open -> delta = 0.10 + 0.10 = 0.20
    assert np.isclose(portfolio.loc[pd.Timestamp("2020-01-16"), "delta"], 0.20)


def test_portfolio_exposure_empty_trades_returns_empty_frame_with_greek_columns():
    result = portfolio_greek_exposure(trades=[], sized_trades=[], greeks_history=pd.DataFrame(columns=["date", "optionid", *GREEK_COLUMNS]))
    assert result.empty
    assert list(result.columns) == GREEK_COLUMNS


def test_portfolio_exposure_rejects_mismatched_trade_and_sizing_lengths():
    trade = _vertical_spread()
    result = _trade_result("2020-01-15", "2020-01-20")
    sized = SizedTradeResult(trade_result=result, contracts=1, pnl_dollars=80.0)

    with pytest.raises(ValueError, match="same length"):
        portfolio_greek_exposure(trades=[(trade, result)], sized_trades=[sized, sized], greeks_history=pd.DataFrame())


def test_portfolio_exposure_reindexes_to_a_given_date_range_with_zero_fill():
    trade = _vertical_spread(entry_date="2020-01-15", exdate="2020-02-15")
    result = _trade_result("2020-01-15", "2020-01-15")
    sized = SizedTradeResult(trade_result=result, contracts=1, pnl_dollars=80.0)
    history = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
    ])
    date_range = pd.date_range("2020-01-14", "2020-01-16")

    portfolio = portfolio_greek_exposure([(trade, result)], [sized], history, contract_multiplier=1, date_range=date_range)

    assert list(portfolio.index) == list(date_range)
    assert portfolio.loc[pd.Timestamp("2020-01-14"), "delta"] == 0.0
    assert np.isclose(portfolio.loc[pd.Timestamp("2020-01-15"), "delta"], 0.10)
    assert portfolio.loc[pd.Timestamp("2020-01-16"), "delta"] == 0.0


def test_greeks_history_from_chain_slices_to_relevant_optionids_and_dates():
    trade = _vertical_spread(entry_date="2020-01-15", exdate="2020-02-15")
    result = _trade_result("2020-01-15", "2020-01-20")

    chain = pd.DataFrame([
        _greek_row("2020-01-15", "short_put", -0.20, 0.01, 0.05, -0.02),
        _greek_row("2020-01-15", "long_put", -0.10, 0.008, 0.03, -0.015),
        _greek_row("2020-01-15", "unrelated_option", 0.5, 0.02, 0.1, -0.05),  # different position -- excluded
        _greek_row("2020-03-01", "short_put", -0.50, 0.02, 0.08, -0.04),  # outside window -- excluded
    ])

    sliced = greeks_history_from_chain(chain, trades=[(trade, result)])

    assert set(sliced["optionid"]) == {"short_put", "long_put"}
    assert list(sliced.columns) == ["date", "optionid", *GREEK_COLUMNS]


def test_greeks_history_from_chain_empty_trades_returns_empty_frame():
    chain = pd.DataFrame([_greek_row("2020-01-15", "x", 0.1, 0.01, 0.02, -0.01)])
    result = greeks_history_from_chain(chain, trades=[])
    assert result.empty
    assert list(result.columns) == ["date", "optionid", *GREEK_COLUMNS]

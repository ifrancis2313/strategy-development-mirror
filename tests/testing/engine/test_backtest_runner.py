"""Tests `walk_forward`'s day-by-day driver in isolation from any real
strategy, using a minimal duck-typed selector double -- this doubles as a
regression test of the "adding a new strategy" extension contract itself
(CONTEXT.md: any object with a `.select_trade(chain_snapshot, entry_date,
has_open_position, **kwargs)` method plugs in, no registration needed).
"""
import pandas as pd

from testing.engine.backtest_runner import CHAIN_COLUMNS, summarize, walk_forward
from testing.engine.execution_config import ExecutionConfig
from testing.engine.leg import Leg
from testing.engine.position import PositionSpec


def _chain_row(date, exdate, cp_flag, strike, bid, offer, delta, optionid, secid=1):
    return {
        "secid": secid, "date": pd.Timestamp(date), "exdate": pd.Timestamp(exdate),
        "cp_flag": cp_flag, "strike_price": strike, "best_bid": bid, "best_offer": offer,
        "delta": delta, "impl_volatility": 0.20, "optionid": optionid,
    }


def _synthetic_chain():
    # One candidate entry day (2020-01-15, dte=31) with a matching exit-check
    # day (2020-01-20, dte=26) that clears profit_target -- same trade shape
    # as core.position_simulator's own test fixtures.
    return pd.DataFrame([
        _chain_row("2020-01-15", "2020-02-15", "P", 95, 3.0, 3.2, -0.20, "S"),
        _chain_row("2020-01-15", "2020-02-15", "P", 90, 1.0, 1.2, -0.10, "L"),
        _chain_row("2020-01-20", "2020-02-15", "P", 95, 0.9, 1.0, -0.10, "S"),
        _chain_row("2020-01-20", "2020-02-15", "P", 90, 0.2, 0.3, -0.05, "L"),
    ])


class _FixedSpreadSelector:
    """Minimal selector double: always proposes the same 2-leg put spread,
    regardless of the day's chain snapshot -- `walk_forward`'s own
    `open_until` bookkeeping (not `has_open_position`, which the driver
    always passes as False) is what prevents it firing a second entry while
    the first trade is still open.
    """

    def __init__(self):
        self.calls = []

    def select_trade(self, chain_snapshot, entry_date, has_open_position, **kwargs):
        self.calls.append({"entry_date": pd.Timestamp(entry_date), "kwargs": dict(kwargs)})
        entry_date = pd.Timestamp(entry_date)
        exdate = pd.Timestamp("2020-02-15")
        short_put = Leg(optionid="S", exdate=exdate, strike_price=95, delta=-0.20,
                         best_bid=3.0, best_offer=3.2, right="P", side="short")
        long_put = Leg(optionid="L", exdate=exdate, strike_price=90, delta=-0.10,
                        best_bid=1.0, best_offer=1.2, right="P", side="long")
        return PositionSpec(entry_date=entry_date, legs=[short_put, long_put])


def _config(**overrides):
    defaults = dict(min_dte=20, max_dte=35, profit_target_pct=0.50, stop_loss_multiple=2.0,
                     exit_dte=5, fill_mode="bid_ask", fixed_notional=100_000.0, secid=1)
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def test_walk_forward_opens_and_simulates_exactly_one_trade():
    chain = _synthetic_chain()
    selector = _FixedSpreadSelector()

    trades = walk_forward(chain, _config(), selector)

    assert len(trades) == 1
    spec, result = trades[0]
    assert spec.entry_date == pd.Timestamp("2020-01-15")
    assert result.entry_credit == 1.8  # short bid 3.0 - long offer 1.2
    assert result.exit_reason == "profit_target"
    assert result.exit_date == pd.Timestamp("2020-01-20")
    assert result.pnl == 1.0  # entry_credit 1.8 - cost_to_close (1.0-0.2=0.8)


def test_walk_forward_does_not_reenter_while_a_trade_is_open():
    # Both 2020-01-15 (dte=31) and 2020-01-20 (dte=26) fall in the [20,35]
    # DTE window, so both are entry candidates -- but the second is <= the
    # first trade's exit_date, so it must be skipped, not just deduped after
    # the fact.
    chain = _synthetic_chain()
    selector = _FixedSpreadSelector()

    trades = walk_forward(chain, _config(), selector)

    assert len(trades) == 1
    assert len(selector.calls) == 1  # never even asked on the skipped day


def test_walk_forward_resolves_a_series_valued_kwarg_to_the_days_value():
    # The generalized per-day signal lookup (`isinstance(value, pd.Series)`
    # -> `.get(entry_date)`), documented in `core/backtest_runner.py` as the
    # generalization of the old hardcoded VRP-only lookup.
    chain = _synthetic_chain()
    selector = _FixedSpreadSelector()
    signal = pd.Series({pd.Timestamp("2020-01-15"): 0.07})

    walk_forward(chain, _config(), selector, my_signal=signal, static_kwarg="unchanged")

    assert selector.calls[0]["kwargs"]["my_signal"] == 0.07
    assert selector.calls[0]["kwargs"]["static_kwarg"] == "unchanged"


def test_walk_forward_dte_candidate_window_includes_max_dte_boundary():
    """`dte <= config.max_dte` -- mutation testing (2026-07-12) found this
    survives being tightened to `< config.max_dte`, which would silently
    exclude a candidate day landing exactly on the DTE ceiling. No prior
    test's fixture landed exactly on that boundary (31/26 days, config
    max_dte=35)."""
    chain = pd.DataFrame([
        # dte = (2020-02-19 - 2020-01-15).days == 35 == config's max_dte exactly
        _chain_row("2020-01-15", "2020-02-19", "P", 95, 3.0, 3.2, -0.20, "S"),
        _chain_row("2020-01-15", "2020-02-19", "P", 90, 1.0, 1.2, -0.10, "L"),
    ])
    selector = _FixedSpreadSelector()

    walk_forward(chain, _config(min_dte=20, max_dte=35), selector)

    assert len(selector.calls) == 1
    assert selector.calls[0]["entry_date"] == pd.Timestamp("2020-01-15")


def test_walk_forward_leg_history_includes_the_expiration_day_itself():
    """`chain["date"] <= trade.exdate` -- mutation testing (2026-07-12)
    found this survives being tightened to `< trade.exdate`, silently
    dropping expiration-day pricing from leg_history. If that day is the
    only post-entry data available, the trade would fall through to
    data_exhausted instead of actually exiting on real exdate pricing.

    The mirror mutation on the left edge (`chain["date"] >= trade.entry_date`
    tightened to `>`) is a confirmed equivalent mutant, not a real gap:
    PositionSimulator.simulate's own ordered_check_dates already excludes
    entry_date itself, and entry-day pricing comes from the trade spec
    object directly (entry_value_by_right), never from leg_history --
    checked by applying that mutant directly and re-running the full
    suite, not left untested by omission."""
    exdate = pd.Timestamp("2020-02-15")
    chain = pd.DataFrame([
        _chain_row("2020-01-15", "2020-02-15", "P", 95, 3.0, 3.2, -0.20, "S"),
        _chain_row("2020-01-15", "2020-02-15", "P", 90, 1.0, 1.2, -0.10, "L"),
        # only other pricing available is exactly on the expiration day
        _chain_row("2020-02-15", "2020-02-15", "P", 95, 0.1, 0.2, -0.01, "S"),
        _chain_row("2020-02-15", "2020-02-15", "P", 90, 0.0, 0.05, -0.001, "L"),
    ])
    selector = _FixedSpreadSelector()

    trades = walk_forward(chain, _config(exit_dte=0), selector)

    assert len(trades) == 1
    _spec, result = trades[0]
    assert result.exit_reason != "data_exhausted"
    assert result.exit_date == exdate


def test_walk_forward_returns_no_trades_when_selector_never_fires():
    chain = _synthetic_chain()

    class _NeverEnters:
        def select_trade(self, chain_snapshot, entry_date, has_open_position, **kwargs):
            return None

    trades = walk_forward(chain, _config(), _NeverEnters())
    assert trades == []


def test_chain_columns_matches_what_position_simulator_and_selectors_need():
    assert set(CHAIN_COLUMNS) == {
        "secid", "date", "exdate", "cp_flag", "strike_price",
        "best_bid", "best_offer", "delta", "impl_volatility", "optionid",
    }


def test_summarize_prints_a_report_without_raising_for_a_real_trade_set(capsys):
    chain = _synthetic_chain()
    trades = walk_forward(chain, _config(), _FixedSpreadSelector())

    summarize("test label", trades, _config())

    out = capsys.readouterr().out
    assert "test label" in out
    assert "Trades:        1" in out


def test_summarize_handles_no_trades_without_raising(capsys):
    summarize("empty label", [], _config())
    out = capsys.readouterr().out
    assert "no trades generated" in out

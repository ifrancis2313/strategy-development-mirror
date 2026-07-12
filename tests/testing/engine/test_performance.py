import numpy as np
import pandas as pd

from testing.engine.performance import PerformanceAnalyzer, PerformanceReport
from testing.engine.portfolio import PortfolioAggregator
from testing.engine.position_simulator import TradeResult
from testing.engine.sizing import SizedTradeResult

_NOTIONAL = 10_000.0


def _sized(entry_date, exit_date, pnl_dollars, exit_reason="profit_target"):
    result = TradeResult(
        entry_date=pd.Timestamp(entry_date), exit_date=pd.Timestamp(exit_date),
        exit_reason=exit_reason, entry_credit=0.0, exit_debit=0.0,
        pnl=pnl_dollars / 100,  # contracts=1, multiplier=100 in these fixtures' implied economics
    )
    return SizedTradeResult(trade_result=result, contracts=1, pnl_dollars=pnl_dollars)


def _four_trade_set():
    # One trade per week in January 2020, mixed win/loss, notional=$10,000 so
    # trade-return fractions and drawdown fractions are easy round numbers.
    return [
        _sized("2020-01-01", "2020-01-08", 1000.0),
        _sized("2020-01-09", "2020-01-16", -3000.0),
        _sized("2020-01-17", "2020-01-24", 2000.0),
        _sized("2020-01-25", "2020-02-01", 500.0),
    ]


def test_analyze_empty_trades_returns_zeroed_report():
    report = PerformanceAnalyzer().analyze([], pd.Series(dtype=float), _NOTIONAL)
    assert report == PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_analyze_num_trades_win_rate_and_avg_pnl():
    sized = _four_trade_set()
    equity = PortfolioAggregator().build_equity_curve(sized)
    report = PerformanceAnalyzer().analyze(sized, equity, _NOTIONAL)

    assert report.num_trades == 4
    assert report.win_rate == 0.75  # 3 of 4 trades (+1000, +2000, +500) are winners
    assert np.isclose(report.avg_pnl_dollars, 125.0)  # (1000-3000+2000+500)/4


def test_win_rate_does_not_count_a_breakeven_trade_as_a_win():
    """`(pnl_dollars > 0).mean()` -- mutation testing (2026-07-12) found the
    boundary survives being loosened to `>= 0`. A trade with exactly zero
    P&L is breakeven, not a win -- no prior fixture used pnl_dollars=0.0."""
    sized = [
        _sized("2020-01-01", "2020-01-08", 1000.0),
        _sized("2020-01-09", "2020-01-16", 0.0),
        _sized("2020-01-17", "2020-01-24", -500.0),
    ]
    equity = PortfolioAggregator().build_equity_curve(sized)
    report = PerformanceAnalyzer().analyze(sized, equity, _NOTIONAL)
    assert np.isclose(report.win_rate, 1 / 3)  # only the +1000 trade counts


def test_analyze_sharpe_and_calmar_match_independent_numpy_derivation():
    # Reference values independently derived from the same trade-return
    # fractions/dates via plain numpy (mean/std of pnl/notional, annualized by
    # trades-per-year implied by the entry/exit date span; drawdown/calmar off
    # the cumulative equity curve) -- not copied from PerformanceAnalyzer's
    # own code path, so this catches a wrong ddof/annualization/drawdown-basis
    # regression rather than just re-asserting the implementation.
    sized = _four_trade_set()
    equity = PortfolioAggregator().build_equity_curve(sized)
    report = PerformanceAnalyzer().analyze(sized, equity, _NOTIONAL)

    assert np.isclose(report.sharpe, 0.3946044369309445)
    assert np.isclose(report.max_drawdown_pct, -0.3)  # trough after trade 2: -2000/10000
    assert np.isclose(report.calmar, 1.963709677419355)


def test_analyze_sharpe_is_zero_when_all_trade_returns_are_identical():
    # Zero-variance fallback (division-by-zero guard), not a real Sharpe of 0.
    # Uses a power-of-two-fraction P&L (2500/10000 = 0.25 exactly in binary
    # floating point) so the mean/variance arithmetic is exact and std is
    # bit-for-bit 0.0 -- an arbitrary identical value like 1000.0 can leave a
    # ~1e-17 floating-point residue in the mean (0.1+0.1+0.1 != 0.3*1 exactly
    # in IEEE754), which would spuriously blow up the ratio instead of
    # exercising the intended zero-variance branch.
    sized = [
        _sized("2020-01-01", "2020-01-08", 2500.0),
        _sized("2020-01-09", "2020-01-16", 2500.0),
        _sized("2020-01-17", "2020-01-24", 2500.0),
    ]
    equity = PortfolioAggregator().build_equity_curve(sized)
    report = PerformanceAnalyzer().analyze(sized, equity, _NOTIONAL)
    assert report.sharpe == 0.0


def test_analyze_calmar_is_zero_when_no_drawdown_ever_occurs():
    # Monotonically increasing equity -> max_drawdown_pct is 0 -> calmar's
    # "< 0" guard means it stays 0.0 rather than a division by zero.
    sized = [
        _sized("2020-01-01", "2020-01-08", 100.0),
        _sized("2020-01-09", "2020-01-16", 100.0),
        _sized("2020-01-17", "2020-01-24", 100.0),
    ]
    equity = PortfolioAggregator().build_equity_curve(sized)
    report = PerformanceAnalyzer().analyze(sized, equity, _NOTIONAL)
    assert report.max_drawdown_pct == 0.0
    assert report.calmar == 0.0


def test_analyze_by_regime_buckets_trades_by_entry_date_label():
    sized = _four_trade_set()
    regime_by_date = {
        pd.Timestamp("2020-01-01"): "calm",
        pd.Timestamp("2020-01-09"): "calm",
        pd.Timestamp("2020-01-17"): "crisis",
        pd.Timestamp("2020-01-25"): "crisis",
    }

    by_regime = PerformanceAnalyzer().analyze_by_regime(sized, regime_by_date, _NOTIONAL)

    assert set(by_regime.keys()) == {"calm", "crisis"}
    assert by_regime["calm"].num_trades == 2
    assert by_regime["crisis"].num_trades == 2
    # calm bucket: +1000, -3000 -> avg -1000; crisis bucket: +2000, +500 -> avg 1250
    assert np.isclose(by_regime["calm"].avg_pnl_dollars, -1000.0)
    assert np.isclose(by_regime["crisis"].avg_pnl_dollars, 1250.0)


def test_analyze_by_regime_unmapped_entry_dates_fall_into_none_bucket():
    sized = _four_trade_set()
    regime_by_date = {pd.Timestamp("2020-01-01"): "calm"}  # only one of four dates mapped

    by_regime = PerformanceAnalyzer().analyze_by_regime(sized, regime_by_date, _NOTIONAL)

    assert by_regime[None].num_trades == 3
    assert by_regime["calm"].num_trades == 1

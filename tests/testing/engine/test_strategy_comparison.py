"""Tests the backtest-engine-agnostic sign-permutation MCPT
(`evaluate_strategy_pnl`) and the benchmark-curve helper in isolation from
any real chain data -- `run_multi_strategy_comparison` itself is covered by
a real-data verification run outside this mirror, not a unit test, since it
requires a real option chain to be meaningful.

Per the PRD's Testing Decisions, a sign-permutation p-value calculation is
exactly the kind of "easy to silently get wrong" logic that gets real,
non-obvious-logic coverage here rather than being left to eyeballing.
"""
import numpy as np
import pandas as pd

from testing.engine.strategy_comparison import build_benchmark_equity_curve, evaluate_strategy_pnl


def _equity_curve(pnl, dates):
    return pd.Series(np.cumsum(pnl), index=pd.to_datetime(dates), name="equity")


def test_empty_trade_pnl_returns_the_degenerate_result():
    result = evaluate_strategy_pnl("x", [], pd.Series(dtype=float))
    assert result == {"sharpe": None, "mcpt": {"p_value": None, "n_permutations": 0, "null_distribution": []}}


def test_empty_equity_curve_returns_the_degenerate_result():
    result = evaluate_strategy_pnl("x", [10.0, -5.0], pd.Series(dtype=float))
    assert result["sharpe"] is None


def test_zero_variance_pnl_returns_none_sharpe_not_a_real_zero():
    # observed Sharpe is nan (std=0) -- must be reported as None (unknown),
    # never silently coerced to a real Sharpe of 0.0.
    pnl = [100.0, 100.0, 100.0]
    equity = _equity_curve(pnl, ["2020-01-01", "2020-02-01", "2020-03-01"])
    result = evaluate_strategy_pnl("x", pnl, equity)
    assert result["sharpe"] is None
    assert result["mcpt"]["p_value"] is None


def test_p_value_matches_the_documented_formula_for_a_single_permutation():
    # n_permutations=1 is fully deterministic (rng seed 0) -- reproduces the
    # exact null Sharpe and p_value independently derived via plain numpy
    # (see this test's docstring-adjacent derivation, not copied from the
    # implementation): only the equity curve's index span (366 days here)
    # feeds the annualization, so its values are irrelevant and left at 0.
    # observed=0.9206008757791715; the seed-0 permutation flips all three
    # signs (null=-observed), so p_value=(1+0)/(1+1)=0.5.
    pnl = [10.0, -5.0, 8.0]
    equity = pd.Series([0.0, 0.0], index=pd.to_datetime(["2020-01-01", "2021-01-01"]))

    result = evaluate_strategy_pnl("x", pnl, equity, n_permutations=1)

    assert np.isclose(result["sharpe"], 0.9206008757791715)
    assert result["mcpt"]["n_permutations"] == 1
    assert np.isclose(result["mcpt"]["null_distribution"][0], -0.9206008757791715)
    assert result["mcpt"]["p_value"] == 0.5


def test_p_value_is_low_for_a_strong_consistent_edge():
    # 20 trades, all positive with modest noise -> real Sharpe is large and
    # positive; sign-flipping ~half the trades in most permutations should
    # rarely produce a null Sharpe that beats it, so p_value should land low
    # (a real edge is distinguishable from the permuted null, the whole
    # point of the test).
    rng = np.random.default_rng(7)
    pnl = list(100.0 + rng.normal(0, 5, size=20))
    dates = pd.date_range("2020-01-01", periods=20, freq="30D")
    equity = _equity_curve(pnl, dates)

    result = evaluate_strategy_pnl("x", pnl, equity, n_permutations=500)

    assert result["sharpe"] > 0
    assert result["mcpt"]["p_value"] < 0.1


def test_p_value_and_null_distribution_are_reproducible_across_calls():
    # Each permutation draw is seeded by its own index (np.random.default_rng(i)),
    # so re-running with the same inputs must be bit-for-bit reproducible --
    # required for a comparison result to be a trustworthy audit artifact.
    pnl = [12.0, -8.0, 15.0, -3.0, 9.0, -11.0]
    equity = _equity_curve(pnl, pd.date_range("2020-01-01", periods=6, freq="60D"))

    first = evaluate_strategy_pnl("x", pnl, equity, n_permutations=50)
    second = evaluate_strategy_pnl("x", pnl, equity, n_permutations=50)

    assert first == second


def test_build_benchmark_equity_curve_is_rebased_to_zero_at_first_observation():
    rows = [
        {"date": pd.Timestamp("2020-01-01"), "exdate": pd.Timestamp("2020-01-31"),
         "cp_flag": "C", "strike_price": 100.0, "best_bid": 3.0, "best_offer": 3.2},
        {"date": pd.Timestamp("2020-01-01"), "exdate": pd.Timestamp("2020-01-31"),
         "cp_flag": "P", "strike_price": 100.0, "best_bid": 3.0, "best_offer": 3.2},
        {"date": pd.Timestamp("2020-01-02"), "exdate": pd.Timestamp("2020-01-31"),
         "cp_flag": "C", "strike_price": 100.0, "best_bid": 4.0, "best_offer": 4.2},
        {"date": pd.Timestamp("2020-01-02"), "exdate": pd.Timestamp("2020-01-31"),
         "cp_flag": "P", "strike_price": 100.0, "best_bid": 3.0, "best_offer": 3.2},
    ]
    chain = pd.DataFrame(rows)

    curve = build_benchmark_equity_curve(chain, notional=100_000.0, target_dte=30)

    assert curve.iloc[0] == 0.0  # rebased so day 1 always reads as zero cumulative P&L
    assert curve.name == "cumulative_value"


def test_build_benchmark_equity_curve_empty_chain_returns_empty_series():
    empty = pd.DataFrame(columns=["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer"])
    curve = build_benchmark_equity_curve(empty)
    assert curve.empty

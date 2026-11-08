"""Offline synthetic illustration using the repository's validation APIs."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from testing.engine.strategy_comparison import StrategyRunner
from testing.graduation.bar import check_graduation_bar
from testing.monte_carlo.mcpt import sign_permutation_mcpt
from testing.monte_carlo.pbo import probability_of_backtest_overfitting
from testing.validation.walk_forward_validation import run_walk_forward_validation

DISCLAIMER = "SYNTHETIC DATA — demonstrates the validation machinery; not a strategy or a result."
SKIP = "skipped: PBO needs the external pypbo-cscv implementation (AGPL, not included — see README, License)"
LOOKBACKS = (5, 10, 20)


def synthetic_prices(seed: int) -> pd.DataFrame:
    """Generate prices from zero-drift independent synthetic log increments."""
    dates = pd.bdate_range("2015-01-01", "2023-01-02")
    prices = 100 * np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.01, len(dates))))
    return pd.DataFrame({"date": dates, "price": prices})


def toy_returns(data: pd.DataFrame, lookback: int) -> pd.Series:
    """Lagged toy long/short rule; no fills or trading costs are modeled."""
    prices = data.set_index("date")["price"]
    average = prices.rolling(lookback).mean()
    signal = pd.Series(np.where(prices > average, 1.0, -1.0), index=prices.index)
    signal = signal.where(average.notna(), 0.0).shift(1).fillna(0.0)
    return (signal * prices.pct_change().fillna(0.0)).rename(f"ma_{lookback}")


def toy_runner(lookback: int) -> StrategyRunner:
    def run(data: pd.DataFrame) -> tuple[list[float], pd.Series, dict]:
        values = toy_returns(data, lookback)
        # Each daily observation is a unit-notional toy P&L observation.
        return values.tolist(), values.cumsum(), {}

    return run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(DISCLAIMER)
    print(f"Seed: {args.seed}; toy moving-average lookbacks: {LOOKBACKS}")
    print("Synthetic unit-notional observations only; no fills or trading costs modeled.")
    print(
        "Prices are a zero-drift random walk, so any apparent performance above is noise; "
        "the Monte Carlo p-values are the check that flags it."
    )
    data = synthetic_prices(args.seed)
    runners = {f"ma_{n}": toy_runner(n) for n in LOOKBACKS}
    walk_forward = run_walk_forward_validation(data, runners, train_years=3, test_years=1)
    print(walk_forward.summary())
    panel = pd.concat([toy_returns(data, n) for n in LOOKBACKS], axis=1)
    results = []
    for name in panel:
        result = sign_permutation_mcpt(panel[name], n_permutations=1000)
        if result is None:
            raise RuntimeError("Synthetic toy observations unexpectedly degenerate")
        results.append(result)
        print(f"MCPT {name}: p={result.p_value}; permutations={result.n_permutations}")
    print("MCPT values describe the full synthetic series, not a held-out conclusion.")
    try:
        pbo = probability_of_backtest_overfitting(panel)
    except ImportError:
        print("PBO:")
        print(SKIP)
        print("Graduation gate:")
        print(SKIP)
        return
    print(f"PBO: {pbo.pbo:.6f}; combinations={pbo.n_combinations}")
    for name, result in zip(panel, results, strict=True):
        gate = check_graduation_bar(
            result, pbo, n_tests=len(LOOKBACKS),
            has_named_source=False, has_sensitivity_analysis=False,
        )
        print(f"Graduation gate {name}: {'pass' if gate.ready_to_graduate else 'fail'}; "
              f"MCPT={gate.mcpt_verdict}; PBO={gate.pbo_verdict}; "
              f"named source={gate.named_source_verdict}; sensitivity={gate.sensitivity_analysis_verdict}")
        if gate.ready_to_graduate:
            print("A pass on random data is the overfitting case these checks exist to catch.")


if __name__ == "__main__":
    main()

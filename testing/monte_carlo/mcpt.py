"""Sign-permutation Monte Carlo Permutation Test (MCPT), Masters/Aronson
variant -- asset-class-agnostic.

Takes nothing but a plain returns/P&L series (any unit, any cadence) plus an
explicit annualization factor. Each of `n_permutations` independent draws
flips every observation's sign with probability 0.5 and recomputes the
annualized Sharpe ratio -- this destroys any genuine directional edge while
preserving the magnitude distribution of the realized wins/losses, giving a
null distribution the observed Sharpe can be compared against.
`p_value = (1 + count(null >= observed)) / (1 + n_valid)`.

Originally ported byte-for-byte from Options and Futures Pipeline's
`block3_risk/mc_overfitting_mcpt.py` (see ADR 0011) independently into
`testing.engine.strategy_comparison.evaluate_strategy_pnl`, then factored out
here as the one standalone implementation once that duplication was found
(2026-07-11/12) -- this module doesn't need a trade-level equity curve to
infer its own annualization factor, so it works for a plain daily-returns
series (a factor portfolio, a prediction-market P&L stream) just as well as
irregularly-dated trade P&L. `evaluate_strategy_pnl` is now a thin wrapper
around `sign_permutation_mcpt` (its only real job is inferring
`periods_per_year` from an equity curve's date span via
`strategy_comparison.periods_per_year_from_trades`, then reshaping the
result into its own dict schema) -- the permutation loop, Sharpe formula,
and p-value convention live only here, not reimplemented per caller.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MCPTResult:
    """`p_value`/`null_distribution` are `None`/empty only when every
    permutation draw produced a degenerate (zero-variance or non-finite)
    Sharpe -- kept distinguishable from a real p-value of 0 or 1."""

    observed_sharpe: float
    p_value: float | None
    n_permutations: int
    null_distribution: np.ndarray


def annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    """Mean/std (ddof=1) of `returns`, annualized by `periods_per_year`.
    Returns `nan` for a zero-variance or empty input rather than raising or
    silently coercing to 0 -- callers decide what a degenerate Sharpe means
    for their use case.
    """
    if returns.size < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if std == 0 or not np.isfinite(std):
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sign_permutation_mcpt(
    returns: pd.Series | Sequence[float] | np.ndarray,
    periods_per_year: float = 252.0,
    n_permutations: int = 1000,
) -> MCPTResult | None:
    """Sign-permutation MCPT against a plain returns/P&L series.

    `returns` -- any 1-D numeric sequence: daily returns, per-trade dollar
    P&L, per-trade return-on-notional, settlement outcomes -- this function
    never inspects dates or units, only values. `periods_per_year` is the
    caller's own annualization factor (252 for daily returns; trades/year
    implied by a backtest's date span for irregularly-dated trade P&L --
    computing that span is the caller's job, exactly as
    `evaluate_strategy_pnl` already does for trade-level callers).

    Returns `None` for a degenerate input (fewer than 2 observations, or a
    zero-variance/non-finite observed Sharpe) -- distinguishes "no real
    signal to test" from a real result. Each permutation draw is seeded by
    its own index (`np.random.default_rng(i)`), so repeated calls on the
    same input are bit-for-bit reproducible.
    """
    values = np.asarray(list(returns) if not isinstance(returns, np.ndarray) else returns, dtype=float)

    observed = annualized_sharpe(values, periods_per_year)
    if not np.isfinite(observed):
        return None

    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        rng = np.random.default_rng(i)
        signs = rng.choice([1.0, -1.0], size=values.size)
        null[i] = annualized_sharpe(values * signs, periods_per_year)

    valid_null = null[np.isfinite(null)]
    if valid_null.size == 0:
        return MCPTResult(
            observed_sharpe=round(float(observed), 6),
            p_value=None,
            n_permutations=0,
            null_distribution=valid_null,
        )

    p_value = (1 + int(np.sum(valid_null >= observed))) / (1 + int(valid_null.size))

    return MCPTResult(
        observed_sharpe=round(float(observed), 6),
        p_value=round(float(p_value), 6),
        n_permutations=int(valid_null.size),
        null_distribution=np.round(valid_null, 6),
    )

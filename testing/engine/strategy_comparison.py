"""Reusable strategy-comparison engine: run any named set of backtest
variants, evaluate each via sign-permutation MCPT, and assemble a single
comparison JSON a dashboard (or a human) can read generically.

Deliberately split so the MCPT layer never has to know what produced the
trade P&L:

  - `evaluate_strategy_pnl` -- backtest-engine-agnostic. Takes a trade-level
    P&L series plus its already-computed equity curve (any strategy, any
    engine) and returns `{sharpe, mcpt: {p_value, n_permutations,
    null_distribution}}`. This is a straight port of the sign-permutation
    Masters/Aronson method already validated for this exact codebase's
    discrete, irregularly-dated trade P&L in
    `Options and Futures Pipeline/pipeline/block3_risk/mc_overfitting_mcpt.py`
    (same formulas, same p-value convention) -- not a new statistical method.

  - `run_multi_strategy_comparison` -- the strategy-agnostic comparison
    entry point: takes one named "runner" callable per strategy (each
    already doing its own strategy-specific selector/sizing, returning
    plain `(trade_pnl, equity_curve, meta)`), loads the shared chain once,
    and assembles one comparison output (`comparison_id`, `description`,
    `created_at`, `data_range`, `configs`, `benchmark`). `core/` itself
    never imports a strategy module here -- the runners are supplied by
    the caller, which is the only layer that imports across strategies.

  - `build_benchmark_equity_curve` -- honest buy-and-hold benchmark, built
    from the same synthetic-spot series `core/spot.py`'s
    `SyntheticSpotEngine` already derives from matched put/call quotes (via
    put-call parity) -- not a new price source.

Read-only w.r.t. every existing module it imports; this file only composes
them.

(This mirror omits an earlier, strategy-specific comparison helper this
module used to also carry -- generalized away by `run_multi_strategy_comparison`
already, and any remaining trace would have depended on an excluded
strategy package.)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from testing.engine.backtest_runner import CHAIN_COLUMNS
from testing.engine.chain_repository import OptionChainRepository
from testing.engine.spot import SyntheticSpotEngine
from testing.monte_carlo.mcpt import sign_permutation_mcpt


def periods_per_year_from_trades(n_trades: int, equity_curve: pd.Series) -> float:
    """Trades/year implied by an equity curve's own date span
    (`n_trades / (span_days / 365.25)`), rather than assuming 252 trading
    days/year -- these are irregularly-dated discrete trade-exit events, not
    daily bars. `equity_curve`'s index is expected to be the trades' exit
    dates (as `PortfolioAggregator.build_equity_curve` produces). Shared by
    `evaluate_strategy_pnl` and `testing.validation.walk_forward_validation`
    so both use one definition of this annualization factor.
    """
    dates = pd.DatetimeIndex(equity_curve.index)
    span_days = (dates.max() - dates.min()).days
    return n_trades / (span_days / 365.25) if span_days > 0 else float(n_trades)


def evaluate_strategy_pnl(
    name: str,
    trade_pnl: Sequence[float],
    equity_curve: pd.Series,
    n_permutations: int = 1000,
) -> dict:
    """Sign-permutation MCPT (Masters/Aronson variant) on a discrete
    trade-level P&L series, from any strategy/backtest engine -- the only
    inputs are the per-trade P&L values and the already-computed cumulative
    equity curve those trades produced. `name` is used only for error/debug
    context, not echoed into the return value.

    Thin, trade-level-specific glue around `testing.monte_carlo.mcpt.
    sign_permutation_mcpt`: this function's only real job is inferring the
    annualization factor from the equity curve's own date span (via
    `periods_per_year_from_trades`) and reshaping the result into this
    module's dict schema -- the permutation loop, Sharpe formula, and
    p-value convention live in exactly one place (`mcpt.py`), not
    reimplemented here.

    Returns `{"sharpe": float | None, "mcpt": {"p_value": float | None,
    "n_permutations": int, "null_distribution": list[float]}}`. `None`
    values mean a degenerate input (no trades, or zero-variance P&L) rather
    than a real Sharpe of 0 -- keeps "no signal" distinguishable from "no
    data" for any downstream reader.
    """
    pnl = np.asarray(list(trade_pnl), dtype=float)
    empty_result: dict = {"sharpe": None, "mcpt": {"p_value": None, "n_permutations": 0, "null_distribution": []}}
    if pnl.size == 0 or equity_curve is None or equity_curve.empty:
        return empty_result

    periods_per_year = periods_per_year_from_trades(pnl.size, equity_curve)
    result = sign_permutation_mcpt(pnl, periods_per_year, n_permutations)
    if result is None:
        return empty_result

    return {
        "sharpe": result.observed_sharpe,
        "mcpt": {
            "p_value": result.p_value,
            "n_permutations": result.n_permutations,
            "null_distribution": result.null_distribution.tolist(),
        },
    }


# --- Benchmark: honest buy-and-hold on this repo's existing spot proxy ------
def build_benchmark_equity_curve(
    chain: pd.DataFrame,
    notional: float = 100_000.0,
    target_dte: int = 30,
) -> pd.Series:
    """Buy-and-hold equity curve on the underlying, rebased to the same
    "cumulative dollar P&L from a fixed notional" units every strategy
    equity curve in this repo uses (`PortfolioAggregator.build_equity_curve`
    output) -- so it plots directly against a strategy curve without a unit
    conversion.

    Built from `core.spot.SyntheticSpotEngine.compute_spot_series`, the same
    put-call-parity-derived daily spot proxy other realized-vol inputs in
    this repo already use -- this repo has no separate underlying price feed
    to draw from (`forward_price` is unpopulated in the IvyDB extract), so
    reusing that proxy is the honest choice rather than inventing a second
    price source.
    """
    spot = SyntheticSpotEngine(target_dte=target_dte).compute_spot_series(chain)
    if spot.empty:
        return pd.Series(dtype=float)
    return (notional * (spot / spot.iloc[0] - 1.0)).rename("cumulative_value")


# --- Generalization: any strategy, one comparison ---------------------------
StrategyRunner = Callable[[pd.DataFrame], tuple[list[float], pd.Series, dict]]


def run_multi_strategy_comparison(
    runners: dict[str, StrategyRunner],
    secid: int,
    start_date: str,
    end_date: str,
    parquet_path: str = "data/all.parquet",
    include_benchmark: bool = False,
    benchmark_notional: float = 100_000.0,
    benchmark_target_dte: int = 30,
    n_permutations: int = 1000,
    comparison_id: str | None = None,
    description: str = "",
) -> dict:
    """Strategy-agnostic comparison entry point: takes one already
    strategy-specific "runner" callable per name -- `runner(chain) ->
    (trade_pnl, equity_curve, meta)`, where `trade_pnl` is the sized
    per-trade dollar P&L list, `equity_curve` is the
    `PortfolioAggregator.build_equity_curve`-shaped cumulative series, and
    `meta` is a free-form dict this function copies straight into the
    output's `"params"` key (typically `asdict(config)` plus an
    `"exit_reason_breakdown"` entry if the caller wants one shown).

    Loads the chain exactly once (all runners must share one `secid`) and
    hands it to every runner in turn -- this is why a runner takes a chain
    as its only argument rather than loading one itself.

    Produces one fixed output schema (`comparison_id`, `description`,
    `created_at`, `data_range`, `configs`, `benchmark`) regardless of how
    many or which strategies are being compared.

    Since the configs behind these runners are heterogeneous dataclasses
    with different fields (or no shared notion of "notional" at all,
    depending on the strategy's own sizing basis), `benchmark_notional`/
    `benchmark_target_dte` are explicit parameters here rather than read off
    any one config.
    """
    if not runners:
        raise ValueError("run_multi_strategy_comparison requires at least one named runner")

    repo = OptionChainRepository(parquet_path)
    chain = repo.get_chain(secid, start_date, end_date, columns=CHAIN_COLUMNS)

    configs_out: dict[str, dict] = {}
    for name, runner in runners.items():
        trade_pnl, equity, meta = runner(chain)
        trade_pnl = [float(p) for p in trade_pnl]

        evaluation = evaluate_strategy_pnl(name, trade_pnl, equity, n_permutations=n_permutations)

        configs_out[name] = {
            "params": meta.get("params", {}),
            "n_trades": len(trade_pnl),
            "exit_reason_breakdown": meta.get("exit_reason_breakdown", {}),
            "sharpe": evaluation["sharpe"],
            "equity_curve": [{"date": d.strftime("%Y-%m-%d"), "cumulative_value": float(v)} for d, v in equity.items()],
            "trade_pnl": trade_pnl,
            "mcpt": evaluation["mcpt"],
        }

    benchmark = None
    if include_benchmark:
        bench_equity = build_benchmark_equity_curve(chain, notional=benchmark_notional, target_dte=benchmark_target_dte)
        benchmark = {
            "name": (
                f"SPX buy-and-hold (synthetic spot via put-call parity, "
                f"rebased to ${benchmark_notional:,.0f} notional equivalent)"
            ),
            "equity_curve": [
                {"date": d.strftime("%Y-%m-%d"), "cumulative_value": float(v)} for d, v in bench_equity.items()
            ],
        }

    if comparison_id is None:
        comparison_id = "_vs_".join(runners.keys())

    return {
        "comparison_id": comparison_id,
        "description": description,
        "created_at": datetime.now(UTC).isoformat(),
        "data_range": {"start": start_date, "end": end_date},
        "configs": configs_out,
        "benchmark": benchmark,
    }

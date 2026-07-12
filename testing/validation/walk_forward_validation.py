"""Walk-forward validation (ADR 0005): the repo-wide default methodology for
any parameter tuning or train/test generalization claim.

**Naming disambiguation, read this first:** `testing.engine.backtest_runner.
walk_forward` is a *different* concept with the same name -- it's the
per-day trade-selection driver inside a single backtest run (walks forward
day-by-day deciding whether to open/close a position). This module's
"walk-forward" is the classic Pardo-style rolling train/test *evaluation*
methodology: slide a (train window, test window) pair forward across the
whole history, tune only on train, score only on test, repeat. Unrelated
mechanisms that happen to share a name -- a fold's train/test windows here
each still get backtested via `backtest_runner.walk_forward` internally,
through whatever `StrategyRunner` the caller supplies.

Reuses `testing.engine.strategy_comparison.StrategyRunner`'s existing shape
(`chain -> (trade_pnl, equity_curve, meta)`) and its exact Sharpe/
annualization convention (trades/year implied by the window's own date
span, not a hardcoded guess) -- this module is a new evaluation loop around
already-existing, already-verified building blocks, not a new engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from testing.engine.strategy_comparison import StrategyRunner, periods_per_year_from_trades
from testing.monte_carlo.mcpt import annualized_sharpe


def _annualized_sharpe_from_trades(trade_pnl: list[float], equity_curve: pd.Series) -> tuple[float, int]:
    """Same annualization convention as `strategy_comparison.
    evaluate_strategy_pnl` (`periods_per_year_from_trades`), and the same
    `annualized_sharpe` (`mcpt.py`) every other Sharpe consumer in this repo
    uses. Returns `(nan, 0)` for degenerate input (no trades, single trade,
    zero-variance P&L, zero-span dates) -- never a fabricated 0.
    """
    pnl = np.asarray(trade_pnl, dtype=float)
    if pnl.size == 0 or equity_curve is None or equity_curve.empty:
        return float("nan"), 0
    periods_per_year = periods_per_year_from_trades(pnl.size, equity_curve)
    return annualized_sharpe(pnl, periods_per_year), int(pnl.size)


@dataclass(frozen=True)
class WalkForwardWindow:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_walk_forward_windows(
    data_start: pd.Timestamp | str,
    data_end: pd.Timestamp | str,
    *,
    train_years: float = 4.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
) -> list[WalkForwardWindow]:
    """Rolling (not anchored/expanding) windows: each fold's train window is
    the same fixed length and slides forward by `step_years`; test windows
    are contiguous and non-overlapping when `step_years == test_years` (the
    default and the standard rolling walk-forward design) -- together they
    tile the entire out-of-sample history exactly once, so `mean_out_of_sample_sharpe`
    isn't double-counting any period. A `step_years` smaller than
    `test_years` would overlap test windows between folds; this function
    allows it (the caller's choice) but callers should know that changes
    what the aggregate OOS metrics mean.

    Returns an empty list if `data_end - data_start` is shorter than one
    full `train_years + test_years` window -- never raises, since "not
    enough data for even one fold" is itself a real, reportable finding
    about a young/short dataset, not necessarily a caller error.
    """
    start = pd.Timestamp(data_start)
    end = pd.Timestamp(data_end)

    windows = []
    fold = 0
    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        windows.append(
            WalkForwardWindow(
                fold=fold, train_start=train_start, train_end=train_end, test_start=train_end, test_end=test_end
            )
        )
        fold += 1
        train_start = train_start + pd.DateOffset(years=step_years)
    return windows


@dataclass(frozen=True)
class WalkForwardFoldResult:
    window: WalkForwardWindow
    selected_candidate: str
    in_sample_sharpe: float
    in_sample_n_trades: int
    out_of_sample_sharpe: float
    out_of_sample_n_trades: int

    @property
    def walk_forward_efficiency(self) -> float | None:
        """`out_of_sample_sharpe / in_sample_sharpe`, but **only when
        in-sample Sharpe is positive and finite, and out-of-sample Sharpe is
        finite** (per ADR 0005). A ratio of two negative Sharpes can look
        reassuring by sign cancellation alone (e.g. IS=-0.5, OOS=-0.3 ->
        ratio=0.6) while both periods are actually losing -- reporting
        `None` here, not a misleading ratio, matches this repo's existing
        convention (`MCPTResult.p_value`, `PerformanceAnalyzer`'s NaN
        fields) for a degenerate/inapplicable result.
        """
        if not np.isfinite(self.in_sample_sharpe) or self.in_sample_sharpe <= 0:
            return None
        if not np.isfinite(self.out_of_sample_sharpe):
            return None
        return self.out_of_sample_sharpe / self.in_sample_sharpe


@dataclass(frozen=True)
class WalkForwardReport:
    folds: tuple[WalkForwardFoldResult, ...]

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    @property
    def mean_out_of_sample_sharpe(self) -> float:
        """Mean OOS Sharpe across folds with a finite value. `nan` if no
        fold has one (e.g. every fold was degenerate) -- never silently 0.
        """
        values = [f.out_of_sample_sharpe for f in self.folds if np.isfinite(f.out_of_sample_sharpe)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def out_of_sample_positive_fraction(self) -> float:
        """Fraction of folds where OOS Sharpe is finite and > 0 -- the
        primary consistency signal when `mean_walk_forward_efficiency` is
        `None` because in-sample was never positive (a losing strategy
        doesn't get a WFE ratio, but "does it ever make money OOS" is still
        answerable). `nan` if there are no folds at all.
        """
        if not self.folds:
            return float("nan")
        positive = sum(1 for f in self.folds if np.isfinite(f.out_of_sample_sharpe) and f.out_of_sample_sharpe > 0)
        return positive / len(self.folds)

    @property
    def mean_walk_forward_efficiency(self) -> float | None:
        """Mean of `walk_forward_efficiency` over folds where it's defined
        (in-sample was positive). `None` if no fold ever had a positive
        in-sample Sharpe to evaluate generalization *of a apparent edge*
        against -- distinct from `out_of_sample_positive_fraction`, which
        stays meaningful even then.
        """
        values = [wfe for f in self.folds if (wfe := f.walk_forward_efficiency) is not None]
        return float(np.mean(values)) if values else None

    def summary(self) -> str:
        wfe = self.mean_walk_forward_efficiency
        lines = [
            f"Folds: {self.n_folds}",
            f"Mean out-of-sample Sharpe: {self.mean_out_of_sample_sharpe:.4f}",
            f"Out-of-sample positive fraction: {self.out_of_sample_positive_fraction:.2%}",
            f"Mean walk-forward efficiency: {wfe:.4f}" if wfe is not None else "Mean walk-forward efficiency: not defined (in-sample never positive)",
        ]
        for f in self.folds:
            w = f.window
            wfe_str = f"{f.walk_forward_efficiency:.4f}" if f.walk_forward_efficiency is not None else "n/a"
            lines.append(
                f"  Fold {w.fold}: train {w.train_start.date()}-{w.train_end.date()} "
                f"({f.selected_candidate}, IS sharpe={f.in_sample_sharpe:.4f}, n={f.in_sample_n_trades}) -> "
                f"test {w.test_start.date()}-{w.test_end.date()} "
                f"(OOS sharpe={f.out_of_sample_sharpe:.4f}, n={f.out_of_sample_n_trades}, WFE={wfe_str})"
            )
        return "\n".join(lines)


def _slice_chain(chain: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return chain[(chain["date"] >= start) & (chain["date"] < end)]


def run_walk_forward_validation(
    chain: pd.DataFrame,
    candidates: dict[str, StrategyRunner],
    *,
    train_years: float = 4.0,
    test_years: float = 1.0,
    step_years: float = 1.0,
) -> WalkForwardReport:
    """For each rolling fold: run every candidate on the train-window chain
    slice only, select whichever has the highest in-sample Sharpe (ties
    broken by dict insertion order), then run *only that selected candidate*
    on the test-window slice -- the test window never influences which
    candidate is picked, only how the already-picked one is scored.

    A candidate whose in-sample Sharpe is `nan` in a given fold (e.g. zero
    trades in that window) is never selected as the winner unless every
    candidate is degenerate in that fold, in which case the first
    (insertion-order) candidate is selected and both IS/OOS Sharpe are
    `nan` for that fold -- a real, reportable "no candidate had enough data
    this fold" finding, not an error.

    Raises `ValueError` if `candidates` is empty, or if `chain` doesn't span
    at least one full `train_years + test_years` window (see
    `generate_walk_forward_windows`'s own "empty is a valid, reportable
    answer" convention -- this function raises instead, at this layer,
    since running zero folds and calling it "validated" would be a much
    worse silent failure than a raised error the caller has to notice).

    Known, accepted limitation: chain slicing is a clean date cutoff, not a
    lookback-buffered one -- any signal a candidate computes on a rolling
    window (e.g. `vrp_harvester`'s realized-vol lookback) will have reduced
    data right at the start of each fold's train/test window, same
    warm-up effect the full-sample backtest already has once at its very
    start, just repeated at each fold boundary instead of only the first.
    """
    if not candidates:
        raise ValueError("run_walk_forward_validation requires at least one candidate")

    dates = pd.DatetimeIndex(chain["date"])
    windows = generate_walk_forward_windows(
        dates.min(), dates.max(), train_years=train_years, test_years=test_years, step_years=step_years
    )
    if not windows:
        raise ValueError(
            f"chain spans {dates.min().date()} to {dates.max().date()}, too short for one "
            f"{train_years}-year train + {test_years}-year test window"
        )

    fold_results = []
    for window in windows:
        train_chain = _slice_chain(chain, window.train_start, window.train_end)
        in_sample_scores: dict[str, tuple[float, int]] = {
            name: _annualized_sharpe_from_trades(*runner(train_chain)[:2]) for name, runner in candidates.items()
        }

        finite_scores = {name: sharpe for name, (sharpe, _n) in in_sample_scores.items() if np.isfinite(sharpe)}
        if finite_scores:
            selected = max(finite_scores, key=lambda name: finite_scores[name])
        else:
            selected = next(iter(candidates))

        in_sample_sharpe, in_sample_n = in_sample_scores[selected]

        test_chain = _slice_chain(chain, window.test_start, window.test_end)
        oos_trade_pnl, oos_equity, _meta = candidates[selected](test_chain)
        out_of_sample_sharpe, out_of_sample_n = _annualized_sharpe_from_trades(oos_trade_pnl, oos_equity)

        fold_results.append(
            WalkForwardFoldResult(
                window=window,
                selected_candidate=selected,
                in_sample_sharpe=in_sample_sharpe,
                in_sample_n_trades=in_sample_n,
                out_of_sample_sharpe=out_of_sample_sharpe,
                out_of_sample_n_trades=out_of_sample_n,
            )
        )

    return WalkForwardReport(folds=tuple(fold_results))

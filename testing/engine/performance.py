from dataclasses import dataclass

import numpy as np
import pandas as pd

from testing.engine.portfolio import PortfolioAggregator
from testing.engine.sizing import SizedTradeResult
from testing.monte_carlo.mcpt import annualized_sharpe


def drawdown_series(equity_curve: pd.Series, fixed_notional: float) -> pd.Series:
    """Running drawdown (as a fraction of `fixed_notional`) off a cumulative
    equity curve. Shared by `PerformanceAnalyzer.analyze` (which takes
    `.min()` for the scalar `max_drawdown_pct`) and `testing.viz.equity_curve`
    (which plots the full series) -- one definition, not two independently
    maintained copies of the same two-line formula.
    """
    if equity_curve.empty:
        return pd.Series(dtype=float)
    running_max = equity_curve.cummax()
    return ((equity_curve - running_max) / fixed_notional).rename("drawdown_pct")


@dataclass(frozen=True)
class PerformanceReport:
    num_trades: int
    win_rate: float
    avg_pnl_dollars: float
    sharpe: float
    max_drawdown_pct: float
    calmar: float


class PerformanceAnalyzer:
    """Computes Sharpe/Calmar/max-drawdown/win-rate/avg-P&L off sized trade
    results and the cumulative equity curve.

    Sharpe is computed at the trade level (mean/std of each trade's P&L as a
    fraction of the fixed notional, annualized by actual trades-per-year)
    rather than from daily-diffed equity, since the equity curve here only
    moves on exit dates -- daily-diffing it would pad in artificial zero-return
    days between trades and distort the ratio.
    """

    def analyze(
        self,
        sized_trades: list[SizedTradeResult],
        equity_curve: pd.Series,
        fixed_notional: float,
    ) -> PerformanceReport:
        if not sized_trades:
            return PerformanceReport(0, 0.0, 0.0, 0.0, 0.0, 0.0)

        pnl_dollars = np.array([t.pnl_dollars for t in sized_trades])
        trade_returns = pnl_dollars / fixed_notional

        num_trades = len(sized_trades)
        win_rate = float((pnl_dollars > 0).mean())
        avg_pnl_dollars = float(pnl_dollars.mean())

        entry_dates = [t.trade_result.entry_date for t in sized_trades]
        exit_dates = [t.trade_result.exit_date for t in sized_trades]
        years_spanned = max((max(exit_dates) - min(entry_dates)).days / 365.25, 1e-9)
        trades_per_year = num_trades / years_spanned

        # annualized_sharpe returns NaN for a single trade or zero variance --
        # coerced to 0.0 here (this report's own convention, unlike mcpt's,
        # since PerformanceReport.sharpe is a plain float field callers print
        # directly, e.g. a thin analyze_by_regime bucket).
        raw_sharpe = annualized_sharpe(trade_returns, trades_per_year)
        sharpe = raw_sharpe if np.isfinite(raw_sharpe) else 0.0

        drawdown = drawdown_series(equity_curve, fixed_notional)
        max_drawdown_pct = float(drawdown.min()) if not drawdown.empty else 0.0

        total_return_pct = float(equity_curve.iloc[-1] / fixed_notional) if not equity_curve.empty else 0.0
        annualized_return_pct = total_return_pct / years_spanned
        calmar = float(annualized_return_pct / abs(max_drawdown_pct)) if max_drawdown_pct < 0 else 0.0

        return PerformanceReport(
            num_trades=num_trades,
            win_rate=win_rate,
            avg_pnl_dollars=avg_pnl_dollars,
            sharpe=sharpe,
            max_drawdown_pct=max_drawdown_pct,
            calmar=calmar,
        )

    def analyze_by_regime(
        self,
        sized_trades: list[SizedTradeResult],
        regime_by_date: dict[pd.Timestamp, str],
        fixed_notional: float,
    ) -> dict[str | None, PerformanceReport]:
        """Buckets trades by whatever regime label is in effect at each
        trade's entry date, then runs the same analysis per bucket. Works
        with any date->label mapping -- a 3-bucket VIX threshold, an HMM
        regime classifier, anything -- the bucketing logic itself doesn't
        care what the labels mean.
        """
        groups: dict[str | None, list[SizedTradeResult]] = {}
        for sized in sized_trades:
            label = regime_by_date.get(sized.trade_result.entry_date)
            groups.setdefault(label, []).append(sized)

        aggregator = PortfolioAggregator()
        return {
            label: self.analyze(group, aggregator.build_equity_curve(group), fixed_notional)
            for label, group in groups.items()
        }

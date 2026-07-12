
import pandas as pd

from testing.engine.sizing import SizedTradeResult


class PortfolioAggregator:
    """Rolls sized trade-level P&L into a single daily cumulative equity curve.

    Each trade's P&L is realized on its exit date; trades that exit on the
    same day (e.g. two concurrently open structures, issue #4) are summed
    rather than overwritten, which is how overlap is handled in v1.
    """

    def build_equity_curve(
        self,
        sized_trades: list[SizedTradeResult],
        date_range: pd.DatetimeIndex | None = None,
    ) -> pd.Series:
        if not sized_trades:
            return pd.Series(dtype=float)

        pnl_by_exit_date: dict = {}
        for sized in sized_trades:
            exit_date = sized.trade_result.exit_date
            pnl_by_exit_date[exit_date] = pnl_by_exit_date.get(exit_date, 0.0) + sized.pnl_dollars

        pnl_series = pd.Series(pnl_by_exit_date).sort_index()
        if date_range is not None:
            pnl_series = pnl_series.reindex(date_range, fill_value=0.0)

        return pnl_series.cumsum().rename("equity")

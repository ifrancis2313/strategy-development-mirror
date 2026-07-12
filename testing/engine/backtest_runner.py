
import pandas as pd

from testing.engine.execution_config import ExecutionConfig
from testing.engine.performance import PerformanceAnalyzer
from testing.engine.portfolio import PortfolioAggregator
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import PositionSimulator, TradeResult
from testing.engine.sizing import SizingEngine

CHAIN_COLUMNS = [
    "secid", "date", "exdate", "cp_flag", "strike_price",
    "best_bid", "best_offer", "delta", "impl_volatility", "optionid",
]


def walk_forward(
    chain: pd.DataFrame,
    config: ExecutionConfig,
    selector,
    **selector_kwargs,
) -> list[tuple[PositionSpec, TradeResult]]:
    """Walks every DTE-eligible day in `chain` forward, opening a new trade
    whenever nothing is open, simulating it to exit before considering the
    next candidate day.

    `selector` is duck-typed: anything with a
    `.select_trade(chain_snapshot, entry_date, has_open_position, **kwargs)
    -> Optional[PositionSpec]` method. `**selector_kwargs` are forwarded to
    it on every call, so this driver doesn't need to know anything about a
    particular strategy's signal. Any kwarg value that is itself a
    `pd.Series` (e.g. a per-day VRP value keyed by date) is looked up at
    `entry_date` via `.get()` before being forwarded -- everything else is
    passed through unchanged. This generalizes the old hardcoded
    `vrp.get(entry_date) -> vrp_value` lookup to any per-day signal a future
    strategy's selector might need, without this driver hardcoding "vrp".

    `chain` is expected to already be filtered to a single secid and to
    cover the full backtest window in memory -- the caller loads it once via
    OptionChainRepository and reuses it across as many configs/scripts as
    needed, rather than re-querying per trade.
    """
    simulator = PositionSimulator()

    dte = (chain["exdate"] - chain["date"]).dt.days
    candidates = sorted(chain.loc[(dte >= config.min_dte) & (dte <= config.max_dte), "date"].unique())

    trades: list[tuple[PositionSpec, TradeResult]] = []
    open_until = None
    for entry_date in candidates:
        if open_until is not None and entry_date <= open_until:
            continue

        day_chain = chain[chain["date"] == entry_date]
        resolved_kwargs = {
            key: (value.get(entry_date) if isinstance(value, pd.Series) else value)
            for key, value in selector_kwargs.items()
        }
        trade = selector.select_trade(day_chain, entry_date, has_open_position=False, **resolved_kwargs)
        if trade is None:
            continue

        optionids = [leg.optionid for leg in trade.legs]
        leg_history = chain[
            (chain["date"] >= trade.entry_date) & (chain["date"] <= trade.exdate)
            & (chain["optionid"].isin(optionids))
        ][["date", "optionid", "best_bid", "best_offer"]]

        result = simulator.simulate(trade, leg_history, config)
        trades.append((trade, result))
        open_until = result.exit_date

    return trades


def summarize(label: str, trades: list[tuple[PositionSpec, TradeResult]], config: ExecutionConfig) -> None:
    """Sizes, aggregates, and prints a performance report for one walk_forward() run."""
    if not trades:
        print(f"{label}: no trades generated")
        return
    sized = SizingEngine().size_fixed_notional(trades, config)
    equity = PortfolioAggregator().build_equity_curve(sized)
    perf = PerformanceAnalyzer().analyze(sized, equity, config.fixed_notional)
    print(f"\n--- {label} ---")
    print(f"Trades:        {perf.num_trades}")
    print(f"Win rate:      {perf.win_rate:.1%}")
    print(f"Avg P&L/trade: ${perf.avg_pnl_dollars:,.2f}")
    print(f"Sharpe:        {perf.sharpe:.2f}")
    print(f"Max drawdown:  {perf.max_drawdown_pct:.1%}")
    print(f"Calmar:        {perf.calmar:.2f}")
    print(f"Final equity:  ${equity.iloc[-1]:,.2f}")

    multi_leg = [(spec, res) for spec, res in trades if spec.is_iron_condor]
    if multi_leg:
        put_pnl_total = sum(res.put_pnl for _, res in multi_leg if res.put_pnl is not None)
        call_pnl_total = sum(res.call_pnl for _, res in multi_leg if res.call_pnl is not None)
        combined = put_pnl_total + call_pnl_total
        put_share = put_pnl_total / combined if combined else float("nan")
        print(f"  Put-side P&L (premium units):  {put_pnl_total:,.2f}  ({put_share:.1%} of combined)")
        print(f"  Call-side P&L (premium units): {call_pnl_total:,.2f}  ({1 - put_share:.1%} of combined)")

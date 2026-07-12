"""Options Greek exposure metrics -- reports (never derives) delta/gamma/vega/
theta exposure at the position and portfolio level, over an options
strategy's backtest time series (wired here against the ported
`vrp_harvester`, but generic to any strategy built on `PositionSpec`/`Leg`).

Greeks are already present natively in the underlying OptionMetrics/
Tastytrade data (`delta`/`gamma`/`vega`/`theta` columns on every chain row)
-- this module only signs, scales, and sums those already-computed values
across legs and across time. No Black-Scholes derivation lives here; per the
issue spec, that's explicitly out of scope.

Independent of every other asset-class metrics module (`testing.metrics.
factor`, `testing.metrics.prediction_market`, if/when they exist) -- this
module only imports from `testing.engine`, so it can be added or removed
without touching anything else under `testing/metrics/`.
"""

import pandas as pd

from testing.engine.leg import Leg
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import TradeResult
from testing.engine.sizing import SizedTradeResult

#: The four Greeks this module reports, in the order OptionMetrics/Tastytrade
#: both carry them -- shared by every function below so column order never
#: drifts between position- and portfolio-level output.
GREEK_COLUMNS = ["delta", "gamma", "vega", "theta"]

#: Columns a chain slice needs for `greeks_history_from_chain` to work --
#: the identity columns (`date`, `optionid`) plus the four Greeks.
GREEKS_HISTORY_COLUMNS = ["date", "optionid", *GREEK_COLUMNS]


def _leg_sign(leg: Leg) -> float:
    """A short leg's exposure is the *negative* of the option's own Greek --
    you're on the other side of that contract -- while a long leg's is the
    Greek unchanged: long=+1, short=-1.

    Deliberately the OPPOSITE polarity from `position_simulator.net_credit`
    and `payoff.max_loss_at_expiration`, which both use short=+1/long=-1 --
    those answer "how much cash did I collect" (selling collects a credit,
    buying pays a debit), this answers "how much market exposure do I
    carry" (owning a call means you're long its delta; selling one means
    you're short it). Two different questions, correctly opposite signs --
    not an inconsistency to reconcile.
    """
    return 1.0 if leg.side == "long" else -1.0


def position_greek_exposure(
    trade: PositionSpec,
    exit_date: pd.Timestamp,
    greeks_history: pd.DataFrame,
    contracts: int = 1,
    contract_multiplier: int = 100,
) -> pd.DataFrame:
    """Daily Greek exposure for a single (possibly multi-leg) position, from
    its entry date through `exit_date` inclusive, aggregated across every leg.

    `greeks_history` is a pre-fetched slice (one row per date per optionid,
    columns `date`/`optionid`/`GREEK_COLUMNS`) covering at least this trade's
    legs across its holding period -- the same pre-fetched-slice contract
    `PositionSimulator.simulate` uses for `leg_history`, so this module stays
    fully testable against synthetic data with no Parquet dependency and
    never re-queries a repository itself.

    Returns a date-indexed DataFrame (one row per date any leg has data,
    columns `GREEK_COLUMNS`), each leg's raw per-share Greek scaled by
    side sign x `contracts` x `contract_multiplier` and summed across legs --
    the standard "share-equivalent portfolio Greek" convention (e.g. position
    delta = sum of each leg's signed delta x shares-per-contract x contracts
    held).
    """
    exit_date = pd.Timestamp(exit_date)
    optionids = [leg.optionid for leg in trade.legs]
    mask = (
        greeks_history["optionid"].isin(optionids)
        & (greeks_history["date"] >= trade.entry_date)
        & (greeks_history["date"] <= exit_date)
    )
    relevant = greeks_history.loc[mask, ["date", "optionid", *GREEK_COLUMNS]].copy()
    if relevant.empty:
        return pd.DataFrame(columns=GREEK_COLUMNS, dtype=float).rename_axis("date")

    sign_by_optionid = {leg.optionid: _leg_sign(leg) for leg in trade.legs}
    weight = relevant["optionid"].map(sign_by_optionid) * (contracts * contract_multiplier)
    for col in GREEK_COLUMNS:
        relevant[col] = relevant[col] * weight

    return relevant.groupby("date")[GREEK_COLUMNS].sum().sort_index()


def portfolio_greek_exposure(
    trades: list[tuple[PositionSpec, TradeResult]],
    sized_trades: list[SizedTradeResult],
    greeks_history: pd.DataFrame,
    contract_multiplier: int = 100,
    date_range: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Time series of portfolio-level Greek exposure -- the Greek analogue of
    `PortfolioAggregator.build_equity_curve`: sums every position's daily
    exposure across whatever positions are open on a given date (unlike
    P&L, which only realizes on an exit date, Greek exposure is live every
    day a position is held -- so overlap is summed across the *entire* open
    window, not just a shared exit day).

    `trades` and `sized_trades` must be the same backtest's outputs, aligned
    index-for-index -- exactly how `backtest_runner.summarize` already pairs
    `walk_forward`'s trades with `SizingEngine`'s sized results; contracts
    held is a sizing concern this module doesn't recompute.
    """
    if len(trades) != len(sized_trades):
        raise ValueError("trades and sized_trades must be the same length, index-aligned")

    per_position = [
        position_greek_exposure(spec, result.exit_date, greeks_history, sized.contracts, contract_multiplier)
        for (spec, result), sized in zip(trades, sized_trades, strict=True)
    ]
    non_empty = [df for df in per_position if not df.empty]
    if non_empty:
        portfolio = pd.concat(non_empty).groupby(level=0)[GREEK_COLUMNS].sum().sort_index()
    else:
        portfolio = pd.DataFrame(columns=GREEK_COLUMNS, dtype=float)
    portfolio = portfolio.rename_axis("date")

    if date_range is not None:
        portfolio = portfolio.reindex(date_range, fill_value=0.0)
    return portfolio


def greeks_history_from_chain(
    chain: pd.DataFrame,
    trades: list[tuple[PositionSpec, TradeResult]],
) -> pd.DataFrame:
    """Slices a full chain DataFrame (must carry `GREEK_COLUMNS` alongside the
    usual `date`/`optionid`/... columns -- OptionMetrics and Tastytrade both
    do) down to just the optionids and date range `trades`' legs actually
    span, mirroring how `backtest_runner.walk_forward` slices `leg_history`
    for `best_bid`/`best_offer` but keeping the Greek columns instead of the
    quote columns. Convenience for wiring a real backtest's already-loaded
    `chain` into `portfolio_greek_exposure` without every caller re-deriving
    this filter.
    """
    if not trades:
        return pd.DataFrame(columns=GREEKS_HISTORY_COLUMNS)

    optionids: set = set()
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    for spec, result in trades:
        optionids.update(leg.optionid for leg in spec.legs)
        start = spec.entry_date if start is None else min(start, spec.entry_date)
        end = result.exit_date if end is None else max(end, result.exit_date)

    mask = chain["optionid"].isin(optionids) & (chain["date"] >= start) & (chain["date"] <= end)
    return chain.loc[mask, GREEKS_HISTORY_COLUMNS]

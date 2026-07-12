from dataclasses import dataclass

import pandas as pd

from testing.engine.execution_config import ExecutionConfig
from testing.engine.payoff import max_loss_at_expiration
from testing.engine.position import PositionSpec


@dataclass(frozen=True)
class TradeResult:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    exit_reason: str  # "profit_target" | "hard_stop" | "dte_stop" | "data_exhausted"
    entry_credit: float
    exit_debit: float
    pnl: float
    put_pnl: float | None = None
    call_pnl: float | None = None


def fill_price(bid: float, offer: float, action: str, fill_mode) -> float:
    """`action` is "sell" (receive bid) or "buy" (pay offer) under the
    realistic "bid_ask" mode (fill_mode="bid_ask", equivalent to a spread
    capture of 0.0). `fill_mode="mid"` is a spread capture of 1.0 -- both
    legs fill exactly at the midpoint. `fill_mode` also accepts a numeric
    capture fraction in [0, 1] directly, for modeling partial price
    improvement from working a limit order inside the spread rather than
    crossing it outright; 0.0 and 1.0 are identical to "bid_ask"/"mid".
    Mid (and any capture > 0) is optimistic relative to a market order and
    only meant for quantifying slippage (issue #8), never for a result
    meant to be trusted as the realistic baseline.

    Public (not module-private) because `core.broker.order_manager` reuses
    it for live limit-price construction -- the same capture-fraction math
    that produced the issue #8 follow-up's 60-70% break-even finding, not a
    separate pricing model for live orders.
    """
    if fill_mode == "bid_ask":
        capture = 0.0
    elif fill_mode == "mid":
        capture = 1.0
    else:
        capture = float(fill_mode)

    mid = (bid + offer) / 2
    if action == "sell":
        return bid + capture * (mid - bid)
    return offer - capture * (offer - mid)


def check_exit(
    trade: PositionSpec,
    current_date,
    cost_to_close: float,
    entry_credit: float,
    dte_remaining: int,
    put_pnl: float | None,
    call_pnl: float | None,
    config: ExecutionConfig,
) -> TradeResult | None:
    """Checks the three exit conditions in precedence order -- profit target,
    hard stop, DTE stop -- against a single day's already-computed
    cost_to_close. Returns None if nothing fires yet.

    Shared by `PositionSimulator.simulate()`'s historical walk-forward loop
    and the live daily-check script's single-day mark-to-market, so exit
    precedence can't drift between backtest and live (the same risk ADR
    0004 flagged for combined-leg exits).

    Compares against `abs(entry_credit)`, not the raw signed value -- a
    same-strike calendar spread (sell near, buy far) is a net *debit*
    (`entry_credit` negative), and the raw-signed comparison silently
    inverts for a negative value: traced through with concrete numbers
    during the calendar spread's design and confirmed the un-abs'd version
    would fire "profit_target" near-immediately (even at a loss) and
    "hard_stop" while genuinely profitable. `abs()` makes both checks mean
    the same thing regardless of credit/debit sign: "X% of whatever was
    paid or received." That basis is itself configurable (ADR 0011,
    `config.exit_trigger_basis`) -- default "entry_credit" keeps exactly
    this `abs(entry_credit)` behavior; "max_loss" swaps in
    `max_loss_at_expiration(trade.legs, entry_credit)` instead, for a
    strategy whose entry credit can collapse toward 0 (or go net debit)
    while its actual worst-case risk stays large -- see that config field's
    own docstring.
    """
    profit_so_far = entry_credit - cost_to_close

    if config.exit_trigger_basis == "max_loss":
        trigger_basis = max_loss_at_expiration(trade.legs, entry_credit)
    else:
        trigger_basis = abs(entry_credit)

    if profit_so_far >= config.profit_target_pct * trigger_basis:
        return TradeResult(
            trade.entry_date, current_date, "profit_target", entry_credit, cost_to_close,
            profit_so_far, put_pnl, call_pnl,
        )

    loss = cost_to_close - entry_credit
    if loss >= config.stop_loss_multiple * trigger_basis:
        return TradeResult(
            trade.entry_date, current_date, "hard_stop", entry_credit, cost_to_close,
            -loss, put_pnl, call_pnl,
        )

    if dte_remaining <= config.exit_dte:
        return TradeResult(
            trade.entry_date, current_date, "dte_stop", entry_credit, cost_to_close,
            profit_so_far, put_pnl, call_pnl,
        )

    return None


def net_credit(legs, price_fn) -> float:
    """Sums signed, weighted credit across a set of legs sharing one `right`
    (e.g. just the put legs, or just the call legs) -- short legs contribute
    +price (sold), long legs contribute -price (bought), each leg's
    contribution scaled by `leg.weight` (1.0 for every existing strategy, so
    this changes nothing for them; `variance_swap`'s strike strip is the
    first caller with non-uniform weights).

    Public/free (not a `PositionSimulator` method) because
    `core.broker.order_manager` reuses the exact same short/long sign
    convention to price a live combo order -- the entry-credit and
    cost-to-close math shouldn't be re-derived independently for live
    orders.
    """
    credit = 0.0
    for leg in legs:
        signed_price = price_fn(leg) if leg.side == "short" else -price_fn(leg)
        credit += signed_price * leg.weight
    return credit


def mark_to_market(trade: PositionSpec, quotes: dict, entry_credit_by_right: dict, fill_mode) -> tuple:
    """Cost-to-close and put/call P&L attribution from current bid/ask
    quotes per leg (`{optionid: (bid, offer)}`) -- the live-quote analogue
    of `PositionSimulator._pnl_at`, which instead reads from a pre-fetched,
    date-indexed `leg_history`. Same close-side fill convention (short legs
    bought back at offer, long legs sold at bid) and the same put/call
    partial-sum grouping via `net_credit`, so the live daily-check script's
    notion of "what would it cost to close this right now" can't drift from
    the backtest's.
    """
    def close_price(leg):
        bid, offer = quotes[leg.optionid]
        action = "buy" if leg.side == "short" else "sell"
        return fill_price(bid, offer, action, fill_mode)

    put_legs, call_legs = trade.put_legs, trade.call_legs
    put_cost = net_credit(put_legs, close_price) if put_legs else None
    call_cost = net_credit(call_legs, close_price) if call_legs else None
    cost_to_close = (put_cost or 0.0) + (call_cost or 0.0)

    # Same attribution form as PositionSimulator._pnl_at -- a right's stored
    # entry value is non-None whenever that right has legs (so put_cost/call_cost
    # being non-None is the only gate needed), keeping live and backtest aligned.
    put_pnl = entry_credit_by_right["P"] - put_cost if put_cost is not None else None
    call_pnl = entry_credit_by_right["C"] - call_cost if call_cost is not None else None

    return cost_to_close, put_pnl, call_pnl


def entry_value_by_right(spec: PositionSpec, capture: str | float) -> dict:
    """Per-right (P/C) net entry value estimate at a given spread-capture
    fraction, from each leg's currently-known bid/offer. Shared glue between
    any strategy's live daily-check script and `record_entry`'s expected
    `entry_credit_by_right` shape -- promoted out of the VRP harvester's own
    script (originally `_entry_credit_estimate`) once the calendar spread
    strategy needed the exact same thing; nothing about it is VRP-specific.

    Positive = net credit collected (a vertical spread); negative = net
    debit paid (a calendar spread) -- the sign is whatever the legs'
    `side`/bid/offer actually produce, not assumed.
    """
    def entry_price(leg):
        action = "sell" if leg.side == "short" else "buy"
        return fill_price(leg.best_bid, leg.best_offer, action, capture)

    return {
        "P": net_credit(spec.put_legs, entry_price) if spec.put_legs else None,
        "C": net_credit(spec.call_legs, entry_price) if spec.call_legs else None,
    }


class PositionSimulator:
    """Marks a position to market day by day and closes it at whichever exit
    condition fires first: 50% of max profit captured, the 21-DTE stop, or
    the hard-stop loss multiple -- applied to the *combined* position across
    however many legs are present (a put spread, an iron condor's put spread
    + call spread, or any other leg count), not per leg.

    Takes a pre-fetched `leg_history` (one row per date per optionid, covering
    every leg across the trade's life) rather than querying the repository
    itself -- keeps this module fully testable against synthetic data, with no
    Parquet dependency.

    `config.fill_mode` selects "bid_ask" (default; sell at bid, buy at ask)
    or "mid" (both legs at midpoint, for slippage comparison only).
    """

    def _leg_histories(self, trade: PositionSpec, leg_history: pd.DataFrame):
        return {
            leg.optionid: leg_history[leg_history["optionid"] == leg.optionid].set_index("date")
            for leg in trade.legs
        }

    def _entry_credit_by_right(self, trade: PositionSpec, fill_mode: str) -> dict:
        # Identical math to the module-level entry_value_by_right (fill_mode and
        # the function's `capture` arg both flow to fill_price the same way) --
        # delegate rather than keep a second copy of the per-right net-credit calc.
        return entry_value_by_right(trade, fill_mode)

    def _pnl_at(self, trade: PositionSpec, hists: dict, current_date, entry_credit_by_right: dict, fill_mode: str) -> tuple:
        """Returns (cost_to_close, put_pnl, call_pnl). To close, short legs
        are bought back (pay offer-side fill) and long legs are sold (receive
        bid-side fill) -- the reverse of how each leg was opened. put_pnl/
        call_pnl are entry-credit-attributed-by-right minus cost-to-close
        summed over that right's legs, generalizing the old hardcoded
        put/call breakdown to "however many legs of each right are present."
        Costs are summed per-right then combined (put_cost + call_cost),
        matching the original arithmetic grouping exactly.
        """
        def close_price(leg):
            row = hists[leg.optionid].loc[current_date]
            action = "buy" if leg.side == "short" else "sell"
            return fill_price(row["best_bid"], row["best_offer"], action, fill_mode)

        put_legs, call_legs = trade.put_legs, trade.call_legs
        put_cost = net_credit(put_legs, close_price) if put_legs else None
        call_cost = net_credit(call_legs, close_price) if call_legs else None

        cost_to_close = (put_cost or 0.0) + (call_cost or 0.0)

        put_pnl = entry_credit_by_right["P"] - put_cost if put_cost is not None else None
        call_pnl = entry_credit_by_right["C"] - call_cost if call_cost is not None else None

        return cost_to_close, put_pnl, call_pnl

    def simulate(self, trade: PositionSpec, leg_history: pd.DataFrame, config: ExecutionConfig) -> TradeResult:
        fill_mode = config.fill_mode
        hists = self._leg_histories(trade, leg_history)
        entry_credit_by_right = self._entry_credit_by_right(trade, fill_mode)
        entry_credit = (entry_credit_by_right["P"] or 0.0) + (entry_credit_by_right["C"] or 0.0)

        check_dates: set | None = None
        for hist in hists.values():
            check_dates = set(hist.index) if check_dates is None else check_dates & set(hist.index)
        ordered_check_dates = sorted(d for d in (check_dates or set()) if d > trade.entry_date)

        last_checked = None
        for current_date in ordered_check_dates:
            cost_to_close, put_pnl, call_pnl = self._pnl_at(trade, hists, current_date, entry_credit_by_right, fill_mode)
            dte_remaining = (trade.exdate - current_date).days
            last_checked = (current_date, cost_to_close, put_pnl, call_pnl)

            result = check_exit(trade, current_date, cost_to_close, entry_credit, dte_remaining, put_pnl, call_pnl, config)
            if result is not None:
                return result

        if last_checked is None:
            return TradeResult(trade.entry_date, trade.entry_date, "data_exhausted", entry_credit, entry_credit, 0.0)
        current_date, cost_to_close, put_pnl, call_pnl = last_checked
        return TradeResult(
            trade.entry_date, current_date, "data_exhausted", entry_credit, cost_to_close,
            entry_credit - cost_to_close, put_pnl, call_pnl,
        )

    def reprice_at_exit(
        self,
        trade: PositionSpec,
        leg_history: pd.DataFrame,
        exit_date,
        exit_reason: str,
        fill_mode: str,
    ) -> TradeResult:
        """Recomputes P&L for an *already-decided* entry/exit (e.g. from a
        prior `simulate()` call under a different fill_mode) without
        re-running the exit-trigger loop. Holds trade timing fixed and only
        changes the fill price assumption -- isolates "what would this exact
        same trading decision have been worth at a different execution
        price," for the slippage comparison (issue #8). Deliberately not
        "how would the strategy behave in a frictionless world" (that would
        also let exit timing itself shift, conflating two different effects).
        """
        hists = self._leg_histories(trade, leg_history)
        entry_credit_by_right = self._entry_credit_by_right(trade, fill_mode)
        entry_credit = (entry_credit_by_right["P"] or 0.0) + (entry_credit_by_right["C"] or 0.0)
        exit_date = pd.Timestamp(exit_date)

        if exit_date == trade.entry_date:
            return TradeResult(trade.entry_date, exit_date, exit_reason, entry_credit, entry_credit, 0.0)

        cost_to_close, put_pnl, call_pnl = self._pnl_at(trade, hists, exit_date, entry_credit_by_right, fill_mode)
        return TradeResult(
            trade.entry_date, exit_date, exit_reason, entry_credit, cost_to_close,
            entry_credit - cost_to_close, put_pnl, call_pnl,
        )

"""Fill-price, mark-to-market, exit-trigger, and max-loss-based sizing math --
ported from `Options_Trading_Strategy`'s `core/position_simulator.py`,
`core/sizing.py`, and `core/payoff.py` (the same capture-fraction math
`core/broker/shadow_engine.py` reuses there so live economics can't drift
from the backtest).

**Why this is duplicated rather than shared with `testing/engine/`'s own
identical-looking copy of this math:** Shadow's whole reason to exist is
that it never live-imports Testing (see `CONTEXT.md`'s Shadow glossary
entry and ADR 0001) -- not even for genuinely-shared, rarely-changing math.
A strategy's `config`/`selector` modules reach Shadow via a frozen
Graduation snapshot (`testing/graduation/`), but that snapshot's dependency
closure deliberately does NOT include `position_simulator.py`/`sizing.py`
(see `testing/graduation/graduate.py`'s module docstring: those stay live,
un-frozen Testing infrastructure, used by the backtest driver, not by a
strategy's own decision logic). So Shadow needs its own standalone copy of
this specific execution math to mark a live position to market and decide
entries/exits -- this module is that copy, independently ported from the
same `Options_Trading_Strategy` source Testing's own port started from, not
derived from `testing/engine/position_simulator.py`.

Every function here is duck-typed against a `Leg`/`PositionSpec`/
`ExecutionConfig`-shaped object (attributes: `leg.side`, `leg.right`,
`leg.best_bid`, `leg.best_offer`, `leg.optionid`, `leg.strike_price`;
`spec.legs`/`put_legs`/`call_legs`/`entry_date`/`exdate`;
`config.profit_target_pct`/`stop_loss_multiple`/`exit_dte`/
`exit_trigger_basis`/`fixed_notional`/`contract_multiplier`) rather than
importing concrete classes -- so it works unchanged against whatever
`Leg`/`PositionSpec`/`ExecutionConfig` classes a given graduated snapshot
happens to carry (each graduation freezes its own copy of those, see
`testing/graduation/snapshot.py`), with zero static import of any of them.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


class _LegLike(Protocol):
    optionid: object
    strike_price: float
    best_bid: float
    best_offer: float
    right: str
    side: str


def _leg_weight(leg: _LegLike) -> float:
    """`leg.weight` via getattr, not direct access -- older graduated
    snapshots (e.g. vrp_harvester's v1, from before `Leg.weight` existed)
    freeze a `Leg` class with no `weight` field at all, and this module
    duck-types against whatever shape a given snapshot's Leg happens to
    carry. Defaults to 1.0, matching every pre-variance_swap Leg's
    implicit weight. Found missing entirely from net_credit/
    max_loss_at_expiration during the front-to-back review (2026-07-12) --
    a real, live gap: variance_swap (genuinely non-uniform per-leg
    weights) just graduated to Shadow, and net_credit feeds
    entry_value_by_right/mark_to_market, i.e. every trade's real entry
    credit and daily P&L, not just sizing.
    """
    return getattr(leg, "weight", 1.0)


class _PositionSpecLike(Protocol):
    legs: list
    entry_date: Any

    @property
    def put_legs(self) -> list: ...

    @property
    def call_legs(self) -> list: ...

    @property
    def exdate(self) -> Any: ...


@dataclass(frozen=True)
class TradeResult:
    """Mirrors `core.position_simulator.TradeResult`'s shape -- one resolved
    exit (or a live "still open" caller can construct on its own)."""

    entry_date: Any
    exit_date: Any
    exit_reason: str  # "profit_target" | "hard_stop" | "dte_stop"
    entry_credit: float
    exit_debit: float
    pnl: float
    put_pnl: float | None = None
    call_pnl: float | None = None


def fill_price(bid: float, offer: float, action: str, fill_mode) -> float:
    """`action` is "sell" (receive bid) or "buy" (pay offer). `fill_mode`
    accepts "bid_ask" (capture=0.0, realistic/pessimistic), "mid"
    (capture=1.0, optimistic ceiling), or a numeric capture fraction in
    [0, 1] directly -- identical semantics to the source repo's
    `core.position_simulator.fill_price`, ported byte-for-byte.
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


def net_credit(legs: list, price_fn: Callable) -> float:
    """Sums signed, weighted credit across a set of legs sharing one
    `right` -- short legs contribute +price (sold), long legs contribute
    -price (bought), each leg's contribution scaled by `_leg_weight(leg)`
    (1.0 for every pre-variance_swap Leg, so this changes nothing for
    them; variance_swap's strike-strip is the first caller with non-uniform
    weights).
    """
    credit = 0.0
    for leg in legs:
        signed_price = price_fn(leg) if leg.side == "short" else -price_fn(leg)
        credit += signed_price * _leg_weight(leg)
    return credit


def entry_value_by_right(spec: _PositionSpecLike, capture: float) -> dict:
    """Per-right (P/C) net entry value estimate at a given spread-capture
    fraction, from each leg's currently-known bid/offer. Positive = net
    credit collected; negative = net debit paid.
    """

    def entry_price(leg):
        action = "sell" if leg.side == "short" else "buy"
        return fill_price(leg.best_bid, leg.best_offer, action, capture)

    return {
        "P": net_credit(spec.put_legs, entry_price) if spec.put_legs else None,
        "C": net_credit(spec.call_legs, entry_price) if spec.call_legs else None,
    }


def mark_to_market(spec: _PositionSpecLike, quotes: dict, entry_credit_by_right: dict, fill_mode) -> tuple:
    """Cost-to-close and put/call P&L attribution from current bid/ask
    quotes per leg (`{optionid: (bid, offer)}`).
    """

    def close_price(leg):
        bid, offer = quotes[leg.optionid]
        action = "buy" if leg.side == "short" else "sell"
        return fill_price(bid, offer, action, fill_mode)

    put_legs, call_legs = spec.put_legs, spec.call_legs
    put_cost = net_credit(put_legs, close_price) if put_legs else None
    call_cost = net_credit(call_legs, close_price) if call_legs else None
    cost_to_close = (put_cost or 0.0) + (call_cost or 0.0)

    put_pnl = entry_credit_by_right["P"] - put_cost if put_cost is not None else None
    call_pnl = entry_credit_by_right["C"] - call_cost if call_cost is not None else None

    return cost_to_close, put_pnl, call_pnl


def _intrinsic(leg: _LegLike, spot: float) -> float:
    if leg.right == "P":
        return max(leg.strike_price - spot, 0.0)
    return max(spot - leg.strike_price, 0.0)


def max_loss_at_expiration(legs: list, entry_credit: float) -> float:
    """Worst-case loss at expiration, in premium terms (positive = a loss)
    -- ported from `core.payoff.max_loss_at_expiration` (see that module's
    docstring for the full piecewise-linear-kink argument). `math.inf` for
    a genuinely uncovered short call.
    """
    if not legs:
        return 0.0

    weighted_short_calls = sum(_leg_weight(leg) for leg in legs if leg.right == "C" and leg.side == "short")
    weighted_long_calls = sum(_leg_weight(leg) for leg in legs if leg.right == "C" and leg.side == "long")
    if weighted_short_calls > weighted_long_calls:
        return math.inf

    candidate_spots = {0.0} | {leg.strike_price for leg in legs}

    def pnl_at(spot: float) -> float:
        cost_to_close = sum(
            (_intrinsic(leg, spot) if leg.side == "short" else -_intrinsic(leg, spot)) * _leg_weight(leg)
            for leg in legs
        )
        return entry_credit - cost_to_close

    worst_pnl = min(pnl_at(spot) for spot in candidate_spots)
    return max(-worst_pnl, 0.0)


def check_exit(
    spec: _PositionSpecLike,
    current_date,
    cost_to_close: float,
    entry_credit: float,
    dte_remaining: int,
    put_pnl: float | None,
    call_pnl: float | None,
    config,
) -> TradeResult | None:
    """Checks the three exit conditions in precedence order -- profit
    target, hard stop, DTE stop -- ported from `core.position_simulator
    .check_exit`. `config.exit_trigger_basis` selects what
    profit_target_pct/stop_loss_multiple are a fraction of: "entry_credit"
    (default, abs(entry_credit)) or "max_loss"
    (`max_loss_at_expiration(spec.legs, entry_credit)`), matching
    `Options_Trading_Strategy` ADR 0011.
    """
    profit_so_far = entry_credit - cost_to_close

    exit_trigger_basis = getattr(config, "exit_trigger_basis", "entry_credit")
    if exit_trigger_basis == "max_loss":
        trigger_basis = max_loss_at_expiration(spec.legs, entry_credit)
    else:
        trigger_basis = abs(entry_credit)

    if profit_so_far >= config.profit_target_pct * trigger_basis:
        return TradeResult(
            spec.entry_date, current_date, "profit_target", entry_credit, cost_to_close, profit_so_far,
            put_pnl, call_pnl,
        )

    loss = cost_to_close - entry_credit
    if loss >= config.stop_loss_multiple * trigger_basis:
        return TradeResult(
            spec.entry_date, current_date, "hard_stop", entry_credit, cost_to_close, -loss, put_pnl, call_pnl,
        )

    if dte_remaining <= config.exit_dte:
        return TradeResult(
            spec.entry_date, current_date, "dte_stop", entry_credit, cost_to_close, profit_so_far,
            put_pnl, call_pnl,
        )

    return None


def contracts_for_entry(spec: _PositionSpecLike, entry_credit: float, config, notional: float | None = None) -> int:
    """Max-loss-based contract count -- ported from `core.sizing
    .contracts_for_entry`. `notional` defaults to `config.fixed_notional`.
    """
    if notional is None:
        notional = config.fixed_notional
    max_loss_per_contract = max_loss_at_expiration(spec.legs, entry_credit) * config.contract_multiplier
    return max(1, int(notional // max_loss_per_contract)) if max_loss_per_contract > 0 else 1

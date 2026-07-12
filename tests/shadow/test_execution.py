"""Behavior-preservation tests for shadow/execution.py -- ported from
Options_Trading_Strategy's tests/core/test_payoff.py, test_sizing.py, and
test_position_simulator.py (see those files for the original fixtures;
several exact reference numbers below are reused verbatim to prove this is
a byte-for-byte-behavior port, not a re-derivation, per ADR 0001's
"migration, not a rewrite" framing).

Uses small local test-double Leg/PositionSpec/Config classes rather than
importing anything from testing.engine -- deliberately, since
shadow/execution.py is duck-typed against *any* compatible shape (a
graduated snapshot's own frozen classes in production), and this proves
that duck-typing actually works against an independent, non-testing.engine
implementation of the same shape, not just one specific class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pytest

from shadow.execution import (
    check_exit,
    contracts_for_entry,
    entry_value_by_right,
    fill_price,
    mark_to_market,
    max_loss_at_expiration,
    net_credit,
)

_EXDATE = pd.Timestamp("2020-02-15")


@dataclass(frozen=True)
class Leg:
    optionid: object
    exdate: pd.Timestamp
    strike_price: float
    delta: float
    best_bid: float
    best_offer: float
    right: str
    side: str
    impl_volatility: float | None = None


@dataclass(frozen=True)
class PositionSpec:
    entry_date: pd.Timestamp
    legs: list[Leg]

    @property
    def exdate(self) -> pd.Timestamp:
        return min(leg.exdate for leg in self.legs)

    @property
    def put_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.right == "P"]

    @property
    def call_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.right == "C"]


@dataclass(frozen=True)
class Config:
    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    exit_dte: int = 21
    exit_trigger_basis: Literal["entry_credit", "max_loss"] = "entry_credit"
    fixed_notional: float = 100_000.0
    contract_multiplier: int = 100


def _leg(strike, right, side, optionid=None, bid=0.0, offer=0.0):
    return Leg(
        optionid=optionid or f"{right}{side}{strike}", exdate=_EXDATE, strike_price=strike,
        delta=0.0, best_bid=bid, best_offer=offer, right=right, side=side,
    )


def _spread(short_strike=95, long_strike=90):
    short_put = Leg(optionid="short", exdate=_EXDATE, strike_price=short_strike, delta=-0.20, best_bid=3.0, best_offer=3.2, right="P", side="short")
    long_put = Leg(optionid="long", exdate=_EXDATE, strike_price=long_strike, delta=-0.10, best_bid=1.0, best_offer=1.2, right="P", side="long")
    return PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[short_put, long_put])


def _condor(short_put=95, long_put=90, short_call=105, long_call=115):
    legs = [
        Leg(optionid="sp", exdate=_EXDATE, strike_price=short_put, delta=-0.20, best_bid=3.0, best_offer=3.2, right="P", side="short"),
        Leg(optionid="lp", exdate=_EXDATE, strike_price=long_put, delta=-0.10, best_bid=1.0, best_offer=1.2, right="P", side="long"),
        Leg(optionid="sc", exdate=_EXDATE, strike_price=short_call, delta=0.30, best_bid=2.0, best_offer=2.2, right="C", side="short"),
        Leg(optionid="lc", exdate=_EXDATE, strike_price=long_call, delta=0.20, best_bid=0.8, best_offer=1.0, right="C", side="long"),
    ]
    return PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=legs)


# ---------------------------------------------------------------------
# fill_price / net_credit
# ---------------------------------------------------------------------


def test_fill_price_bid_ask_sells_at_bid_buys_at_offer():
    assert fill_price(bid=10.0, offer=10.5, action="sell", fill_mode="bid_ask") == 10.0
    assert fill_price(bid=10.0, offer=10.5, action="buy", fill_mode="bid_ask") == 10.5


def test_fill_price_mid_both_actions_land_on_midpoint():
    assert fill_price(bid=10.0, offer=10.5, action="sell", fill_mode="mid") == 10.25
    assert fill_price(bid=10.0, offer=10.5, action="buy", fill_mode="mid") == 10.25


def test_fill_price_numeric_capture_interpolates():
    # capture=0.5 halfway between bid_ask and mid
    assert fill_price(bid=10.0, offer=11.0, action="sell", fill_mode=0.5) == 10.25
    assert fill_price(bid=10.0, offer=11.0, action="buy", fill_mode=0.5) == 10.75


def test_net_credit_short_adds_long_subtracts():
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long")]
    assert net_credit(legs, lambda leg: 1.0) == 0.0  # +1 - 1


@dataclass(frozen=True)
class WeightedLeg:
    """A second, independent Leg-like shape that DOES carry `weight` --
    proves net_credit/max_loss_at_expiration duck-type against it via
    getattr, same as `Leg` above (no `weight` field at all) proves the
    no-weight case. Mirrors testing.engine.leg.Leg's variance_swap-era
    `weight` field, added during the front-to-back review (2026-07-12)
    after finding net_credit/max_loss_at_expiration here silently ignored
    it -- a real gap, since variance_swap's real legs carry genuinely
    non-uniform weights and just graduated to Shadow.
    """

    optionid: object
    exdate: pd.Timestamp
    strike_price: float
    best_bid: float
    best_offer: float
    right: str
    side: str
    weight: float = 1.0


def test_net_credit_scales_by_leg_weight():
    legs = [
        WeightedLeg(optionid="a", exdate=_EXDATE, strike_price=100, best_bid=0.0, best_offer=0.0, right="P", side="short", weight=3.0),
        WeightedLeg(optionid="b", exdate=_EXDATE, strike_price=90, best_bid=0.0, best_offer=0.0, right="P", side="long", weight=2.0),
    ]
    # +1*3.0 - 1*2.0 = 1.0, not the unweighted +1-1=0.0
    assert net_credit(legs, lambda leg: 1.0) == 1.0


def test_net_credit_falls_back_to_weight_one_when_leg_has_no_weight_field():
    # Leg (above) has no `weight` attribute at all -- the shape an older
    # graduated snapshot (e.g. vrp_harvester's v1, pre-dating Leg.weight)
    # freezes. Must not raise, and must behave identically to weight=1.0.
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long")]
    assert net_credit(legs, lambda leg: 1.0) == 0.0


def test_max_loss_at_expiration_weighted_naked_short_call_is_still_unbounded():
    legs = [WeightedLeg(optionid="a", exdate=_EXDATE, strike_price=50, best_bid=0.0, best_offer=0.0, right="C", side="short", weight=0.001)]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == math.inf


def test_max_loss_at_expiration_weighted_matches_hand_computed_value():
    # Same shape as the plain-vertical reference test above, but with
    # weight=2.0 on both legs -- only the leg-based cost-to-close scales by
    # weight, not entry_credit (a given dollar value, not a per-leg
    # quantity): worst point is spot=90, cost_to_close = 5*2 - 0*2 = 10,
    # max_loss = 10 - 1.8 = 8.2, not the unweighted case's 3.2.
    legs = [
        WeightedLeg(optionid="short", exdate=_EXDATE, strike_price=95, best_bid=0.0, best_offer=0.0, right="P", side="short", weight=2.0),
        WeightedLeg(optionid="long", exdate=_EXDATE, strike_price=90, best_bid=0.0, best_offer=0.0, right="P", side="long", weight=2.0),
    ]
    assert max_loss_at_expiration(legs, entry_credit=1.8) == pytest.approx(8.2)


# ---------------------------------------------------------------------
# max_loss_at_expiration -- ported reference values from test_payoff.py
# ---------------------------------------------------------------------


def test_matches_old_width_formula_for_a_plain_vertical():
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=1.8) == 3.2


def test_matches_old_width_formula_for_iron_condor_wider_call_side():
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long"), _leg(105, "C", "short"), _leg(115, "C", "long")]
    assert max_loss_at_expiration(legs, entry_credit=2.8) == 7.2


def test_three_leg_structure_finds_the_elbow_not_the_outer_span():
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long"), _leg(85, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=0.5) == 4.5


def test_naked_short_put_is_bounded_at_spot_zero_not_zero_width():
    legs = [_leg(50, "P", "short")]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == 49.0


def test_naked_short_call_is_unbounded():
    legs = [_leg(50, "C", "short")]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == math.inf


def test_covered_call_not_naked_is_bounded():
    legs = [_leg(50, "C", "short"), _leg(60, "C", "long")]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == 9.0


def test_empty_legs_returns_zero():
    assert max_loss_at_expiration([], entry_credit=0.0) == 0.0


# ---------------------------------------------------------------------
# contracts_for_entry -- ported reference values from test_sizing.py
# ---------------------------------------------------------------------


def test_contracts_for_entry_uses_put_width_for_a_put_spread():
    spec = _spread(short_strike=95, long_strike=90)
    config = Config()
    assert contracts_for_entry(spec, entry_credit=1.8, config=config) == 312


def test_contracts_for_entry_uses_wider_side_for_iron_condor():
    spec = _condor(short_put=95, long_put=90, short_call=105, long_call=115)
    config = Config()
    assert contracts_for_entry(spec, entry_credit=2.8, config=config) == 138


def test_contracts_for_entry_respects_explicit_notional_override():
    spec = _spread()
    config = Config()
    assert contracts_for_entry(spec, entry_credit=1.8, config=config, notional=50_000.0) == 156


def test_contracts_for_entry_floors_at_one_when_max_loss_exceeds_notional():
    spec = _spread(short_strike=995, long_strike=90)
    config = Config(fixed_notional=100.0)
    assert contracts_for_entry(spec, entry_credit=1.8, config=config) == 1


# ---------------------------------------------------------------------
# entry_value_by_right / mark_to_market / check_exit
# ---------------------------------------------------------------------


def test_entry_value_by_right_floor_capture_is_bid_ask_credit():
    spec = _spread()  # short 3.0/3.2, long 1.0/1.2
    by_right = entry_value_by_right(spec, capture=0.0)
    # short sells at bid (3.0), long buys at offer (1.2) -> 3.0 - 1.2 = 1.8
    assert by_right["P"] == pytest.approx(1.8)
    assert by_right["C"] is None


def test_mark_to_market_cost_to_close_and_pnl_attribution():
    spec = _spread()
    entry_by_right = entry_value_by_right(spec, capture=0.0)  # 1.8
    quotes = {"short": (0.5, 0.6), "long": (0.2, 0.3)}
    # close: short buys at offer 0.6, long sells at bid 0.2 -> cost = 0.6 - 0.2 = 0.4
    cost, put_pnl, call_pnl = mark_to_market(spec, quotes, entry_by_right, fill_mode=0.0)
    assert cost == pytest.approx(0.4)
    assert put_pnl == pytest.approx(1.8 - 0.4)
    assert call_pnl is None


def test_check_exit_profit_target_fires():
    spec = _spread()
    config = Config()
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=0.4, entry_credit=1.8,
                         dte_remaining=30, put_pnl=1.4, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "profit_target"


def test_check_exit_hard_stop_fires():
    spec = _spread()
    config = Config()
    # loss = cost_to_close(4.0) - entry_credit(1.8) = 2.2 >= 2.0*1.8=3.6? no -> use bigger loss
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=6.0, entry_credit=1.8,
                         dte_remaining=30, put_pnl=-4.2, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "hard_stop"


def test_check_exit_dte_stop_fires_when_neither_target_nor_stop_hit():
    spec = _spread()
    config = Config()
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=1.7, entry_credit=1.8,
                         dte_remaining=10, put_pnl=0.1, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "dte_stop"


def test_check_exit_returns_none_when_nothing_fires():
    spec = _spread()
    config = Config()
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=1.7, entry_credit=1.8,
                         dte_remaining=30, put_pnl=0.1, call_pnl=None, config=config)
    assert result is None


def test_check_exit_abs_handles_net_debit_correctly():
    """Same-strike calendar spread: net debit (entry_credit negative). abs()
    keeps profit_target/hard_stop meaning "X% of what was paid," not
    silently inverting for a negative signed value (ADR 0011's concrete bug)."""
    near = _leg(100, "P", "short")
    far = _leg(100, "P", "long")
    spec = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[near, far])
    config = Config()
    entry_credit = -1.2  # net debit paid
    # cost_to_close = -1.9 (cheaper to close than to open -- a gain)
    # profit_so_far = -1.2 - (-1.9) = 0.7 >= 0.5 * abs(-1.2) = 0.6 -> profit_target
    result = check_exit(spec, pd.Timestamp("2020-01-18"), cost_to_close=-1.9, entry_credit=entry_credit,
                         dte_remaining=30, put_pnl=0.7, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "profit_target"


def test_entry_value_by_right_computes_both_put_and_call_independently():
    """Mutation-testing gap (2026-07-12): put_legs/call_legs are computed by
    two separate net_credit() calls -- a test using only _spread() (put-only)
    can't distinguish "C uses call_legs" from "C uses put_legs again" or
    "C's price_fn param got swapped." _condor() has both sides active."""
    spec = _condor()  # put 3.0/3.2-1.0/1.2, call 2.0/2.2-0.8/1.0
    by_right = entry_value_by_right(spec, capture=0.0)
    # put: short sells at bid 3.0, long buys at offer 1.2 -> 3.0 - 1.2 = 1.8
    assert by_right["P"] == pytest.approx(1.8)
    # call: short sells at bid 2.0, long buys at offer 1.0 -> 2.0 - 1.0 = 1.0
    assert by_right["C"] == pytest.approx(1.0)


def test_mark_to_market_put_and_call_combine_additively():
    """Mutation-testing gap (2026-07-12): cost_to_close = put_cost + call_cost
    survived being flipped to put_cost - call_cost, because every prior
    mark_to_market test used a put-only _spread(). This is the same failure
    shape as the leg.weight bug found earlier this session -- variance_swap
    (graduated to Shadow) genuinely has both sides active at once, so an
    additive-vs-subtractive bug here would silently corrupt its live P&L."""
    spec = _condor()
    entry_by_right = entry_value_by_right(spec, capture=0.0)  # P=1.8, C=1.0
    quotes = {"sp": (0.5, 0.6), "lp": (0.2, 0.3), "sc": (0.3, 0.4), "lc": (0.1, 0.2)}
    # put close: short buys at offer 0.6, long sells at bid 0.2 -> 0.6-0.2=0.4
    # call close: short buys at offer 0.4, long sells at bid 0.1 -> 0.4-0.1=0.3
    cost, put_pnl, call_pnl = mark_to_market(spec, quotes, entry_by_right, fill_mode=0.0)
    assert cost == pytest.approx(0.4 + 0.3)  # additive -- NOT 0.4 - 0.3
    assert put_pnl == pytest.approx(1.8 - 0.4)
    assert call_pnl == pytest.approx(1.0 - 0.3)  # NOT 1.0 + 0.3


def test_check_exit_profit_target_boundary_is_inclusive():
    """profit_so_far exactly equal to the trigger (not just past it) must
    fire -- config.profit_target_pct * trigger_basis == profit_so_far."""
    spec = _spread()
    config = Config(profit_target_pct=0.50)  # trigger_basis = abs(1.8) = 1.8, threshold = 0.90
    # entry_credit - cost_to_close == 0.90 exactly
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=0.90, entry_credit=1.80,
                         dte_remaining=30, put_pnl=0.90, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "profit_target"


def test_check_exit_hard_stop_boundary_is_inclusive():
    """loss exactly equal to stop_loss_multiple * trigger_basis must fire --
    mutation testing found `>=` silently survives as `>` here (2026-07-12),
    meaning the exact stop-loss boundary was never verified. Uses
    integer-valued inputs deliberately: 1.80/2.0-style decimals don't
    round-trip through +/- exactly in float64, so a loss "constructed" to
    equal the threshold can silently land a float ULP past it and pass even
    under the buggy `>` -- integers avoid that trap and land bit-exact."""
    spec = _spread()
    config = Config(stop_loss_multiple=2.0)
    entry_credit = 2.0  # trigger_basis = abs(2.0) = 2.0, threshold = 2.0*2.0 = 4.0
    cost_to_close = entry_credit + 4.0  # loss = cost_to_close - entry_credit == 4.0 exactly
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=cost_to_close, entry_credit=entry_credit,
                         dte_remaining=30, put_pnl=-4.0, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "hard_stop"


def test_check_exit_hard_stop_trigger_is_a_product_not_a_quotient():
    """stop_loss_multiple * trigger_basis survived being flipped to
    stop_loss_multiple / trigger_basis (2026-07-12) -- at trigger_basis > 1
    the two diverge sharply, so a loss that clears the real (multiplied)
    threshold but not the (much smaller) divided one distinguishes them."""
    spec = _spread()
    config = Config(stop_loss_multiple=2.0)  # trigger_basis = abs(1.8) = 1.8
    # real threshold = 2.0 * 1.8 = 3.6; divided threshold = 2.0 / 1.8 = 1.11
    # loss = 2.0 clears the divided threshold but not the real one -> must NOT fire hard_stop
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=3.80, entry_credit=1.80,
                         dte_remaining=30, put_pnl=-2.0, call_pnl=None, config=config)
    assert result is None or result.exit_reason != "hard_stop"


def test_check_exit_dte_stop_boundary_is_inclusive():
    """dte_remaining exactly equal to config.exit_dte must fire dte_stop --
    mutation testing found `<=` silently survives as `<` here (2026-07-12)."""
    spec = _spread()
    config = Config(exit_dte=21)
    result = check_exit(spec, pd.Timestamp("2020-01-25"), cost_to_close=1.85, entry_credit=1.80,
                         dte_remaining=21, put_pnl=-0.05, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "dte_stop"


def test_check_exit_default_trigger_basis_is_entry_credit_when_config_lacks_the_attribute():
    """A legacy/minimal config shape (predates ADR 0011's exit_trigger_basis
    field) must still behave like exit_trigger_basis="entry_credit", not
    silently use max_loss -- no prior test used a config actually missing
    the attribute (Config() above always sets it).

    Note the *exact string* getattr's default falls back to
    ("entry_credit" vs None vs anything else) is genuinely unkillable by
    any test: the only place it's read is `== "max_loss"`, so every
    non-"max_loss" default is behaviorally identical. Confirmed by hand
    (mutmut_6/12/13 all survive and are equivalent, not real gaps) rather
    than assumed -- this test exists to pin the entry_credit-vs-max_loss
    *behavior* for a missing attribute, not that specific string."""

    @dataclass(frozen=True)
    class MinimalConfig:
        profit_target_pct: float = 0.50
        stop_loss_multiple: float = 2.0
        exit_dte: int = 21
        fixed_notional: float = 100_000.0
        contract_multiplier: int = 100
        # deliberately no exit_trigger_basis field

    spec = _spread()  # width 5, abs(entry_credit)=1.8 vs max_loss=3.2 -- values that distinguish the two bases
    config = MinimalConfig(profit_target_pct=0.10)
    # 10% of abs(1.8)=0.18 -> profit_so_far=0.40 clears it (entry_credit basis)
    # 10% of max_loss=3.2 -> 0.32 also clears 0.40, so widen the gap: use a
    # profit_so_far that clears entry_credit-basis but not max_loss-basis
    result = check_exit(spec, pd.Timestamp("2020-01-18"), cost_to_close=1.60, entry_credit=1.80,
                         dte_remaining=30, put_pnl=0.20, call_pnl=None, config=config)
    assert result is not None and result.exit_reason == "profit_target"


def test_contracts_for_entry_zero_max_loss_returns_one_without_dividing_by_zero():
    """max_loss_per_contract > 0 else 1 -- mutation testing found `> 0`
    silently survives as `>= 0` (2026-07-12). At max_loss_per_contract == 0
    exactly, `>= 0` takes the true branch and divides by zero; `> 0` (correct)
    falls through to the safe `else 1`. A naked short put at a strike equal
    to its own entry credit has exactly zero worst-case loss at expiration."""
    spec = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[_leg(95, "P", "short")])
    config = Config()
    # worst_pnl at spot=0: entry_credit(95) - intrinsic(95) = 95-95 = 0 -> max_loss = 0
    assert contracts_for_entry(spec, entry_credit=95.0, config=config) == 1


def test_check_exit_max_loss_basis_uses_max_loss_at_expiration():
    """exit_trigger_basis='max_loss' (ADR 0011) swaps the trigger basis
    from abs(entry_credit) to max_loss_at_expiration -- verifies the
    ported check_exit actually reads config.exit_trigger_basis."""
    spec = _spread()  # width 5, entry_credit small -> max_loss = 5 - credit, much bigger than abs(credit)
    config = Config(exit_trigger_basis="max_loss", profit_target_pct=0.10)
    entry_credit = 0.10
    # max_loss = 5 - 0.10 = 4.9; 10% of that = 0.49 -- needs profit_so_far >= 0.49 to fire
    # profit_so_far = 0.10 - (-0.30) = 0.40 -- NOT enough under max_loss basis
    result_holds = check_exit(spec, pd.Timestamp("2020-01-18"), cost_to_close=-0.30, entry_credit=entry_credit,
                               dte_remaining=30, put_pnl=0.40, call_pnl=None, config=config)
    assert result_holds is None
    # under entry_credit basis (default) the same numbers WOULD fire:
    # 10% of abs(0.10) = 0.01 <= profit_so_far 0.40
    config_default = Config(profit_target_pct=0.10)
    result_fires = check_exit(spec, pd.Timestamp("2020-01-18"), cost_to_close=-0.30, entry_credit=entry_credit,
                               dte_remaining=30, put_pnl=0.40, call_pnl=None, config=config_default)
    assert result_fires is not None and result_fires.exit_reason == "profit_target"

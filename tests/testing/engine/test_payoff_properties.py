"""Property-based tests for payoff.py's max_loss_at_expiration -- complements
test_payoff.py's hand-picked reference cases with invariants checked across
a randomly-sampled input space. Two of these encode specific things this
front-to-back review actually found/verified by hand (2026-07-12): the
uniform-weight linear-scaling property is the exact invariant that would
have caught the shadow/execution.py leg-weight bug if it had existed there
too, and the same-strike-calendar-collapses-to-entry-credit property
permanently locks in the "false alarm" analytical result from that same
review, so it can't quietly regress into an actual bug later.
"""

import math

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from testing.engine.leg import Leg
from testing.engine.payoff import max_loss_at_expiration

_EXDATE = pd.Timestamp("2020-02-15")

_strike = st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
_credit = st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_weight = st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
_side = st.sampled_from(["short", "long"])
_right = st.sampled_from(["P", "C"])


def _leg(strike, right, side, optionid, weight=1.0):
    return Leg(
        optionid=optionid, exdate=_EXDATE, strike_price=strike, delta=0.0,
        best_bid=0.0, best_offer=0.0, right=right, side=side, weight=weight,
    )


@given(short_strike=_strike, long_strike=_strike, right=_right, weight=_weight)
@settings(max_examples=150)
def test_uniform_weight_scaling_is_linear_at_zero_credit(short_strike, long_strike, right, weight):
    # At entry_credit=0, scaling every leg's weight by a constant factor k
    # must scale the whole result by k -- max_loss_at_expiration only ever
    # sums leg-weighted intrinsic values against candidate spots, so this
    # holds for any 2-leg vertical (same or different strikes, same right).
    # This is exactly the invariant the shadow/execution.py leg-weight bug
    # (found this same review) would have violated had it existed here.
    unweighted = [
        _leg(short_strike, right, "short", "s"),
        _leg(long_strike, right, "long", "l"),
    ]
    weighted = [
        _leg(short_strike, right, "short", "s", weight=weight),
        _leg(long_strike, right, "long", "l", weight=weight),
    ]
    base = max_loss_at_expiration(unweighted, entry_credit=0.0)
    scaled = max_loss_at_expiration(weighted, entry_credit=0.0)
    if math.isinf(base):
        assert math.isinf(scaled)
    else:
        # Two independent computation paths (compute at weight=1 then scale
        # externally, vs. compute directly at the target weight) aren't
        # guaranteed bit-identical -- only mathematically equal.
        assert _approx_eq(scaled, base * weight, tol=max(1e-9, abs(base * weight) * 1e-9))


@given(strike=_strike, credit=_credit, right=_right, weight=_weight)
@settings(max_examples=150)
def test_same_strike_same_right_two_leg_position_equals_abs_entry_credit(strike, credit, right, weight):
    # A short+long pair at the identical strike/right (a same-strike
    # calendar, or any degenerate same-strike vertical) has intrinsic
    # values that cancel exactly at every candidate spot, regardless of
    # weight -- the worst case is always just the debit/credit paid.
    # Verified analytically during the front-to-back review when this
    # looked like it might be a bug (it wasn't) -- locked in here so it
    # can't silently regress.
    legs = [
        _leg(strike, right, "short", "s", weight=weight),
        _leg(strike, right, "long", "l", weight=weight),
    ]
    result = max_loss_at_expiration(legs, entry_credit=credit)
    assert _approx_eq(result, max(-credit, 0.0))


def _approx_eq(a, b, tol=1e-9):
    return abs(a - b) <= tol


@given(
    strikes=st.lists(_strike, min_size=1, max_size=4, unique=True),
    credit=_credit,
    weight=_weight,
)
@settings(max_examples=100)
def test_naked_short_calls_with_no_long_calls_are_always_unbounded(strikes, credit, weight):
    legs = [_leg(k, "C", "short", f"c{i}", weight=weight) for i, k in enumerate(strikes)]
    assert max_loss_at_expiration(legs, entry_credit=credit) == math.inf


@given(
    naked_strike=_strike,
    # A fraction of naked_strike, not an independent absolute gap -- keeps
    # the derived protective_strike realistically positive for a put (a
    # real strike is never <= 0; ingestion/schema.py's ContractRow.strike
    # enforces this with Field(gt=0), though testing.engine.leg.Leg itself
    # doesn't, since real strikes only ever arrive via real chain data).
    gap_fraction=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False),
    right=_right,
    credit=_credit,
    weight=_weight,
)
@settings(max_examples=150)
def test_adding_a_protective_long_leg_never_increases_max_loss(naked_strike, gap_fraction, right, credit, weight):
    # Buying protection further OTM (a long put below a short put's strike,
    # or a long call above a short call's strike) can only cap risk, never
    # add to it -- the two-legged spread's max loss must be <= the naked
    # single leg's own max loss (which may itself be math.inf for a naked
    # short call).
    naked = [_leg(naked_strike, right, "short", "s", weight=weight)]
    protection_gap = naked_strike * gap_fraction
    protective_strike = naked_strike - protection_gap if right == "P" else naked_strike + protection_gap
    protected = naked + [_leg(protective_strike, right, "long", "l", weight=weight)]

    naked_loss = max_loss_at_expiration(naked, entry_credit=credit)
    protected_loss = max_loss_at_expiration(protected, entry_credit=credit)

    if math.isinf(naked_loss):
        assert not math.isinf(protected_loss) or protected_loss == naked_loss
    else:
        assert protected_loss <= naked_loss + 1e-9

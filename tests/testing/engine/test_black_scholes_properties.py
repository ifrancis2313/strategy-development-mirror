"""Property-based tests for black_scholes.py -- complements the example-based
tests in test_payoff.py/test_position_simulator.py with invariants checked
across a wide, randomly-sampled input space instead of a handful of
hand-picked points. Put-call parity in particular is THE defining identity
of the Black-Scholes model -- if it ever breaks, something is fundamentally
wrong with the pricing formula, not just a rounding nit.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from testing.engine.black_scholes import black_scholes_price

# Bounded to realistic-ish ranges: extreme values (e.g. vol=1e-8 or
# T=1e-8) push d1/d2 toward +/-inf and make float comparisons meaningless
# noise rather than real bugs -- the T=0/vol=0 boundary itself is covered
# separately below via an exact (not sampled-near-zero) case.
_spot = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_strike = st.floats(min_value=1.0, max_value=10_000.0, allow_nan=False, allow_infinity=False)
_time = st.floats(min_value=0.01, max_value=5.0, allow_nan=False, allow_infinity=False)
_vol = st.floats(min_value=0.01, max_value=3.0, allow_nan=False, allow_infinity=False)
_rate = st.floats(min_value=0.0, max_value=0.20, allow_nan=False, allow_infinity=False)


@given(S=_spot, K=_strike, T=_time, vol=_vol, r=_rate)
@settings(max_examples=200)
def test_put_call_parity(S, K, T, vol, r):
    # The defining identity: C - P = S - K*e^(-rT), for the same
    # (S, K, T, vol, r) on both sides.
    call = black_scholes_price(S, K, T, vol, "C", r)
    put = black_scholes_price(S, K, T, vol, "P", r)
    assert call - put == _approx(S - K * math.exp(-r * T))


@given(S=_spot, K=_strike, T=_time, vol=_vol, r=_rate)
@settings(max_examples=200)
def test_price_never_negative(S, K, T, vol, r):
    assert black_scholes_price(S, K, T, vol, "C", r) >= -1e-9
    assert black_scholes_price(S, K, T, vol, "P", r) >= -1e-9


@given(S=_spot, K=_strike, T=_time, vol=_vol, r=_rate)
@settings(max_examples=200)
def test_call_never_worth_more_than_spot(S, K, T, vol, r):
    # A call's value can never exceed owning the underlying outright --
    # the standard no-arbitrage upper bound.
    call = black_scholes_price(S, K, T, vol, "C", r)
    assert call <= S + 1e-6


@given(S=_spot, K=_strike, T=_time, vol=_vol, r=_rate)
@settings(max_examples=200)
def test_put_never_worth_more_than_discounted_strike(S, K, T, vol, r):
    put = black_scholes_price(S, K, T, vol, "P", r)
    assert put <= K * math.exp(-r * T) + 1e-6


@given(S=_spot, K=_strike, r=_rate, right=st.sampled_from(["C", "P"]))
@settings(max_examples=100)
def test_collapses_to_exact_intrinsic_at_expiry(S, K, r, right):
    # time_to_expiry_years=0 must equal exact intrinsic value, not just
    # something close to it -- this is the explicit boundary guard the
    # function's own docstring describes, and scenario_based_margin_with_
    # vol_shock's correctness for a leg with no recorded IV depends on this
    # being exact.
    intrinsic = max(S - K, 0.0) if right == "C" else max(K - S, 0.0)
    assert black_scholes_price(S, K, 0.0, 0.3, right, r) == intrinsic


@given(S=_spot, K=_strike, T=_time, r=_rate, right=st.sampled_from(["C", "P"]))
@settings(max_examples=100)
def test_collapses_to_exact_intrinsic_at_zero_vol(S, K, T, r, right):
    intrinsic = max(S - K, 0.0) if right == "C" else max(K - S, 0.0)
    assert black_scholes_price(S, K, T, 0.0, right, r) == intrinsic


def _approx(expected, tol=1e-6):
    """Local tolerance helper -- pytest.approx's default relative tolerance
    is too loose for values near zero (deep OTM parity differences) and too
    tight for the largest sampled spot/strike combinations; a small
    absolute+relative band matches what these prices' own float precision
    actually supports."""
    return pytest.approx(expected, abs=tol, rel=1e-6)

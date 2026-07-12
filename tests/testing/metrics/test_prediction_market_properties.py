"""Property-based tests for prediction_market.py's Kelly-sizing functions --
complements test_prediction_market.py's hand-picked reference cases
(including the canonical 60%/even-money f*=0.2 worked example) with
invariants checked across a randomly-sampled input space.
"""

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from testing.metrics.prediction_market import (
    kelly_fraction_binary_market,
    kelly_fraction_from_odds,
)

_prob = st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)
_odds = st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False)
_price = st.floats(min_value=0.001, max_value=0.999, allow_nan=False, allow_infinity=False)


@given(prob=_prob, b=_odds)
@settings(max_examples=200)
def test_breakeven_odds_gives_zero_kelly_fraction(prob, b):
    # Fair/breakeven odds for a given probability: b = (1-p)/p (the market
    # is offering exactly what the probability implies, no edge either
    # way) -- f* must be ~0 at exactly that point.
    fair_b = (1.0 - prob) / prob
    assert math.isclose(kelly_fraction_from_odds(prob, fair_b), 0.0, abs_tol=1e-9)


@given(prob=_prob, b=_odds)
@settings(max_examples=200)
def test_kelly_sign_matches_expected_value_sign(prob, b):
    # f* > 0 iff the bet has positive expected value (p*b > q); f* < 0 iff
    # negative EV (p*b < q) -- Kelly should never recommend a positive
    # stake on a bet with negative expected value, or vice versa.
    q = 1.0 - prob
    ev_sign = (prob * b) - q
    f_star = kelly_fraction_from_odds(prob, b)
    if ev_sign > 1e-9:
        assert f_star > 0
    elif ev_sign < -1e-9:
        assert f_star < 0
    else:
        assert math.isclose(f_star, 0.0, abs_tol=1e-6)


@given(prob=_prob, b1=_odds, b2=_odds)
@settings(max_examples=200)
def test_kelly_fraction_increasing_in_odds(prob, b1, b2):
    # Holding probability fixed, better offered odds (higher b) should
    # never recommend a smaller stake.
    assume(b1 < b2)
    assert kelly_fraction_from_odds(prob, b1) <= kelly_fraction_from_odds(prob, b2) + 1e-9


@given(p1=_prob, p2=_prob, b=_odds)
@settings(max_examples=200)
def test_kelly_fraction_increasing_in_probability(p1, p2, b):
    # Holding odds fixed, a higher win probability should never recommend
    # a smaller stake.
    assume(p1 < p2)
    assert kelly_fraction_from_odds(p1, b) <= kelly_fraction_from_odds(p2, b) + 1e-9


@given(predicted_prob=_prob, market_price_yes=_price)
@settings(max_examples=200)
def test_binary_market_side_matches_edge_sign(predicted_prob, market_price_yes):
    sizing = kelly_fraction_binary_market(predicted_prob, market_price_yes)
    if predicted_prob > market_price_yes:
        assert sizing.side == "yes"
    elif predicted_prob < market_price_yes:
        assert sizing.side == "no"
    else:
        assert sizing.side == "none"


@given(predicted_prob=_prob, market_price_yes=_price)
@settings(max_examples=200)
def test_binary_market_edge_is_the_absolute_probability_gap(predicted_prob, market_price_yes):
    sizing = kelly_fraction_binary_market(predicted_prob, market_price_yes)
    assert math.isclose(sizing.edge, abs(predicted_prob - market_price_yes), abs_tol=1e-9)


@given(predicted_prob=_prob, market_price_yes=_price)
@settings(max_examples=200)
def test_binary_market_complement_is_a_mirror_image(predicted_prob, market_price_yes):
    # A "NO" contract is just a "YES" contract on the complementary event
    # (buying NO at price p is economically the same position as buying
    # YES at 1-p on the opposite outcome) -- sizing the complementary
    # market must swap "yes"<->"no" and produce the identical fraction.
    # Excludes the sub-ULP-scale "edge" region (predicted_prob and
    # market_price_yes differing by ~1e-16): kelly_fraction_binary_market's
    # `edge_yes == 0.0` check is exact, and `1.0 - x` doesn't invert
    # differences that small symmetrically, so the side can flip between
    # forward and complement at a scale with zero economic meaning --
    # discovered by Hypothesis at predicted_prob=0.001,
    # market_price_yes=0.0010000000000000002.
    assume(abs(predicted_prob - market_price_yes) > 1e-9)
    forward = kelly_fraction_binary_market(predicted_prob, market_price_yes)
    complement = kelly_fraction_binary_market(1.0 - predicted_prob, 1.0 - market_price_yes)

    swap = {"yes": "no", "no": "yes", "none": "none"}
    assert complement.side == swap[forward.side]
    assert math.isclose(complement.edge, forward.edge, abs_tol=1e-9)
    assert math.isclose(complement.full_kelly_fraction, forward.full_kelly_fraction, abs_tol=1e-6)


@given(predicted_prob=_prob, market_price_yes=_price)
@settings(max_examples=200)
def test_binary_market_fraction_never_negative(predicted_prob, market_price_yes):
    # kelly_fraction_binary_market always picks the positive-edge side (or
    # "none") -- full_kelly_fraction should never come back negative.
    sizing = kelly_fraction_binary_market(predicted_prob, market_price_yes)
    assert sizing.full_kelly_fraction >= 0.0

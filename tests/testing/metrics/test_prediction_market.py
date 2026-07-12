import numpy as np
import pytest

from testing.metrics.prediction_market import (
    CalibrationCurve,
    KellySizing,
    brier_score,
    calibration_curve,
    edge_vs_market,
    fractional_kelly,
    kelly_fraction_binary_market,
    kelly_fraction_from_odds,
    log_loss,
)

# --- Brier score / log loss ------------------------------------------------


def test_brier_score_perfect_predictions_is_zero():
    assert brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0


def test_brier_score_always_wrong_confident_predictions_is_one():
    assert brier_score([1.0, 1.0], [0, 0]) == 1.0


def test_brier_score_always_predict_half_matches_known_reference_value():
    # An always-0.5 forecaster scores exactly 0.25 against any realized
    # outcomes -- (0.5-1)^2 == (0.5-0)^2 == 0.25 regardless of outcome mix.
    # Independently derived (not copied from the implementation): mean of
    # four (0.5 - o)^2 terms is always 0.25 since each term is 0.25.
    assert np.isclose(brier_score([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]), 0.25)


def test_brier_score_matches_hand_derived_mixed_case():
    # p=[0.9, 0.2, 0.6], o=[1, 0, 0] -> (0.1^2 + 0.2^2 + 0.6^2)/3
    expected = (0.1**2 + 0.2**2 + 0.6**2) / 3
    assert np.isclose(brier_score([0.9, 0.2, 0.6], [1, 0, 0]), expected)


def test_brier_score_rejects_out_of_range_probability():
    with pytest.raises(ValueError, match="0, 1"):
        brier_score([1.5], [1])


def test_brier_score_rejects_non_binary_outcome():
    with pytest.raises(ValueError, match="binary"):
        brier_score([0.5], [0.5])


def test_brier_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        brier_score([0.5, 0.6], [1])


def test_log_loss_matches_hand_derived_value_for_even_predictions():
    # p=0.5 for every observation -> -log(0.5) regardless of outcome,
    # independently derived from the log-loss definition, not the
    # implementation.
    expected = -np.log(0.5)
    assert np.isclose(log_loss([0.5, 0.5, 0.5], [1, 0, 1]), expected)


def test_log_loss_perfect_predictions_is_near_zero():
    assert log_loss([1.0, 0.0], [1, 0]) < 1e-10


def test_log_loss_confidently_wrong_prediction_is_large_but_finite():
    # A single 0.999-confident call that resolves the other way should cost
    # roughly -log(0.001) ~= 6.9, and must not be inf/nan thanks to clipping.
    loss = log_loss([0.999], [0])
    assert np.isfinite(loss)
    assert loss > 5.0


def test_log_loss_exact_zero_and_one_predictions_do_not_produce_infinite_loss():
    # Without clipping, log(0) would be -inf here (predicted 0.0, resolved
    # YES) -- confirms the eps clip actually protects the aggregate.
    loss = log_loss([0.0, 1.0], [1, 1])
    assert np.isfinite(loss)


# --- Calibration curve -----------------------------------------------------


def test_calibration_curve_well_calibrated_bin_matches_predicted_and_empirical():
    # 10 predictions at 0.7, 7 resolve YES, 3 resolve NO -> bin [0.7, 0.8)
    # should show predicted_mean=0.7, empirical_freq=0.7 exactly.
    preds = [0.7] * 10
    outcomes = [1] * 7 + [0] * 3
    curve = calibration_curve(preds, outcomes, n_bins=10)
    assert isinstance(curve, CalibrationCurve)

    occupied = [b for b in curve.bins if b.count > 0]
    assert len(occupied) == 1
    bucket = occupied[0]
    assert bucket.count == 10
    assert np.isclose(bucket.predicted_mean, 0.7)
    assert np.isclose(bucket.empirical_freq, 0.7)
    assert np.isclose(curve.calibration_error, 0.0)


def test_calibration_curve_miscalibrated_forecaster_shows_nonzero_error():
    # Predicts 0.9 every time but only wins half the time -> calibration_error
    # should be exactly |0.9 - 0.5| = 0.4.
    preds = [0.9] * 10
    outcomes = [1] * 5 + [0] * 5
    curve = calibration_curve(preds, outcomes, n_bins=10)
    assert np.isclose(curve.calibration_error, 0.4)


def test_calibration_curve_prediction_of_exactly_one_lands_in_last_bin():
    curve = calibration_curve([1.0], [1], n_bins=10)
    last_bin = curve.bins[-1]
    assert last_bin.count == 1
    assert np.isclose(last_bin.bin_low, 0.9)
    assert np.isclose(last_bin.bin_high, 1.0)


def test_calibration_curve_empty_bins_report_none_not_nan():
    curve = calibration_curve([0.05], [0], n_bins=10)
    empty_bins = [b for b in curve.bins if b.count == 0]
    assert len(empty_bins) == 9
    assert all(b.predicted_mean is None and b.empirical_freq is None for b in empty_bins)


def test_calibration_curve_rejects_zero_bins():
    with pytest.raises(ValueError, match="n_bins"):
        calibration_curve([0.5], [1], n_bins=0)


# --- Edge vs. market implied probability -----------------------------------


def test_edge_vs_market_positive_when_prediction_exceeds_price():
    edges = edge_vs_market([0.7, 0.3], [0.5, 0.5])
    assert np.allclose(edges, [0.2, -0.2])


def test_edge_vs_market_rejects_degenerate_market_price():
    with pytest.raises(ValueError, match="market_prices"):
        edge_vs_market([0.5], [1.0])


# --- Kelly criterion: textbook cases ----------------------------------------


def test_kelly_fraction_from_odds_matches_canonical_textbook_example():
    # The standard worked example cited across Kelly-criterion references
    # (e.g. Wikipedia's "Statement" section, Thorp's applied papers): a
    # bettor with a 60% win probability at even-money (1:1) odds should bet
    # 20% of bankroll. f* = p - q/b = 0.6 - 0.4/1 = 0.2.
    assert np.isclose(kelly_fraction_from_odds(prob=0.6, b=1.0), 0.2)


def test_kelly_fraction_from_odds_matches_second_textbook_example():
    # 3:1 net odds (b=3) at a 50% win probability: f* = p - q/b
    # = 0.5 - 0.5/3 = 1/3.
    assert np.isclose(kelly_fraction_from_odds(prob=0.5, b=3.0), 1.0 / 3.0)


def test_kelly_fraction_from_odds_zero_edge_gives_zero_fraction():
    # Fair-odds bet (implied prob == true prob) should never be sized --
    # b=1 means implied prob 0.5, matching true prob 0.5.
    assert np.isclose(kelly_fraction_from_odds(prob=0.5, b=1.0), 0.0)


def test_kelly_fraction_from_odds_negative_edge_gives_negative_fraction():
    # A bettor who is wrong more than the odds justify gets a negative
    # Kelly fraction (i.e. "don't take this bet"), not clamped to zero here
    # -- clamping is `kelly_fraction_binary_market`'s job once it knows
    # which side of the market to attribute the fraction to.
    assert kelly_fraction_from_odds(prob=0.4, b=1.0) < 0.0


def test_kelly_fraction_from_odds_rejects_invalid_prob():
    with pytest.raises(ValueError, match="prob"):
        kelly_fraction_from_odds(prob=1.5, b=1.0)


def test_kelly_fraction_from_odds_rejects_non_positive_b():
    with pytest.raises(ValueError, match="b"):
        kelly_fraction_from_odds(prob=0.5, b=0.0)


# --- Kelly sizing for a binary prediction-market contract -------------------


def test_kelly_fraction_binary_market_matches_textbook_via_even_money_equivalent():
    # A market price of 0.5 is the prediction-market equivalent of even-money
    # (1:1) odds: b = (1 - 0.5) / 0.5 = 1.0. A 60% true-probability forecast
    # should therefore reproduce the same canonical f*=0.2 as the odds-based
    # textbook case above, cross-checking the two formulas agree.
    sizing = kelly_fraction_binary_market(predicted_prob=0.6, market_price_yes=0.5)
    assert sizing.side == "yes"
    assert np.isclose(sizing.edge, 0.1)
    assert np.isclose(sizing.full_kelly_fraction, 0.2)


def test_kelly_fraction_binary_market_picks_no_side_when_market_overprices_yes():
    # Market prices YES and NO both at 0.5 (even-money-equivalent on either
    # leg), but the forecaster believes YES is only 40% likely (NO is 60%
    # likely) -- the correct side is NO. NO's own bet shape is then
    # identical to the canonical 60%-probability/even-money textbook case,
    # so it should also reproduce f*=0.2, cross-checking the NO-side
    # derivation against the same reference value as the YES-side test above.
    sizing = kelly_fraction_binary_market(predicted_prob=0.4, market_price_yes=0.5)
    assert sizing.side == "no"
    assert np.isclose(sizing.edge, 0.1)
    assert np.isclose(sizing.full_kelly_fraction, 0.2)


def test_kelly_fraction_binary_market_no_edge_returns_none_side_and_zero_fraction():
    sizing = kelly_fraction_binary_market(predicted_prob=0.35, market_price_yes=0.35)
    assert sizing == KellySizing(side="none", edge=0.0, full_kelly_fraction=0.0)


def test_kelly_fraction_binary_market_rejects_degenerate_market_price():
    with pytest.raises(ValueError, match="market_price_yes"):
        kelly_fraction_binary_market(predicted_prob=0.5, market_price_yes=0.0)


def test_kelly_fraction_binary_market_rejects_invalid_predicted_prob():
    with pytest.raises(ValueError, match="predicted_prob"):
        kelly_fraction_binary_market(predicted_prob=-0.1, market_price_yes=0.5)


# --- Fractional Kelly --------------------------------------------------------


def test_fractional_kelly_half_kelly_halves_the_full_fraction():
    sizing = KellySizing(side="yes", edge=0.1, full_kelly_fraction=0.2)
    assert np.isclose(fractional_kelly(sizing, fraction=0.5), 0.1)


def test_fractional_kelly_defaults_to_half_kelly():
    sizing = KellySizing(side="yes", edge=0.1, full_kelly_fraction=0.4)
    assert np.isclose(fractional_kelly(sizing), 0.2)


def test_fractional_kelly_caps_at_full_bankroll_by_default():
    # A large full-Kelly fraction (e.g. from a near-certain edge) should not
    # size past 100% of bankroll even at fraction=1.0.
    sizing = KellySizing(side="yes", edge=0.9, full_kelly_fraction=1.5)
    assert fractional_kelly(sizing, fraction=1.0) == 1.0


def test_fractional_kelly_respects_custom_cap():
    sizing = KellySizing(side="yes", edge=0.5, full_kelly_fraction=0.5)
    assert fractional_kelly(sizing, fraction=1.0, cap=0.25) == 0.25


def test_fractional_kelly_rejects_invalid_fraction():
    sizing = KellySizing(side="yes", edge=0.1, full_kelly_fraction=0.2)
    with pytest.raises(ValueError, match="fraction"):
        fractional_kelly(sizing, fraction=0.0)


def test_fractional_kelly_rejects_a_cap_of_exactly_zero():
    """`cap <= 0.0` -- mutation testing (2026-07-12) found the boundary
    survives being tightened to `< 0.0`. cap=0.0 is nonsensical (caps every
    stake to zero regardless of edge) and must raise, not silently succeed."""
    sizing = KellySizing(side="yes", edge=0.1, full_kelly_fraction=0.2)
    with pytest.raises(ValueError, match="cap"):
        fractional_kelly(sizing, cap=0.0)

import math

import pandas as pd
import pytest

from testing.engine.leg import Leg
from testing.engine.payoff import (
    max_loss_at_expiration,
    scenario_based_margin,
    scenario_based_margin_with_vol_shock,
)

_EXDATE = pd.Timestamp("2020-02-15")


def _leg(strike, right, side, optionid=None, weight=1.0, impl_volatility=None):
    return Leg(
        optionid=optionid or f"{right}{side}{strike}", exdate=_EXDATE, strike_price=strike,
        delta=0.0, best_bid=0.0, best_offer=0.0, right=right, side=side, weight=weight,
        impl_volatility=impl_volatility,
    )


def test_matches_old_width_formula_for_a_plain_vertical():
    # short put 95 / long put 90, width 5, entry_credit 1.8 -> old formula: 5 - 1.8 = 3.2
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=1.8) == 3.2


def test_matches_old_width_formula_for_iron_condor_wider_call_side():
    # put width 5, call width 10 -> old formula used the wider side: 10 - 2.8 = 7.2
    legs = [
        _leg(95, "P", "short"), _leg(90, "P", "long"),
        _leg(105, "C", "short"), _leg(115, "C", "long"),
    ]
    assert max_loss_at_expiration(legs, entry_credit=2.8) == 7.2


def test_matches_old_zero_width_convention_for_same_strike_calendar():
    # same strike both legs (different real-world expiries, same-expiry payoff
    # approximation collapses identically here) -> old formula: 0 - entry_credit
    legs = [_leg(100, "P", "short"), _leg(100, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=-2.0) == 2.0  # debit paid


def test_three_leg_structure_finds_the_elbow_not_the_outer_span():
    # short put 95 / long put 90 / long tail put 85 (gamma_hedged_vrp shape).
    # Old width heuristic would've used the outer span (95-85=10). True worst
    # point is at the middle strike (90): loss = (95-90) - entry_credit = 5 - 0.5 = 4.5,
    # NOT the outer-span answer of 10 - 0.5 = 9.5.
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long"), _leg(85, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=0.5) == 4.5


def test_naked_short_put_is_bounded_at_spot_zero_not_zero_width():
    # Old heuristic returned width=0 for a single naked leg (dangerously
    # implies ~zero risk). True worst case is at spot=0: loss = strike - credit.
    legs = [_leg(50, "P", "short")]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == 49.0


def test_naked_short_call_is_unbounded():
    legs = [_leg(50, "C", "short")]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == math.inf


def test_covered_call_not_naked_is_bounded():
    # short call covered by a long call further out -> bounded vertical, not inf.
    legs = [_leg(50, "C", "short"), _leg(60, "C", "long")]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == 9.0


def test_long_only_max_loss_is_the_debit_paid():
    legs = [_leg(50, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=-2.0) == 2.0


def test_ratio_backspread_1_short_2_long_further_otm():
    # sell 1 put @95, buy 2 puts @85 (further OTM), net small debit/credit.
    # Worst point is the elbow at 85 (short ITM by 10, longs still OTM):
    # loss = 10 - entry_credit.
    legs = [_leg(95, "P", "short"), _leg(85, "P", "long"), _leg(85, "P", "long")]
    assert max_loss_at_expiration(legs, entry_credit=0.2) == 9.8


def test_naked_risk_reversal_short_put_long_call_no_shared_strike():
    # risk_reversal shape: short put 90, long call 110, near-zero net credit.
    # Old heuristic: both P and C groups have <2 legs -> width 0 either way ->
    # silently implies ~zero risk for what's actually a large bounded loss.
    legs = [_leg(90, "P", "short"), _leg(110, "C", "long")]
    # worst point is spot=0: put intrinsic=90, call intrinsic=0 -> loss = 90 - credit
    assert max_loss_at_expiration(legs, entry_credit=0.1) == 89.9


def test_empty_legs_returns_zero():
    assert max_loss_at_expiration([], entry_credit=0.0) == 0.0


def test_default_weight_is_one_and_unchanged_from_pre_weight_behavior():
    # Same fixture as test_matches_old_width_formula_for_a_plain_vertical --
    # confirms the weight field (added for variance_swap) is a true no-op
    # for every existing leg-count-based caller.
    legs = [_leg(95, "P", "short"), _leg(90, "P", "long")]
    assert all(leg.weight == 1.0 for leg in legs)
    assert max_loss_at_expiration(legs, entry_credit=1.8) == 3.2


def test_weight_scales_each_legs_contribution():
    # Doubling one leg's weight doubles its contribution to the worst-case
    # loss, independent of the other leg's (default 1.0) weight.
    legs = [_leg(95, "P", "short", weight=2.0), _leg(90, "P", "long")]
    # worst point is spot=0 (put side, uncapped by an S->inf check): loss =
    # 2*95 - 90 - entry_credit = 190-90-1.8 = 98.2
    assert max_loss_at_expiration(legs, entry_credit=1.8) == 98.2


def test_weighted_short_calls_exceeding_weighted_long_calls_is_unbounded():
    # 1 short call at weight 2.0 vs 1 long call at weight 1.0 -- weighted
    # short exposure (2.0) exceeds weighted long exposure (1.0), so the net
    # slope as spot -> infinity is still negative -> unbounded, even though
    # leg *counts* are equal (1 short, 1 long).
    legs = [_leg(50, "C", "short", weight=2.0), _leg(60, "C", "long", weight=1.0)]
    assert max_loss_at_expiration(legs, entry_credit=1.0) == math.inf


def test_weighted_long_call_fully_covering_short_call_stays_bounded():
    # Long call's weight (2.0) now covers the short call's (1.0) -- bounded.
    legs = [_leg(50, "C", "short", weight=1.0), _leg(60, "C", "long", weight=2.0)]
    assert max_loss_at_expiration(legs, entry_credit=1.0) != math.inf


class TestScenarioBasedMargin:
    def test_bounded_where_max_loss_at_expiration_is_infinite(self):
        # Naked short call -- max_loss_at_expiration is genuinely inf, but a
        # realistic +/-15% price move gives a real, finite margin number.
        legs = [_leg(2000, "C", "short")]
        assert max_loss_at_expiration(legs, entry_credit=10.0) == math.inf
        margin = scenario_based_margin(legs, entry_credit=10.0, spot=2000.0, band_pct=0.15)
        assert margin == pytest.approx(2000 * 0.15 - 10.0)  # intrinsic at the +15% edge, minus credit

    def test_matches_max_loss_at_expiration_for_a_narrow_bounded_spread(self):
        # A tight vertical fully inside the scenario band gives the exact
        # same answer either way -- the band doesn't change a genuinely
        # bounded structure's worst point if that point already sits well
        # inside +/-15%.
        legs = [_leg(1900, "P", "short"), _leg(1850, "P", "long")]
        expiration_max_loss = max_loss_at_expiration(legs, entry_credit=1.8)
        margin = scenario_based_margin(legs, entry_credit=1.8, spot=2000.0, band_pct=0.15)
        assert margin == expiration_max_loss

    def test_worst_point_outside_the_band_is_clamped_to_the_band_edge(self):
        # Short put struck far below the band's lower edge (spot*0.85) --
        # its true worst point (spot=0) is unreachable within a realistic
        # +/-15% move, so margin uses the band edge instead, a materially
        # smaller (more realistic) number than the unbounded max loss.
        legs = [_leg(500, "P", "short")]  # deep OTM relative to spot=2000
        expiration_max_loss = max_loss_at_expiration(legs, entry_credit=1.0)
        margin = scenario_based_margin(legs, entry_credit=1.0, spot=2000.0, band_pct=0.15)
        assert margin < expiration_max_loss

    def test_weight_scales_margin(self):
        legs = [_leg(2000, "C", "short", weight=10.0)]
        margin = scenario_based_margin(legs, entry_credit=10.0, spot=2000.0, band_pct=0.15)
        assert margin == pytest.approx(2000 * 0.15 * 10.0 - 10.0)

    def test_empty_legs_returns_zero(self):
        assert scenario_based_margin([], entry_credit=0.0, spot=2000.0) == 0.0

    def test_floored_at_entry_credit_when_stress_scenario_shows_zero_loss(self):
        # Deep-OTM short put -- even the band's lower edge doesn't bring it
        # in the money, so the raw stress-scenario computation is a genuine
        # 0.0 loss. Real margin rules never let a live short position post
        # less than the premium collected; without the floor this would be
        # 0.0, which also breaks any P&L/margin ratio (division by zero).
        legs = [_leg(500, "P", "short")]  # spot=2000, band lower edge = 1700 -- still far OTM
        margin = scenario_based_margin(legs, entry_credit=3.5, spot=2000.0, band_pct=0.15)
        assert margin == 3.5

    def test_floor_does_not_override_a_genuinely_larger_stress_loss(self):
        legs = [_leg(1900, "P", "short"), _leg(1850, "P", "long")]
        margin = scenario_based_margin(legs, entry_credit=1.8, spot=2000.0, band_pct=0.15)
        assert margin == 48.2  # the real stress loss (50 width - 1.8 credit), not the tiny credit floor


class TestScenarioBasedMarginWithVolShock:
    def test_at_least_as_large_as_intrinsic_only_margin(self):
        # Adding time value on top of intrinsic value at the shocked spot
        # can only ever raise (or match) the worst-case loss for a short
        # position, never lower it.
        legs = [_leg(2000, "C", "short", impl_volatility=0.20)]
        intrinsic_margin = scenario_based_margin(legs, entry_credit=10.0, spot=2000.0, band_pct=0.15)
        vol_shock_margin = scenario_based_margin_with_vol_shock(
            legs, entry_credit=10.0, spot=2000.0, time_to_expiry_years=0.1,
        )
        assert vol_shock_margin >= intrinsic_margin

    def test_defined_risk_spread_can_legitimately_price_below_the_intrinsic_margin(self):
        # NOT a bug, despite looking like one at first (see payoff.py's
        # docstring): a live mark-to-market stress test with real
        # time-to-expiry left is a different question from "worst case at
        # expiration" -- a defined-risk vertical's two legs' time value
        # move together and largely net out near the theoretical worst
        # point, so it can genuinely not yet have realized that terminal
        # worst case. Confirmed against a realistic ~7%/~12%-OTM put
        # spread (vrp_harvester's own rough shape) before accepting this
        # as expected rather than chasing it as a bug.
        legs = [_leg(1860, "P", "short", impl_volatility=0.18), _leg(1760, "P", "long", impl_volatility=0.22)]
        intrinsic_margin = scenario_based_margin(legs, entry_credit=5.0, spot=2000.0, band_pct=0.15)
        vol_shock_margin = scenario_based_margin_with_vol_shock(
            legs, entry_credit=5.0, spot=2000.0, time_to_expiry_years=0.1,
        )
        assert vol_shock_margin < intrinsic_margin

    def test_higher_vol_multiplier_grid_increases_margin(self):
        legs = [_leg(2000, "P", "short", impl_volatility=0.20)]
        narrow_vol_margin = scenario_based_margin_with_vol_shock(
            legs, entry_credit=10.0, spot=2000.0, time_to_expiry_years=0.1,
            vol_multipliers=(1.0,),
        )
        wide_vol_margin = scenario_based_margin_with_vol_shock(
            legs, entry_credit=10.0, spot=2000.0, time_to_expiry_years=0.1,
            vol_multipliers=(1.0, 2.0),
        )
        assert wide_vol_margin >= narrow_vol_margin

    def test_missing_implied_vol_falls_back_to_intrinsic_for_that_leg(self):
        legs = [_leg(2000, "C", "short", impl_volatility=None)]
        margin = scenario_based_margin_with_vol_shock(
            legs, entry_credit=10.0, spot=2000.0, time_to_expiry_years=0.1,
            price_shocks=(0.15,), vol_multipliers=(1.0,),
        )
        # No vol -> pure intrinsic at the shocked spot: (2000*1.15 - 2000) - 10
        assert margin == pytest.approx(2000 * 0.15 - 10.0)

    def test_empty_legs_returns_zero(self):
        assert scenario_based_margin_with_vol_shock([], entry_credit=0.0, spot=2000.0, time_to_expiry_years=0.1) == 0.0

    def test_floored_at_entry_credit(self):
        # Deep OTM short put, even under a severe vol shock, still shows a
        # genuine profit at every scenario point -- floor applies.
        legs = [_leg(200, "P", "short", impl_volatility=0.30)]
        margin = scenario_based_margin_with_vol_shock(
            legs, entry_credit=2.5, spot=2000.0, time_to_expiry_years=0.1,
        )
        assert margin == 2.5

    def test_weight_scales_margin(self):
        legs = [_leg(2000, "C", "short", impl_volatility=0.20, weight=5.0)]
        unweighted = scenario_based_margin_with_vol_shock(
            [_leg(2000, "C", "short", impl_volatility=0.20, weight=1.0)],
            entry_credit=2.0, spot=2000.0, time_to_expiry_years=0.1,
        )
        weighted = scenario_based_margin_with_vol_shock(legs, entry_credit=10.0, spot=2000.0, time_to_expiry_years=0.1)
        # Roughly 5x (not exact due to the entry_credit floor interacting
        # differently at each scale) -- check the dominant stress term scales.
        assert weighted > 4 * unweighted

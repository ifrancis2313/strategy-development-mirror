import math

from testing.engine.black_scholes import black_scholes_price
from testing.engine.leg import Leg

# Standard OCC/TIMS-style broad-based-index scenario scan: price moves in
# +/-5% steps out to +/-15% (the same band `scenario_based_margin` uses),
# combined with a vol-multiplier grid spanning a real vol-down move (0.7x)
# through a severe crash-level vol spike (2.0x -- VIX going from ~15 to
# ~30-45 during 2008/Volmageddon-scale events is exactly this order of
# magnitude). Evaluated as a discrete grid (not a continuous optimization)
# deliberately -- real SPAN/TIMS margin systems scan a discrete set of
# scenario points too, so this isn't a shortcut relative to the standard
# methodology, it's the methodology.
_DEFAULT_PRICE_SHOCKS = (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
_DEFAULT_VOL_MULTIPLIERS = (0.7, 1.0, 1.5, 2.0)


def _intrinsic(leg: Leg, spot: float) -> float:
    if leg.right == "P":
        return max(leg.strike_price - spot, 0.0)
    return max(spot - leg.strike_price, 0.0)


def max_loss_at_expiration(legs: list[Leg], entry_credit: float) -> float:
    """Worst-case loss at expiration, in premium terms (positive = a loss,
    same sign convention `contracts_for_entry`'s old `_max_loss_width -
    entry_credit` expression used), for an arbitrary same-expiry leg
    combination -- however many legs, any mix of rights, any short/long
    ratio. Returns `math.inf` when the structure has genuinely unbounded
    risk (an uncovered short call).

    Replaces the old group-by-right/take-the-span heuristic
    (`_max_loss_width`), which only happened to be correct for a plain
    vertical or a symmetric iron condor (its two documented, tested cases)
    and silently produced wrong answers otherwise: an under-count for any
    same-right structure with more than 2 strikes (e.g. gamma_hedged_vrp's
    3-put structure, whose true worst point is the middle "elbow" strike,
    not the outer span -- confirmed this matches that strategy's own
    hand-worked payoff-region proof), and a *dangerous* near-zero-or-1
    "width" for a naked single leg per right (risk_reversal's short put +
    long call: 0 shared-right legs of either right, old code returned a
    width of 0.0 for a genuinely large, bounded-but-not-tiny risk).

    Algorithm: P&L at expiration as a function of spot `S` is piecewise
    linear with kinks only at each leg's strike (intrinsic value is
    piecewise linear in `S`, kinked at the strike; a sum of piecewise
    linear functions is piecewise linear, kinked at the union of their
    kinks). A piecewise linear function's extrema occur only at its kinks
    or at the domain boundary (S -> 0, S -> infinity) -- so the worst case
    is exactly the minimum of PnL(S) evaluated at S=0 and at every leg's
    strike, UNLESS the net exposure as S -> infinity is unbounded downward
    (more short calls than long calls -- nothing caps a naked short call's
    loss as spot keeps rising). Puts never need an S -> infinity check:
    every put's intrinsic value is already 0 for S at or above its strike,
    so S=0 -- a real, evaluable point -- is always the worst point puts
    can produce, not a boundary that needs a separate unbounded-risk check.

    For a plain 2-leg vertical or a standard iron condor (put spread with
    strikes below spot, call spread with strikes above), this reduces to
    exactly the same number the old width-based formula produced --
    verified by hand and by the regression tests in
    `tests/core/test_payoff.py` -- so swapping it into `contracts_for_entry`
    changes no existing strategy's sizing.
    """
    if not legs:
        return 0.0

    # Weighted, not counted -- as S -> infinity, each call leg's slope is
    # +/-weight (long/short), so the net exposure is unbounded downward
    # only if the weighted short-call slope exceeds the weighted long-call
    # slope. For every pre-variance_swap strategy every weight is 1.0, so
    # this is identical to the old leg-count comparison.
    weighted_short_calls = sum(leg.weight for leg in legs if leg.right == "C" and leg.side == "short")
    weighted_long_calls = sum(leg.weight for leg in legs if leg.right == "C" and leg.side == "long")
    if weighted_short_calls > weighted_long_calls:
        return math.inf

    candidate_spots = {0.0} | {leg.strike_price for leg in legs}

    def pnl_at(spot: float) -> float:
        cost_to_close = sum(
            (_intrinsic(leg, spot) if leg.side == "short" else -_intrinsic(leg, spot)) * leg.weight
            for leg in legs
        )
        return entry_credit - cost_to_close

    worst_pnl = min(pnl_at(spot) for spot in candidate_spots)
    return max(-worst_pnl, 0.0)


def scenario_based_margin(legs: list[Leg], entry_credit: float, spot: float, band_pct: float = 0.15) -> float:
    """Real-broker-style margin estimate: worst-case loss (intrinsic-value
    terms) across a bounded, realistic price-move band around the current
    spot/forward, rather than `max_loss_at_expiration`'s unbounded
    [0, infinity) evaluation.

    Named source: this is a simplified version of OCC/CBOE's customer
    portfolio-margin methodology for broad-based cash-settled index
    options (TIMS and its successors) -- real margin for a position like
    this is set by stress-testing across a scenario grid, standardly
    +/-15% for a broad-based index (`band_pct` default), NOT by the
    theoretical worst case at spot=0 or spot=infinity. This is the
    metric that makes `vrp_harvester` (a defined-risk vertical, already
    bounded well inside any realistic move) and `variance_swap` (an
    unbounded naked strip, where `max_loss_at_expiration` returns `inf`
    and is therefore useless as a capital comparison) actually
    comparable: both get a real, finite, capital-realistic number.

    Simplification, flagged rather than hidden: intrinsic-value-only, no
    volatility-shock component. Real OCC/TIMS methodology also shocks
    implied vol (a real crash brings a vol spike, which raises an option's
    *extrinsic* value beyond pure intrinsic at the shocked spot) -- this
    would require repricing the whole strip off a shocked vol surface, a
    materially bigger build this doesn't attempt. This UNDERSTATES the true
    worst-case margin, especially for `variance_swap` (far more vega
    exposure than a 2-leg spread) -- so a margin-normalized comparison
    using this function is, if anything, still favorable to `variance_swap`
    relative to a true portfolio-margin number.

    Same piecewise-linear-extrema logic as `max_loss_at_expiration`
    (candidate worst points are the band's own endpoints and any leg
    strike that falls inside the band -- a kink outside the band can never
    be the worst point since the function is monotonic between the last
    interior kink and the band edge), just restricted to a bounded domain
    instead of every leg's strike plus spot=0/infinity.

    Floored at `abs(entry_credit)` -- real margin rules (Reg-T and OCC
    portfolio margin alike) never let a live short-option position post
    less than the premium collected, even on a day the stress scenario
    shows zero loss (a deep-OTM short spread on a low-vol day). Without
    this floor, a well-OTM trade can price at exactly 0.0 margin, which
    is not just unrealistic but breaks any margin-normalized return
    calculation (division by zero) for callers computing P&L/margin.
    """
    if not legs:
        return 0.0

    lower, upper = spot * (1 - band_pct), spot * (1 + band_pct)
    candidate_spots = {lower, upper} | {leg.strike_price for leg in legs if lower <= leg.strike_price <= upper}

    def pnl_at(s: float) -> float:
        cost_to_close = sum(
            (_intrinsic(leg, s) if leg.side == "short" else -_intrinsic(leg, s)) * leg.weight
            for leg in legs
        )
        return entry_credit - cost_to_close

    worst_pnl = min(pnl_at(s) for s in candidate_spots)
    return max(-worst_pnl, abs(entry_credit))


def scenario_based_margin_with_vol_shock(
    legs: list[Leg],
    entry_credit: float,
    spot: float,
    time_to_expiry_years: float,
    price_shocks: tuple[float, ...] = _DEFAULT_PRICE_SHOCKS,
    vol_multipliers: tuple[float, ...] = _DEFAULT_VOL_MULTIPLIERS,
) -> float:
    """`scenario_based_margin`'s intrinsic-value-only stress test, extended
    with the volatility-shock dimension real OCC/TIMS broad-based-index
    margin methodology also scans -- closes the gap that function's own
    docstring flagged (a real crash brings a vol spike, which raises an
    option's *extrinsic* value beyond pure intrinsic at the shocked spot;
    intrinsic-only understates the true worst case, more so for a
    vega-heavy structure like `variance_swap` than a simple spread).

    Each leg is repriced via Black-Scholes at every (price_shock,
    vol_multiplier) grid point -- `spot * (1 + price_shock)` and
    `leg.impl_volatility * vol_multiplier` (each leg's OWN entry-day IV,
    not a single shared vol level, so the position's actual vol skew shape
    is preserved under the shock, not flattened) -- with `time_to_expiry_years`
    held fixed (a stress test "as of today", not a simulation of time
    passing). The worst (most negative) P&L across the full grid is the
    margin, same floor-at-premium-collected convention as
    `scenario_based_margin`.

    A leg with no recorded `impl_volatility` (a real, if uncommon, chain
    gap) falls back to that leg's intrinsic value at each shocked spot,
    unaffected by the vol dimension for that leg specifically -- explicit
    per-leg degradation rather than rejecting the whole position or
    silently coercing a fabricated vol number.

    Discrete grid, not a continuous optimization -- matches how real
    SPAN/TIMS scenario systems actually scan (see `_DEFAULT_PRICE_SHOCKS`/
    `_DEFAULT_VOL_MULTIPLIERS` module constants for the specific grid and
    its rationale). Only the fixed, realistic percentage grid is scanned --
    NOT also each leg's exact strike the way `scenario_based_margin` does,
    deliberately: a first version tried adding each leg's strike-implied
    price_shock (to close what looked like a gap vs. that function's
    exact-kink evaluation), which quietly broke the realistic bounding this
    whole approach exists for -- a leg struck 90% away from spot produced a
    "-90% crash" scenario point, wildly outside any real broad-based-index
    margin methodology's scan range, inflating margin for deep-OTM legs
    that pose negligible real risk. Caught by a test with a genuinely
    deep-OTM leg before trusting the output, reverted.

    A real, expected (not a bug) consequence of scanning a fixed realistic
    grid rather than exact kinks: this can legitimately price a *defined-
    risk spread* (both legs' time value moving together, largely netting
    out near the theoretical worst point) at a SMALLER margin than
    `scenario_based_margin`'s at-expiration evaluation -- a live
    mark-to-market stress test with real time-to-expiry remaining isn't
    the same question as "what's the worst case if this were expiring
    right now," and a spread that hasn't reached expiration genuinely
    hasn't realized its terminal worst case yet. The two functions answer
    different, both legitimate, questions; one is not simply a refinement
    of the other, and neither should be assumed to bound the other for a
    hedged (spread) structure. For a naked, unhedged leg (nothing to net
    against), vol-shock margin still comes out at or above the intrinsic-
    only number, as expected.
    """
    if not legs:
        return 0.0

    def leg_value(leg: Leg, shocked_spot: float, vol_multiplier: float) -> float:
        if leg.impl_volatility is None or leg.impl_volatility <= 0:
            return _intrinsic(leg, shocked_spot)
        return black_scholes_price(
            shocked_spot, leg.strike_price, time_to_expiry_years,
            leg.impl_volatility * vol_multiplier, leg.right,
        )

    def pnl_at(price_shock: float, vol_multiplier: float) -> float:
        shocked_spot = spot * (1 + price_shock)
        cost_to_close = sum(
            (leg_value(leg, shocked_spot, vol_multiplier) if leg.side == "short" else -leg_value(leg, shocked_spot, vol_multiplier))
            * leg.weight
            for leg in legs
        )
        return entry_credit - cost_to_close

    worst_pnl = min(pnl_at(ps, vm) for ps in price_shocks for vm in vol_multipliers)
    return max(-worst_pnl, abs(entry_credit))

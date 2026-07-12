import math

from scipy.stats import norm


def black_scholes_price(
    spot: float, strike: float, time_to_expiry_years: float, vol: float, right: str, rate: float = 0.0
) -> float:
    """European option value (Black-Scholes), for a cash-settled index
    option -- SPX/SPXW are European-exercise, cash-settled, so this is the
    correct model, not an American-exercise approximation.

    `rate` defaults to 0.0 -- same "ignore short-dated discounting, it's
    immaterial here" convention `SyntheticSpotEngine` already documents for
    this repo's typical 30-45 DTE holding periods.

    Used by `scenario_based_margin_with_vol_shock` to reprice a leg at a
    stressed (spot, vol) pair with time value still present -- a real
    portfolio-margin stress test, unlike `max_loss_at_expiration`'s pure
    intrinsic-value-at-expiration evaluation. At `time_to_expiry_years=0`
    (or `vol=0`), collapses to intrinsic value exactly (the `<=0` guard
    below), so this is a strict generalization, not a separate code path
    that could silently diverge from the intrinsic-value convention at the
    boundary.
    """
    if time_to_expiry_years <= 0 or vol <= 0:
        return max(spot - strike, 0.0) if right == "C" else max(strike - spot, 0.0)

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * time_to_expiry_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    discount = math.exp(-rate * time_to_expiry_years)

    if right == "C":
        return spot * norm.cdf(d1) - strike * discount * norm.cdf(d2)
    return strike * discount * norm.cdf(-d2) - spot * norm.cdf(-d1)

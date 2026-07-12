import pandas as pd

from testing.engine.spot import SyntheticSpotEngine


def _row(date, exdate, cp_flag, strike, bid, offer):
    return {
        "date": pd.Timestamp(date),
        "exdate": pd.Timestamp(exdate),
        "cp_flag": cp_flag,
        "strike_price": strike,
        "best_bid": bid,
        "best_offer": offer,
    }


def test_recovers_known_forward_at_single_crossing_strike():
    # Constructed so strike 100 is the exact C=P crossing point -> forward = 100.
    rows = [
        _row("2020-01-01", "2020-01-31", "C", 95, 7.0, 7.2),
        _row("2020-01-01", "2020-01-31", "P", 95, 2.0, 2.2),
        _row("2020-01-01", "2020-01-31", "C", 100, 3.0, 3.2),
        _row("2020-01-01", "2020-01-31", "P", 100, 3.0, 3.2),
        _row("2020-01-01", "2020-01-31", "C", 105, 1.0, 1.2),
        _row("2020-01-01", "2020-01-31", "P", 105, 6.0, 6.2),
    ]
    chain = pd.DataFrame(rows)

    engine = SyntheticSpotEngine(target_dte=30, n_strikes=1)
    spots = engine.compute_spot_series(chain)

    assert list(spots.index) == [pd.Timestamp("2020-01-01")]
    assert spots.iloc[0] == 100.0


def test_crossing_distance_uses_the_call_put_difference_not_their_sum():
    """`(mid_call - mid_put).abs()` -- mutation testing (2026-07-12) found
    this survives being flipped to `(mid_call + mid_put).abs()`. The
    existing single-crossing-strike test above happens to pick the same
    strike under both formulas (a symmetric fixture); this one is
    deliberately asymmetric so sum-of-premiums and |difference| disagree
    on which strike is "nearest the true crossing point":
      strike 90:  call=12.0 put=2.0 -> diff=10.0, sum=14.0
      strike 100: call=5.0  put=5.0 -> diff=0.0,  sum=10.0  (the real C=P crossing)
      strike 110: call=1.0  put=6.0 -> diff=5.0,  sum=7.0   (smallest SUM, wrong pick)
    The correct (diff-based) formula must pick strike 100 -> forward=100.0,
    not strike 110's sum-based forward=105.0.
    """
    rows = [
        _row("2020-01-01", "2020-01-31", "C", 90, 11.9, 12.1),
        _row("2020-01-01", "2020-01-31", "P", 90, 1.9, 2.1),
        _row("2020-01-01", "2020-01-31", "C", 100, 4.9, 5.1),
        _row("2020-01-01", "2020-01-31", "P", 100, 4.9, 5.1),
        _row("2020-01-01", "2020-01-31", "C", 110, 0.9, 1.1),
        _row("2020-01-01", "2020-01-31", "P", 110, 5.9, 6.1),
    ]
    chain = pd.DataFrame(rows)

    engine = SyntheticSpotEngine(target_dte=30, n_strikes=1)
    spots = engine.compute_spot_series(chain)

    assert spots.iloc[0] == 100.0


def test_picks_nearest_dte_expiry_when_multiple_available():
    near_term = [
        _row("2020-01-01", "2020-01-10", "C", 100, 3.0, 3.2),
        _row("2020-01-01", "2020-01-10", "P", 100, 50.0, 50.2),  # way off -> would skew result
    ]
    target_term = [
        _row("2020-01-01", "2020-01-31", "C", 100, 3.0, 3.2),
        _row("2020-01-01", "2020-01-31", "P", 100, 3.0, 3.2),
    ]
    chain = pd.DataFrame(near_term + target_term)

    engine = SyntheticSpotEngine(target_dte=30, n_strikes=1)
    spots = engine.compute_spot_series(chain)

    assert spots.iloc[0] == 100.0


def test_missing_quotes_for_a_day_are_dropped_not_raised():
    rows = [
        # 2020-01-01 has no matched put/call strikes at all
        _row("2020-01-01", "2020-01-31", "C", 100, 3.0, 3.2),
        # 2020-01-02 has a valid matched pair
        _row("2020-01-02", "2020-01-31", "C", 100, 3.0, 3.2),
        _row("2020-01-02", "2020-01-31", "P", 100, 3.0, 3.2),
    ]
    chain = pd.DataFrame(rows)

    engine = SyntheticSpotEngine(target_dte=30, n_strikes=1)
    spots = engine.compute_spot_series(chain)

    assert list(spots.index) == [pd.Timestamp("2020-01-02")]


def test_empty_chain_returns_empty_series():
    engine = SyntheticSpotEngine()
    spots = engine.compute_spot_series(pd.DataFrame(columns=["date", "exdate", "cp_flag", "strike_price", "best_bid", "best_offer"]))
    assert spots.empty

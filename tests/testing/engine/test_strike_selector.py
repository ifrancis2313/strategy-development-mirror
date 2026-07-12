import pandas as pd

from testing.engine.strike_selector import StrikeSelector, select_target_expiry


def _chain(rows):
    return pd.DataFrame(rows)


def _expiry_row(exdate, strike=100, optionid=None):
    return {
        "exdate": pd.Timestamp(exdate),
        "strike_price": strike,
        "optionid": optionid if optionid is not None else exdate,
    }


# ---------------------------------------------------------------------
# select_target_expiry -- had zero direct test coverage before this pass
# (only exercised indirectly through full strategy backtests), despite its
# own docstring calling it shared, generic chain mechanics for every
# strategy's trade selector.
# ---------------------------------------------------------------------


def test_select_target_expiry_picks_expiry_nearest_the_window_midpoint():
    entry = pd.Timestamp("2020-01-01")
    chain = _chain([
        _expiry_row(entry + pd.Timedelta(days=20)),
        _expiry_row(entry + pd.Timedelta(days=25)),  # window [20,30], midpoint=25
        _expiry_row(entry + pd.Timedelta(days=30)),
    ])
    result = select_target_expiry(chain, entry, min_dte=20, max_dte=30)
    assert result is not None
    chosen_exdate, _slice = result
    assert chosen_exdate == entry + pd.Timedelta(days=25)


def test_select_target_expiry_midpoint_is_the_average_not_double_the_sum():
    """`(min_dte + max_dte) / 2` -- mutation testing (2026-07-12) found this
    survives being flipped to `* 2`. min_dte=20/max_dte=30 gives midpoint=25
    (correct) vs 100 (mutant) -- an expiry at dte=21 is nearest the real
    midpoint but nowhere near the mutant's, so the two formulas pick
    different expiries."""
    entry = pd.Timestamp("2020-01-01")
    chain = _chain([
        _expiry_row(entry + pd.Timedelta(days=21)),  # nearest real midpoint (25)
        _expiry_row(entry + pd.Timedelta(days=30)),  # nearest mutant "midpoint" (100, clamped by window)
    ])
    result = select_target_expiry(chain, entry, min_dte=20, max_dte=30)
    assert result is not None
    chosen_exdate, _slice = result
    assert chosen_exdate == entry + pd.Timedelta(days=21)


def test_select_target_expiry_window_includes_max_dte_boundary():
    """`dte <= max_dte` -- mutation testing (2026-07-12) found this survives
    being tightened to `< max_dte`. An expiry landing exactly on the DTE
    ceiling, with nothing else in the window, must still be selected."""
    entry = pd.Timestamp("2020-01-01")
    chain = _chain([_expiry_row(entry + pd.Timedelta(days=30))])  # dte == max_dte exactly
    result = select_target_expiry(chain, entry, min_dte=20, max_dte=30)
    assert result is not None
    chosen_exdate, _slice = result
    assert chosen_exdate == entry + pd.Timedelta(days=30)


def test_select_target_expiry_returns_none_when_nothing_in_window():
    entry = pd.Timestamp("2020-01-01")
    chain = _chain([_expiry_row(entry + pd.Timedelta(days=5))])
    assert select_target_expiry(chain, entry, min_dte=20, max_dte=30) is None


def _row(cp_flag, strike, delta, bid=1.0, offer=1.2, optionid=None):
    return {
        "cp_flag": cp_flag,
        "strike_price": strike,
        "delta": delta,
        "best_bid": bid,
        "best_offer": offer,
        "optionid": optionid if optionid is not None else strike,
    }


def test_selects_nearest_delta_on_correct_side():
    chain = _chain([
        _row("P", 90, -0.10),
        _row("P", 95, -0.20),
        _row("P", 100, -0.50),
        _row("C", 105, 0.20),
    ])
    selector = StrikeSelector()

    match = selector.select(chain, target_delta=-0.20, cp_flag="P")

    assert match is not None
    assert match.strike_price == 95
    assert match.delta == -0.20


def test_ignores_contracts_on_the_other_side():
    chain = _chain([
        _row("C", 95, -0.20),  # exact delta match but wrong side, must be ignored
        _row("P", 100, -0.22),  # within tolerance on the correct side
    ])
    selector = StrikeSelector()

    match = selector.select(chain, target_delta=-0.20, cp_flag="P")

    assert match is not None
    assert match.strike_price == 100


def test_accepts_a_match_exactly_at_max_delta_distance():
    """`distances.loc[best_idx] > self.max_delta_distance` -- mutation
    testing (2026-07-12) found the boundary survives being loosened to
    `>=`, which would reject a match landing exactly on the tolerance
    edge instead of accepting it. Uses quarter-valued deltas (-0.25/-0.5,
    both exact in float64) so |delta - target| lands bit-exact on
    max_delta_distance -- 0.20/0.05-style decimals don't round-trip
    through subtraction exactly and can silently miss the boundary."""
    chain = _chain([_row("P", 95, -0.5)])  # |delta - target| == 0.25 exactly
    selector = StrikeSelector(max_delta_distance=0.25)

    match = selector.select(chain, target_delta=-0.25, cp_flag="P")

    assert match is not None
    assert match.strike_price == 95


def test_returns_none_when_nothing_within_tolerance():
    chain = _chain([
        _row("P", 90, -0.01),
        _row("P", 95, -0.60),
    ])
    selector = StrikeSelector(max_delta_distance=0.05)

    match = selector.select(chain, target_delta=-0.20, cp_flag="P")

    assert match is None


def test_returns_none_when_no_contracts_on_that_side():
    chain = _chain([_row("C", 105, 0.20)])
    selector = StrikeSelector()

    assert selector.select(chain, target_delta=-0.20, cp_flag="P") is None


def test_returns_none_on_empty_chain():
    selector = StrikeSelector()
    empty = pd.DataFrame(columns=["cp_flag", "strike_price", "delta", "best_bid", "best_offer", "optionid"])
    assert selector.select(empty, target_delta=-0.20, cp_flag="P") is None


def test_rows_with_nan_delta_are_excluded():
    chain = _chain([
        _row("P", 95, float("nan")),
        _row("P", 100, -0.21),
    ])
    selector = StrikeSelector()

    match = selector.select(chain, target_delta=-0.20, cp_flag="P")

    assert match is not None
    assert match.strike_price == 100

import numpy as np
import pandas as pd
import pytest

from testing.validation.walk_forward_validation import (
    WalkForwardFoldResult,
    WalkForwardReport,
    WalkForwardWindow,
    _annualized_sharpe_from_trades,
    _slice_chain,
    generate_walk_forward_windows,
    run_walk_forward_validation,
)

# ---------------------------------------------------------------------
# _slice_chain -- the actual train/test date partition every fold uses.
# ADR 0005/0006's whole leak-free claim rests on this exact boundary:
# [start, end), so the fold-boundary date (test_start == the prior fold's
# train_end) belongs to test, never train, and never gets silently
# dropped from both. No prior test drove this function directly.
# ---------------------------------------------------------------------


def _dated_chain(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"date": [pd.Timestamp(d) for d in dates], "value": range(len(dates))})


def test_slice_chain_start_is_inclusive() -> None:
    """`date >= start` -- mutation testing (2026-07-12) found this survives
    being tightened to `> start`, which would silently drop the first day
    of every window (not leak it into the wrong set -- lose it entirely)."""
    chain = _dated_chain(["2020-01-01", "2020-01-02", "2020-01-03"])
    sliced = _slice_chain(chain, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
    assert pd.Timestamp("2020-01-01") in sliced["date"].values


def test_slice_chain_end_is_exclusive() -> None:
    """`date < end` -- mutation testing (2026-07-12) found this survives
    being loosened to `<= end`. This is the actual leakage risk: since
    `generate_walk_forward_windows` sets test_start == train_end for
    adjacent folds, an inclusive end would put that boundary date in BOTH
    the training window (used to select a candidate) and the test window
    (used to score it) -- exactly the train/test overlap walk-forward
    exists to prevent."""
    chain = _dated_chain(["2020-01-01", "2020-01-02", "2020-01-03"])
    sliced = _slice_chain(chain, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03"))
    assert pd.Timestamp("2020-01-03") not in sliced["date"].values


def test_slice_chain_fold_boundary_date_belongs_to_test_not_train_and_is_not_dropped() -> None:
    """End-to-end version of the two tests above, using real adjacent-fold
    boundaries from generate_walk_forward_windows: the exact date at
    train_end/test_start must appear in the test slice, must NOT appear in
    the train slice, and must not vanish from both (the failure mode if
    start were also made exclusive)."""
    windows = generate_walk_forward_windows("2000-01-01", "2005-01-01", train_years=4, test_years=1)
    assert len(windows) == 1
    boundary = windows[0].train_end
    assert boundary == windows[0].test_start

    chain = _dated_chain(["1999-12-31", boundary.strftime("%Y-%m-%d"), "2005-06-01"])
    train_slice = _slice_chain(chain, windows[0].train_start, windows[0].train_end)
    test_slice = _slice_chain(chain, windows[0].test_start, windows[0].test_end)

    assert boundary not in train_slice["date"].values
    assert boundary in test_slice["date"].values


# ---------------------------------------------------------------------
# generate_walk_forward_windows
# ---------------------------------------------------------------------


def test_generates_expected_number_and_boundaries_of_folds() -> None:
    # 10 years of data, 4yr train / 1yr test / 1yr step -> folds start at
    # years 0..5 (train_end+test_end must not exceed data_end at year 10):
    # fold k: train=[k, k+4), test=[k+4, k+5) -- last valid k is 5 (test ends at 10).
    windows = generate_walk_forward_windows("2000-01-01", "2010-01-01", train_years=4, test_years=1, step_years=1)
    assert len(windows) == 6
    assert windows[0].train_start == pd.Timestamp("2000-01-01")
    assert windows[0].train_end == pd.Timestamp("2004-01-01")
    assert windows[0].test_start == pd.Timestamp("2004-01-01")
    assert windows[0].test_end == pd.Timestamp("2005-01-01")
    assert windows[-1].fold == 5
    assert windows[-1].train_start == pd.Timestamp("2005-01-01")
    assert windows[-1].test_end == pd.Timestamp("2010-01-01")


def test_test_windows_are_contiguous_non_overlapping_when_step_equals_test_years() -> None:
    windows = generate_walk_forward_windows("2000-01-01", "2010-01-01", train_years=4, test_years=1, step_years=1)
    for a, b in zip(windows, windows[1:], strict=False):
        assert a.test_end == b.test_start


def test_too_little_data_returns_empty_list_not_error() -> None:
    windows = generate_walk_forward_windows("2000-01-01", "2001-01-01", train_years=4, test_years=1)
    assert windows == []


def test_exact_one_window_of_data() -> None:
    windows = generate_walk_forward_windows("2000-01-01", "2005-01-01", train_years=4, test_years=1)
    assert len(windows) == 1


def test_overlapping_test_windows_when_step_smaller_than_test_years() -> None:
    windows = generate_walk_forward_windows("2000-01-01", "2007-01-01", train_years=4, test_years=2, step_years=1)
    assert len(windows) >= 2
    assert windows[1].test_start < windows[0].test_end  # genuinely overlapping


# ---------------------------------------------------------------------
# _annualized_sharpe_from_trades
# ---------------------------------------------------------------------


def test_annualized_sharpe_matches_hand_computation() -> None:
    pnl = [10.0, -5.0, 20.0, 3.0]
    equity = pd.Series([10, 5, 25, 28], index=pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"]))
    sharpe, n = _annualized_sharpe_from_trades(pnl, equity)
    span_days = (pd.Timestamp("2020-10-01") - pd.Timestamp("2020-01-01")).days
    periods_per_year = 4 / (span_days / 365.25)
    arr = np.array(pnl)
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(periods_per_year)
    assert sharpe == pytest.approx(expected)
    assert n == 4


def test_annualized_sharpe_empty_pnl_is_nan() -> None:
    sharpe, n = _annualized_sharpe_from_trades([], pd.Series(dtype=float))
    assert np.isnan(sharpe)
    assert n == 0


def test_annualized_sharpe_zero_variance_is_nan() -> None:
    equity = pd.Series([10, 20], index=pd.to_datetime(["2020-01-01", "2020-06-01"]))
    sharpe, n = _annualized_sharpe_from_trades([5.0, 5.0], equity)
    assert np.isnan(sharpe)
    assert n == 2


def test_annualized_sharpe_none_equity_is_nan() -> None:
    sharpe, n = _annualized_sharpe_from_trades([1.0, 2.0], None)
    assert np.isnan(sharpe)
    assert n == 0


# ---------------------------------------------------------------------
# run_walk_forward_validation -- synthetic candidates with known behavior
# ---------------------------------------------------------------------


def _synthetic_chain(start: str, end: str) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"date": dates})


def _fixed_runner(good: bool):
    """A candidate whose Sharpe is deterministically good or bad, regardless
    of which chain slice it's given -- lets tests assert selection behavior
    precisely without depending on real strategy logic.
    """

    def run(chain: pd.DataFrame):
        if chain.empty:
            return [], pd.Series(dtype=float), {}
        n = min(len(chain) // 30, 20) or 1  # scale trade count with window size, capped
        dates = pd.date_range(chain["date"].min(), chain["date"].max(), periods=n)
        # good: positive, low-variance -> high Sharpe. bad: negative -> negative Sharpe.
        pnl = [10.0 + i * 0.1 for i in range(n)] if good else [-10.0 - i * 0.1 for i in range(n)]
        equity = pd.Series(np.cumsum(pnl), index=dates)
        return pnl, equity, {}

    return run


def _empty_runner():
    def run(chain: pd.DataFrame):
        return [], pd.Series(dtype=float), {}

    return run


def test_selects_better_in_sample_candidate_each_fold() -> None:
    chain = _synthetic_chain("2000-01-01", "2010-01-01")
    candidates = {"good": _fixed_runner(good=True), "bad": _fixed_runner(good=False)}
    report = run_walk_forward_validation(chain, candidates, train_years=4, test_years=1, step_years=1)
    assert report.n_folds == 6
    assert all(f.selected_candidate == "good" for f in report.folds)


def test_out_of_sample_sharpe_positive_for_a_genuinely_good_candidate() -> None:
    chain = _synthetic_chain("2000-01-01", "2010-01-01")
    candidates = {"good": _fixed_runner(good=True), "bad": _fixed_runner(good=False)}
    report = run_walk_forward_validation(chain, candidates, train_years=4, test_years=1, step_years=1)
    assert report.out_of_sample_positive_fraction == 1.0
    assert report.mean_out_of_sample_sharpe > 0


def test_walk_forward_efficiency_defined_when_in_sample_positive() -> None:
    chain = _synthetic_chain("2000-01-01", "2010-01-01")
    candidates = {"good": _fixed_runner(good=True), "bad": _fixed_runner(good=False)}
    report = run_walk_forward_validation(chain, candidates, train_years=4, test_years=1, step_years=1)
    assert report.mean_walk_forward_efficiency is not None
    for f in report.folds:
        assert f.walk_forward_efficiency is not None


def test_walk_forward_efficiency_none_when_in_sample_never_positive() -> None:
    chain = _synthetic_chain("2000-01-01", "2010-01-01")
    candidates = {"only_bad": _fixed_runner(good=False)}
    report = run_walk_forward_validation(chain, candidates, train_years=4, test_years=1, step_years=1)
    assert report.mean_walk_forward_efficiency is None
    for f in report.folds:
        assert f.walk_forward_efficiency is None
    # OOS positive fraction still meaningful even when WFE isn't
    assert report.out_of_sample_positive_fraction == 0.0


def test_degenerate_fold_selects_first_candidate_and_reports_nan() -> None:
    chain = _synthetic_chain("2000-01-01", "2010-01-01")
    candidates = {"first_empty": _empty_runner(), "second_empty": _empty_runner()}
    report = run_walk_forward_validation(chain, candidates, train_years=4, test_years=1, step_years=1)
    for f in report.folds:
        assert f.selected_candidate == "first_empty"
        assert np.isnan(f.in_sample_sharpe)
        assert np.isnan(f.out_of_sample_sharpe)
    assert np.isnan(report.mean_out_of_sample_sharpe)


def test_empty_candidates_raises() -> None:
    chain = _synthetic_chain("2000-01-01", "2010-01-01")
    with pytest.raises(ValueError):
        run_walk_forward_validation(chain, {})


def test_too_short_chain_raises() -> None:
    chain = _synthetic_chain("2000-01-01", "2001-01-01")
    with pytest.raises(ValueError):
        run_walk_forward_validation(chain, {"good": _fixed_runner(good=True)}, train_years=4, test_years=1)


def test_test_window_never_influences_selection() -> None:
    """A candidate that's good in-sample but degenerate out-of-sample is
    still selected (selection uses train data only) -- the OOS score just
    comes back nan/negative for it, it isn't second-guessed retroactively.
    """

    def good_train_bad_test(chain: pd.DataFrame):
        if chain.empty:
            return [], pd.Series(dtype=float), {}
        # Behaves well only in years < 2004 (i.e. only ever seen as a train window here)
        if chain["date"].min() < pd.Timestamp("2004-01-01") and chain["date"].max() <= pd.Timestamp("2004-01-01"):
            pnl = [10.0, 10.5, 11.0]
        else:
            pnl = [-10.0, -10.5, -11.0]
        dates = pd.date_range(chain["date"].min(), chain["date"].max(), periods=3)
        equity = pd.Series(np.cumsum(pnl), index=dates)
        return pnl, equity, {}

    chain = _synthetic_chain("2000-01-01", "2006-01-01")
    candidates = {"train_only_good": good_train_bad_test}
    report = run_walk_forward_validation(chain, candidates, train_years=4, test_years=1, step_years=1)
    fold0 = report.folds[0]
    assert fold0.selected_candidate == "train_only_good"
    assert fold0.in_sample_sharpe > 0
    assert fold0.out_of_sample_sharpe < 0  # test window scored honestly, not cherry-picked


# ---------------------------------------------------------------------
# WalkForwardReport aggregation properties, direct construction
# ---------------------------------------------------------------------


def _fold(fold: int, is_sharpe: float, oos_sharpe: float) -> WalkForwardFoldResult:
    w = WalkForwardWindow(
        fold=fold,
        train_start=pd.Timestamp("2000-01-01"),
        train_end=pd.Timestamp("2004-01-01"),
        test_start=pd.Timestamp("2004-01-01"),
        test_end=pd.Timestamp("2005-01-01"),
    )
    return WalkForwardFoldResult(
        window=w, selected_candidate="x", in_sample_sharpe=is_sharpe, in_sample_n_trades=10,
        out_of_sample_sharpe=oos_sharpe, out_of_sample_n_trades=3,
    )


def test_mean_out_of_sample_sharpe_ignores_nan_folds() -> None:
    report = WalkForwardReport(folds=(_fold(0, 1.0, 0.5), _fold(1, 1.0, float("nan")), _fold(2, 1.0, 1.5)))
    assert report.mean_out_of_sample_sharpe == pytest.approx(1.0)  # mean of 0.5 and 1.5 only


def test_empty_report_has_nan_aggregates() -> None:
    report = WalkForwardReport(folds=())
    assert np.isnan(report.mean_out_of_sample_sharpe)
    assert np.isnan(report.out_of_sample_positive_fraction)
    assert report.mean_walk_forward_efficiency is None


def test_summary_mentions_key_fields() -> None:
    report = WalkForwardReport(folds=(_fold(0, 1.0, 0.5),))
    text = report.summary()
    for keyword in ("Folds", "out-of-sample Sharpe", "positive fraction", "walk-forward efficiency", "Fold 0"):
        assert keyword in text

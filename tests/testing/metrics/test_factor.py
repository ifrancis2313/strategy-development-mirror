import numpy as np
import pandas as pd
import pytest

from testing.metrics.factor import (
    FactorSignificanceReport,
    bonferroni_critical_value,
    build_factor_data,
    cross_sectional_correlation,
    cross_sectional_zscore,
    flatten_tstats,
    information_coefficient,
    multiple_testing_correction,
    newey_west_mean_and_tstat,
    quintile_spread,
    rank_weighted_returns,
    run_significance_suite,
    significance_report,
    slice_by_regime,
)

N_ASSETS = 40
N_DAYS = 150


def _synthetic_panel(signal_strength: float, seed: int = 42) -> tuple[pd.Series, pd.DataFrame]:
    """Builds a (date, asset)-indexed factor and a matching price panel where
    the *next* day's return is partly driven by *today's* cross-sectionally
    z-scored factor value: return[t] = signal_strength * factor_z[t-1] +
    noise[t]. This makes 1D-forward IC a known, reproducible, strongly
    positive quantity when signal_strength > 0, and pure noise (IC ~ 0) when
    signal_strength == 0 - the two cases every test below needs.

    Business-day dates and >=30 names/date are required for alphalens'
    frequency inference and quantile bucketing to behave; this mirrors the
    shape (though not the scale) of a real CRSP-derived panel.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=N_DAYS)
    assets = [f"A{i}" for i in range(N_ASSETS)]

    factor_vals = rng.normal(size=(N_DAYS, N_ASSETS))
    factor_z = (factor_vals - factor_vals.mean(axis=1, keepdims=True)) / factor_vals.std(axis=1, keepdims=True)
    noise = rng.normal(size=(N_DAYS, N_ASSETS))

    shifted_factor_z = np.vstack([np.zeros((1, N_ASSETS)), factor_z[:-1]])
    daily_returns = signal_strength * shifted_factor_z + 0.01 * noise

    returns = pd.DataFrame(daily_returns, index=dates, columns=assets)
    prices = (1.0 + returns).cumprod()

    factor = pd.DataFrame(factor_vals, index=dates, columns=assets).stack()
    factor.index.names = ["date", "asset"]
    return factor, prices


@pytest.fixture(scope="module")
def strong_signal_panel() -> tuple[pd.Series, pd.DataFrame]:
    return _synthetic_panel(signal_strength=0.03)


@pytest.fixture(scope="module")
def null_signal_panel() -> tuple[pd.Series, pd.DataFrame]:
    return _synthetic_panel(signal_strength=0.0)


# --- cross-sectional helpers -------------------------------------------------


def test_cross_sectional_zscore_has_zero_mean_and_unit_std_per_date():
    dates = ["2020-01-01"] * 4 + ["2020-01-02"] * 4
    assets = ["A", "B", "C", "D"] * 2
    values = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0]
    factor = pd.Series(values, index=pd.MultiIndex.from_arrays([pd.to_datetime(dates), assets], names=["date", "asset"]))

    z = cross_sectional_zscore(factor)

    for _, day in z.groupby(level="date"):
        assert day.mean() == pytest.approx(0.0, abs=1e-9)
        assert day.std() == pytest.approx(1.0, abs=1e-9)


def test_cross_sectional_correlation_is_one_for_identical_factors():
    dates = pd.to_datetime(["2020-01-01"] * 35)
    assets = [f"A{i}" for i in range(35)]
    values = np.arange(35, dtype=float)
    idx = pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])
    factor = pd.Series(values, index=idx)

    corr = cross_sectional_correlation(factor, factor, min_names=30)

    assert corr == pytest.approx(1.0, abs=1e-9)


def test_cross_sectional_correlation_skips_dates_below_min_names():
    # Only 5 names on this date - below min_names=30, should be excluded
    # entirely (not just downweighted), leaving no dates to average -> NaN.
    dates = pd.to_datetime(["2020-01-01"] * 5)
    assets = [f"A{i}" for i in range(5)]
    idx = pd.MultiIndex.from_arrays([dates, assets], names=["date", "asset"])
    factor_a = pd.Series(np.arange(5, dtype=float), index=idx)
    factor_b = pd.Series(np.arange(5, dtype=float)[::-1], index=idx)

    corr = cross_sectional_correlation(factor_a, factor_b, min_names=30)

    assert np.isnan(corr)


def test_slice_by_regime_filters_on_date_bounds():
    dates = pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-01"])
    idx = pd.MultiIndex.from_arrays([dates, ["A", "A", "A"]], names=["date", "asset"])
    data = pd.DataFrame({"x": [1, 2, 3]}, index=idx)

    full = slice_by_regime(data, None, None)
    early = slice_by_regime(data, None, "2020-06-01")
    late = slice_by_regime(data, "2020-06-02", None)

    assert len(full) == 3
    assert list(early["x"]) == [1, 2]
    assert list(late["x"]) == [3]


# --- Newey-West t-stat --------------------------------------------------


def test_newey_west_mean_matches_sample_mean():
    # OLS of a series on a constant-1 regressor recovers the sample mean
    # exactly, regardless of the HAC covariance correction applied to its
    # standard error - only the t-stat (not the point estimate) should move
    # with maxlags.
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    mean, _ = newey_west_mean_and_tstat(series, maxlags=0)

    assert mean == pytest.approx(3.0)


def test_newey_west_tstat_is_positive_for_a_clearly_positive_series():
    series = pd.Series(np.full(60, 0.01) + np.random.default_rng(1).normal(scale=0.001, size=60))

    mean, t_stat = newey_west_mean_and_tstat(series, maxlags=4)

    assert mean > 0
    assert t_stat > 5  # tiny noise around a solidly positive mean -> large t


def test_newey_west_drops_nans_before_fitting():
    series = pd.Series([1.0, np.nan, 1.0, np.nan, 1.0])

    mean, t_stat = newey_west_mean_and_tstat(series, maxlags=0)

    assert mean == pytest.approx(1.0)
    # A constant (zero-variance) series has zero HAC standard error, so the
    # t-stat is +inf, not NaN - the point is just that dropna() ran first
    # (fitting OLS on a raw series containing NaN would raise instead).
    assert t_stat == float("inf") or np.isnan(t_stat)


# --- factor_data / IC / spread / rank-weighted returns -----------------------


def test_build_factor_data_has_expected_columns(strong_signal_panel):
    factor, prices = strong_signal_panel

    factor_data = build_factor_data(factor, prices, periods=(1, 5), quantiles=5)

    assert {"1D", "5D", "factor", "factor_quantile"}.issubset(factor_data.columns)
    assert factor_data.index.names == ["date", "asset"]


def test_information_coefficient_strongly_positive_for_strong_signal(strong_signal_panel):
    factor, prices = strong_signal_panel
    factor_data = build_factor_data(factor, prices, periods=(1,), quantiles=5)

    ic = information_coefficient(factor_data)

    assert ic["1D"].mean() > 0.5  # engineered signal is strong; matches the interactive check done during dev


def test_information_coefficient_near_zero_for_null_signal(null_signal_panel):
    factor, prices = null_signal_panel
    factor_data = build_factor_data(factor, prices, periods=(1,), quantiles=5)

    ic = information_coefficient(factor_data)

    assert abs(ic["1D"].mean()) < 0.15  # pure noise: IC should hover near zero, not be reliably large either way


def test_quintile_spread_positive_for_strong_signal_generalizes_beyond_five_quantiles(strong_signal_panel):
    factor, prices = strong_signal_panel
    factor_data = build_factor_data(factor, prices, periods=(1,), quantiles=3)

    spread = quintile_spread(factor_data)

    assert spread["1D"].mean() > 0  # top quantile should out-earn bottom quantile when the signal is real


def test_rank_weighted_returns_positive_for_strong_signal(strong_signal_panel):
    factor, prices = strong_signal_panel
    factor_data = build_factor_data(factor, prices, periods=(1,), quantiles=5)

    rw = rank_weighted_returns(factor_data)

    assert rw["1D"].mean() > 0


# --- significance_report / run_significance_suite ----------------------------


def test_significance_report_returns_none_below_min_trading_days(strong_signal_panel):
    factor, prices = strong_signal_panel
    factor_data = build_factor_data(factor, prices, periods=(1,), quantiles=5)
    too_short = slice_by_regime(factor_data, "2020-01-01", "2020-01-10")

    report = significance_report(too_short, forward_periods=(1,), min_trading_days=30)

    assert report is None


def test_significance_report_strong_signal_clears_uncorrected_bar(strong_signal_panel):
    factor, prices = strong_signal_panel
    factor_data = build_factor_data(factor, prices, periods=(1, 5), quantiles=5)

    report = significance_report(factor_data, forward_periods=(1, 5))

    assert isinstance(report, FactorSignificanceReport)
    assert report.n_trading_days > 100
    row = report.table.loc["1D"]
    assert row["mean_IC"] > 0
    assert abs(row["IC_t_stat_NW"]) > 1.96
    assert abs(row["rank_weighted_t_stat_NW"]) > 1.96


def test_run_significance_suite_defaults_to_full_period(strong_signal_panel):
    factor, prices = strong_signal_panel

    reports = run_significance_suite(factor, prices, forward_periods=(1,))

    assert set(reports.keys()) == {"full_period"}
    assert reports["full_period"] is not None


def test_run_significance_suite_respects_custom_regimes(strong_signal_panel):
    factor, prices = strong_signal_panel
    regimes = {
        "first_half": (None, "2020-05-01"),
        "second_half": ("2020-05-02", None),
    }

    reports = run_significance_suite(factor, prices, forward_periods=(1,), regimes=regimes)

    assert set(reports.keys()) == {"first_half", "second_half"}
    for report in reports.values():
        assert report is not None
        assert report.table.loc["1D", "mean_IC"] > 0


# --- flatten_tstats / multiple_testing_correction ----------------------------


def test_flatten_tstats_produces_one_row_per_metric_per_period(strong_signal_panel):
    factor, prices = strong_signal_panel
    reports = run_significance_suite(factor, prices, forward_periods=(1, 5))

    rows = flatten_tstats("model5_composite", reports)

    # 1 regime (full_period) x 2 periods x 2 default metrics = 4 rows
    assert len(rows) == 4
    assert all(r["test"].startswith("model5_composite | full_period |") for r in rows)


def test_flatten_tstats_skips_none_reports():
    reports: dict[str, FactorSignificanceReport | None] = {"too_short": None}

    rows = flatten_tstats("x", reports)

    assert rows == []


def test_bonferroni_critical_value_increases_with_n_tests():
    single = bonferroni_critical_value(0.05, 1)
    many = bonferroni_critical_value(0.05, 100)

    assert single == pytest.approx(1.9599639845, abs=1e-6)  # uncorrected two-sided 95% critical |z|
    assert many > single


def test_bonferroni_critical_value_rejects_non_positive_n():
    with pytest.raises(ValueError):
        bonferroni_critical_value(0.05, 0)


def test_multiple_testing_correction_flags_survivors_correctly():
    tstats = [
        {"test": "a", "t_stat": 5.0},
        {"test": "b", "t_stat": -0.1},
        {"test": "c", "t_stat": 2.0},
    ]

    table = multiple_testing_correction(tstats, alpha=0.05)

    assert list(table["test"]) == ["a", "c", "b"]  # sorted by |t| descending
    critical = bonferroni_critical_value(0.05, 3)
    assert table.set_index("test").loc["a", "survives_bonferroni"]
    assert not table.set_index("test").loc["b", "survives_bonferroni"]
    assert not table.set_index("test").loc["c", "survives_bonferroni"]
    # pytest.approx doesn't broadcast elementwise against a pandas Series
    # (it always compares as a whole object, not per-element) - compare the
    # underlying numpy array instead.
    assert np.allclose(table["bonferroni_critical"].to_numpy(), critical)


def test_multiple_testing_correction_empty_input_returns_empty_frame():
    table = multiple_testing_correction([])

    assert table.empty

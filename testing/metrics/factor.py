"""Equity factor model metrics: information coefficient (IC) and
cross-sectional (quintile-spread) significance testing.

Matches `alpha-pipeline`'s existing significance-testing conventions
(`research/factor_common.py`, `research/factor_test_model5_robustness.py`,
`research/factor_test_multiple_testing_correction.py`) so a factor already
validated there reproduces the same statistical conclusion here: Spearman-
rank IC and a demeaned quintile-sort spread, both Newey-West (HAC) corrected
for the autocorrelation that overlapping N-day forward returns induce,
sliced across caller-supplied regimes, with a Bonferroni multiple-testing
correction across however many tests were run in a session.

Engine-independent by design (see this repo's `testing/CONTEXT.md`): this
module operates on a plain (date, asset)-indexed factor `pd.Series` and a
date x asset price panel. It has no dependency on `testing.engine` (which is
options-backtesting machinery) and no opinion on where the factor/universe
came from - point-in-time correctness of the input (per
`research-methodology/METHODOLOGY.md`'s pre-flight checklist) is the
caller's responsibility, not this module's.

Default exposure metric is IC-weighted/rank-weighted
(`rank_weighted_returns`, alphalens' own demeaned factor-weighted portfolio
return), not a discrete quintile long-short spread - alpha-pipeline's own
robustness grid found 0/18 parameterizations gave a significant Q5-Q1
t-stat despite strong IC t-stats (see this repo's `AGENTS.md` /
`alpha-pipeline/AGENTS.md`). The quintile-spread metric is still computed
(an explicit acceptance criterion for this module, and a useful
cross-check), but `rank_weighted_returns`/IC significance is the metric a
caller should treat as primary when deciding whether a factor has a real,
sizeable exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import alphalens as al
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

DEFAULT_QUANTILES = 5
MIN_TRADING_DAYS = 30  # below this, a t-stat isn't trustworthy - skip rather than report noise
DEFAULT_ALPHA = 0.05


def newey_west_mean_and_tstat(series: pd.Series, maxlags: int) -> tuple[float, float]:
    """Mean of `series` and its Newey-West (HAC) t-stat against zero,
    correcting for the autocorrelation that overlapping N-day forward
    returns induce. `maxlags` should be `forward_period - 1` for a series
    derived from N-day-forward returns (matches alpha-pipeline's
    convention: consecutive N-day forward-return observations overlap by
    N-1 days).
    """
    series = series.dropna()
    ones = np.ones(len(series))
    model = sm.OLS(series.to_numpy(), ones).fit(cov_type="HAC", cov_kwds={"maxlags": max(maxlags, 0)})
    return float(model.params[0]), float(model.tvalues[0])


def cross_sectional_zscore(factor: pd.Series) -> pd.Series:
    """Per-date cross-sectional z-score: (value - that day's mean) / that
    day's std. Puts differently-scaled factors (e.g. a momentum return vs.
    a volatility level) on a comparable footing before combining them into
    a composite."""
    grouped = factor.groupby(level="date")
    return (factor - grouped.transform("mean")) / grouped.transform("std")


def cross_sectional_correlation(factor_a: pd.Series, factor_b: pd.Series, min_names: int = 30) -> float:
    """Average per-date cross-sectional Pearson correlation between two
    (date, asset)-indexed factors, over dates with at least `min_names`
    names in common.

    This is the redundancy check to run before combining any two factors
    into a composite (alpha-pipeline's own `qepm_factor_models.md`
    convention: anything above ~0.6-0.7 correlated is redundant, not
    additive) - deliberately a per-date cross-sectional correlation (do the
    two factors rank the same names similarly on a given day), not a pooled
    time-series correlation, which would conflate genuine redundancy with
    the two factors simply moving together through time.
    """
    aligned = pd.DataFrame({"a": factor_a, "b": factor_b}).dropna()
    daily_corr = aligned.groupby(level="date").apply(
        lambda day: day["a"].corr(day["b"]) if len(day) >= min_names else np.nan
    )
    return float(daily_corr.mean())


def slice_by_regime(factor_data: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Restricts an already-built factor_data panel (see `build_factor_data`)
    to `[start, end]` (either bound optional/open-ended), for regime-
    bucketed robustness checks - a result that only holds on the full
    period and falls apart on either half is a robustness red flag, not a
    validated finding."""
    dates = factor_data.index.get_level_values("date")
    mask = pd.Series(True, index=factor_data.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    return factor_data[mask]


def build_factor_data(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: tuple[int, ...],
    quantiles: int = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """Aligns `factor` (a (date, asset)-indexed Series, sign convention:
    HIGHER value = expected BETTER forward return, i.e. the top quantile is
    the long leg) against forward returns computed from `prices` (a date x
    asset panel of prices, or any cumulative-return index - only relative
    levels matter, not $ prices).

    Thin wrapper around alphalens' `get_clean_factor_and_forward_returns` -
    alpha-pipeline's own leakage-safe forward-return/quantile-bucketing
    machinery, confirmed point-in-time correct by hand-trace (see
    `research-methodology/METHODOLOGY.md`'s Detection Techniques section on
    `quantopian/alphalens`). Deliberately does not pass `filter_zscore`,
    matching alpha-pipeline's own calls (library default) rather than
    silently changing behavior underneath a reproduction check.
    """
    return al.utils.get_clean_factor_and_forward_returns(factor, prices, periods=periods, quantiles=quantiles)


def rank_weighted_returns(factor_data: pd.DataFrame) -> pd.DataFrame:
    """The default exposure metric: per-date, dollar-neutral, factor-value-
    weighted portfolio return per forward period (alphalens'
    `factor_returns(demeaned=True)`) - a continuous IC-weighted/rank-
    weighted exposure, not a discrete quintile long-short spread. See
    module docstring for why this is the default rather than
    `quintile_spread`."""
    return al.performance.factor_returns(factor_data, demeaned=True)


def information_coefficient(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Per-date Spearman rank IC between factor value and each forward
    return period (alphalens' `factor_information_coefficient`)."""
    return al.performance.factor_information_coefficient(factor_data)


def quintile_spread(factor_data: pd.DataFrame) -> pd.DataFrame:
    """Per-date top-quantile-minus-bottom-quantile mean forward return
    (`mean_return_by_quantile(by_date=True)`), one column per forward
    period. "Top quantile" is whatever the highest quantile label actually
    present is, so this generalizes to however many quantiles
    `build_factor_data` was called with - not hardcoded to 5."""
    mean_ret, _ = al.performance.mean_return_by_quantile(factor_data, by_date=True)
    top_quantile = mean_ret.index.get_level_values("factor_quantile").max()
    spreads = {
        col: mean_ret[col].xs(top_quantile, level="factor_quantile") - mean_ret[col].xs(1, level="factor_quantile")
        for col in mean_ret.columns
    }
    return pd.DataFrame(spreads)


@dataclass(frozen=True)
class FactorSignificanceReport:
    """Per-forward-period significance metrics for one factor/regime slice.

    `table` is indexed by period label (e.g. "21D") with columns
    `mean_IC`/`IC_t_stat_NW`, `Q_top_Q1_spread`/`spread_t_stat_NW`, and
    `rank_weighted_mean_return`/`rank_weighted_t_stat_NW` - mirrors
    alpha-pipeline's `factor_common.significance_report` output shape, plus
    the rank-weighted return metric this module treats as primary.
    """

    n_trading_days: int
    table: pd.DataFrame


def significance_report(
    factor_data: pd.DataFrame,
    forward_periods: tuple[int, ...],
    min_trading_days: int = MIN_TRADING_DAYS,
) -> FactorSignificanceReport | None:
    """IC + quintile-spread + rank-weighted-return significance, each with a
    Newey-West (HAC) t-stat against zero, one row per forward period.

    Returns `None` if `factor_data` covers fewer than `min_trading_days`
    trading days - too few observations to trust a t-stat (matches
    alpha-pipeline's own `MIN_TRADING_DAYS` convention).
    """
    n_days = factor_data.index.get_level_values("date").nunique()
    if n_days < min_trading_days:
        return None

    ic = information_coefficient(factor_data)
    spread = quintile_spread(factor_data)
    rank_ret = rank_weighted_returns(factor_data)

    rows = []
    for period in forward_periods:
        col = f"{period}D"
        ic_mean, ic_t = newey_west_mean_and_tstat(ic[col], maxlags=period - 1)
        spread_mean, spread_t = newey_west_mean_and_tstat(spread[col], maxlags=period - 1)
        rank_mean, rank_t = newey_west_mean_and_tstat(rank_ret[col], maxlags=period - 1)
        rows.append(
            {
                "period": col,
                "mean_IC": ic_mean,
                "IC_t_stat_NW": ic_t,
                "Q_top_Q1_spread": spread_mean,
                "spread_t_stat_NW": spread_t,
                "rank_weighted_mean_return": rank_mean,
                "rank_weighted_t_stat_NW": rank_t,
            }
        )
    table = pd.DataFrame(rows).set_index("period")
    return FactorSignificanceReport(n_trading_days=n_days, table=table)


def run_significance_suite(
    factor: pd.Series,
    prices: pd.DataFrame,
    forward_periods: tuple[int, ...],
    regimes: dict[str, tuple[str | None, str | None]] | None = None,
    quantiles: int = DEFAULT_QUANTILES,
    min_trading_days: int = MIN_TRADING_DAYS,
) -> dict[str, FactorSignificanceReport | None]:
    """The one entry point most callers need: aligns `factor` against
    forward returns, then runs `significance_report` once per regime slice.

    Mirrors alpha-pipeline's `factor_common.run_significance_suite`,
    generalized so regime boundaries are caller-supplied rather than
    hardcoded - a specific rate-hike date is alpha-pipeline's own regime
    choice, not something this engine-independent module should assume.
    Defaults to a single `"full_period"` slice if no regimes are given.
    Returns the per-regime `FactorSignificanceReport` (or `None` where a
    regime slice was too short to trust) so callers can aggregate t-stats
    across factors - e.g. for a multiple-testing correction across
    everything tested in a session (see `flatten_tstats`).
    """
    regimes = regimes if regimes is not None else {"full_period": (None, None)}
    factor_data = build_factor_data(factor, prices, forward_periods, quantiles=quantiles)
    return {
        label: significance_report(slice_by_regime(factor_data, start, end), forward_periods, min_trading_days)
        for label, (start, end) in regimes.items()
    }


def flatten_tstats(
    label: str,
    reports: dict[str, FactorSignificanceReport | None],
    metrics: tuple[str, ...] = ("IC_t_stat_NW", "spread_t_stat_NW"),
) -> list[dict[str, float | str]]:
    """Flattens a `run_significance_suite` result into one row per
    (regime, period, metric) t-stat - the shape `multiple_testing_correction`
    (and alpha-pipeline's own `factor_test_multiple_testing_correction.py`)
    expects for pooling t-stats across every factor/regime/period tested in
    a research session."""
    rows: list[dict[str, float | str]] = []
    for regime, report in reports.items():
        if report is None:
            continue
        for period, row in report.table.iterrows():
            for metric in metrics:
                rows.append({"test": f"{label} | {regime} | {period} | {metric}", "t_stat": row[metric]})
    return rows


def bonferroni_critical_value(alpha: float, n_tests: int) -> float:
    """Two-sided Bonferroni-corrected critical |t| for a family of
    `n_tests` hypotheses at family-wise significance `alpha`."""
    if n_tests < 1:
        raise ValueError("n_tests must be >= 1")
    return float(norm.ppf(1 - (alpha / 2) / n_tests))


def multiple_testing_correction(tstats: list[dict[str, float | str]], alpha: float = DEFAULT_ALPHA) -> pd.DataFrame:
    """Bonferroni multiple-testing correction across every t-stat in
    `tstats` (see `flatten_tstats`) - the standard fix quant research
    applies before trusting a discovered factor's significance once many
    signals/parameters/regimes/periods have been tried against the same
    data (Harvey, Liu & Zhu 2016; `research-methodology/METHODOLOGY.md`
    section E, "Multiple-testing / selection leakage"). Adds `abs_t`,
    `bonferroni_critical`, and `survives_bonferroni` columns, sorted by
    `abs_t` descending.
    """
    table = pd.DataFrame(tstats)
    if table.empty:
        return table
    n = len(table)
    critical = bonferroni_critical_value(alpha, n)
    table = table.assign(abs_t=table["t_stat"].abs(), bonferroni_critical=critical)
    table["survives_bonferroni"] = table["abs_t"] > critical
    return table.sort_values("abs_t", ascending=False)

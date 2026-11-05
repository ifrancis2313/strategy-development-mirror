"""Tests `testing.monte_carlo.pbo.probability_of_backtest_overfitting`
against known statistical properties on synthetic data -- not implementation
details. The reference qualitative signature (per
research-methodology/vendored/pypbo-cscv/SOURCE.md's own synthetic
validation, reproduced here independently): a parameter grid of pure noise
scores a *high* PBO (~0.5, no real edge to distinguish, in-sample-best is
essentially a coin flip out-of-sample), while a grid containing genuine edge
scores a *markedly lower* PBO (~0.26-0.29, the same top performer tends to
hold up out-of-sample too).
"""

import os

import numpy as np
import pandas as pd
import pytest

from testing.monte_carlo.pbo import (
    _VENDORED_PYPBO_DIR,
    _annualized_sharpe_metric,
    probability_of_backtest_overfitting,
)

# The vendored pypbo-cscv code lives in the sibling research-methodology
# repo at a hardcoded absolute path (see pbo.py's own module docstring for
# why -- AGPL-3.0, deliberately never copied into this repo). That path
# only exists on a machine with both repos checked out side by side; a CI
# runner cloning just this repo won't have it. Skip the tests that need the
# real computation there (everything except the one that deliberately
# tests the missing-path failure case, which must always run).
_VENDORED_PYPBO_MISSING = not os.path.isdir(_VENDORED_PYPBO_DIR)
_requires_vendored_pypbo = pytest.mark.skipif(
    _VENDORED_PYPBO_MISSING,
    reason=f"vendored pypbo-cscv not found at {_VENDORED_PYPBO_DIR} on this machine/runner",
)


def _pure_noise_panel(rng: np.random.Generator, t: int = 1000, n: int = 20) -> pd.DataFrame:
    data = rng.normal(0, 0.01, size=(t, n))
    return pd.DataFrame(data, columns=[f"variant_{i}" for i in range(n)])


def _mixed_panel(rng: np.random.Generator, t: int = 1000, n: int = 20, n_edge: int = 3) -> pd.DataFrame:
    # Same scale as research-methodology's own vendoring benchmark: a small
    # but genuine drift (0.0006) relative to noise std (0.01), applied to
    # the first n_edge columns only.
    data = rng.normal(0, 0.01, size=(t, n))
    data[:, :n_edge] += 0.0006
    return pd.DataFrame(data, columns=[f"variant_{i}" for i in range(n)])


def test_annualized_sharpe_metric_scores_a_zero_variance_column_as_zero_not_via_division():
    """`np.where(std > 0, ...)` -- mutation testing (2026-07-12) found the
    boundary survives being loosened to `std >= 0`, which is always true and
    would attempt mean/0.0 for every column, not just the genuinely
    zero-variance one. A pure numpy unit test, no vendored pypbo needed."""
    metric = _annualized_sharpe_metric(periods_per_year=252.0)
    chunk = np.array([
        [1.0, 5.0],
        [1.0, 6.0],
        [1.0, 4.0],
    ])  # column 0 is constant (zero variance), column 1 varies
    result = metric(chunk)
    assert result[0] == 0.0
    assert np.isfinite(result[1])


@_requires_vendored_pypbo
def test_requires_at_least_two_columns():
    single = pd.DataFrame({"only": np.random.default_rng(0).normal(0, 1, 100)})
    with pytest.raises(ValueError, match="at least 2 parameterizations"):
        probability_of_backtest_overfitting(single, s_chunks=16)


@_requires_vendored_pypbo
def test_requires_even_s_chunks():
    rng = np.random.default_rng(1)
    panel = _pure_noise_panel(rng, t=500, n=5)
    with pytest.raises(ValueError, match="must be even"):
        probability_of_backtest_overfitting(panel, s_chunks=15)


@_requires_vendored_pypbo
def test_insufficient_periods_raises():
    rng = np.random.default_rng(2)
    panel = _pure_noise_panel(rng, t=10, n=5)
    with pytest.raises(ValueError, match="need >="):
        probability_of_backtest_overfitting(panel, s_chunks=16)


@_requires_vendored_pypbo
def test_drops_nan_rows_and_reports_the_count():
    rng = np.random.default_rng(3)
    panel = _pure_noise_panel(rng, t=1000, n=5)
    panel.iloc[:50, 0] = np.nan  # simulate a longer-lookback variant's warmup rows

    result = probability_of_backtest_overfitting(panel, s_chunks=16)

    assert result.n_periods_dropped == 50
    assert result.n_periods_used == 950


@_requires_vendored_pypbo
def test_pure_noise_panel_scores_a_high_pbo():
    rng = np.random.default_rng(42)
    panel = _pure_noise_panel(rng)

    result = probability_of_backtest_overfitting(panel, s_chunks=16, periods_per_year=252.0)

    assert result.n_chunks == 16
    assert result.n_combinations == 12_870
    assert result.pbo > 0.40


@_requires_vendored_pypbo
def test_real_edge_panel_scores_a_markedly_lower_pbo_than_pure_noise():
    # Same rng, drawn in the same order as research-methodology's own
    # vendoring benchmark (mixed panel first, then a pure-noise panel from
    # the same generator) -- reproduces PBO=0.259 (mixed) vs PBO=0.519
    # (noise) exactly; asserted here as a qualitative property (mixed <
    # noise, both within the documented reference band) rather than pinned
    # to exact floats, since the property under test is the statistical
    # signature, not a specific implementation detail.
    rng = np.random.default_rng(42)
    mixed = _mixed_panel(rng)
    noise = _pure_noise_panel(rng)

    mixed_result = probability_of_backtest_overfitting(mixed, s_chunks=16, periods_per_year=252.0)
    noise_result = probability_of_backtest_overfitting(noise, s_chunks=16, periods_per_year=252.0)

    assert mixed_result.pbo < noise_result.pbo
    assert 0.15 < mixed_result.pbo < 0.35
    assert 0.40 < noise_result.pbo < 0.60


@_requires_vendored_pypbo
def test_reproducible_across_calls():
    # CSCV/PBO itself has no internal randomness (only combinatorial
    # chunking of the given data) -- identical input must give a
    # bit-for-bit identical result.
    rng = np.random.default_rng(5)
    panel = _mixed_panel(rng, t=800, n=10, n_edge=2)

    first = probability_of_backtest_overfitting(panel, s_chunks=16)
    second = probability_of_backtest_overfitting(panel, s_chunks=16)

    assert first == second


def test_missing_vendored_pypbo_raises_a_clear_import_error(monkeypatch):
    import testing.monte_carlo.pbo as pbo_module

    monkeypatch.setattr(pbo_module, "_VENDORED_PYPBO_DIR", "/nonexistent/path/pypbo-cscv")
    rng = np.random.default_rng(6)
    panel = _pure_noise_panel(rng, t=500, n=5)

    with pytest.raises(ImportError, match="Vendored pypbo-cscv not found"):
        probability_of_backtest_overfitting(panel, s_chunks=16)


def test_vendored_pypbo_directory_is_configurable_through_public_api(monkeypatch, tmp_path):
    import testing.monte_carlo.pbo as pbo_module

    requested_dir = str(tmp_path)
    loaded_dirs = []

    class FakePypbo:
        @staticmethod
        def pbo(*args, **kwargs):
            return type("FakeResult", (), {"pbo": 0.25, "prob_oos_loss": 0.5})()

    def fake_loader(vendored_dir=None):
        loaded_dirs.append(vendored_dir)
        return FakePypbo

    monkeypatch.setattr(pbo_module, "_load_vendored_pypbo", fake_loader)
    panel = pd.DataFrame(np.random.default_rng(7).normal(size=(32, 2)))

    result = probability_of_backtest_overfitting(panel, s_chunks=16, vendored_pypbo_dir=requested_dir)

    assert loaded_dirs == [requested_dir]
    assert result.pbo == 0.25

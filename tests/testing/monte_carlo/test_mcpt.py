"""Tests `testing.monte_carlo.mcpt.sign_permutation_mcpt` against known
statistical properties on synthetic data -- not implementation details. The
central property under test (per this issue's acceptance criteria): a
strategy with genuinely no edge must NOT produce a significant p-value on
synthetic zero-edge data, while a strategy with a real, consistent edge
must.
"""

import numpy as np

from testing.monte_carlo.mcpt import annualized_sharpe, sign_permutation_mcpt


def test_empty_returns_is_degenerate():
    assert sign_permutation_mcpt([], n_permutations=10) is None


def test_single_observation_is_degenerate():
    # ddof=1 std needs >= 2 observations.
    assert sign_permutation_mcpt([5.0], n_permutations=10) is None


def test_zero_variance_returns_is_degenerate():
    assert sign_permutation_mcpt([100.0, 100.0, 100.0], n_permutations=10) is None


def test_annualized_sharpe_returns_a_real_value_at_exactly_two_observations():
    """`returns.size < 2` -- mutation testing (2026-07-12) found this
    boundary untested: at size==2 exactly with the `< 2` guard (correct),
    size==2 is NOT degenerate and a real Sharpe is computed; the buggy
    `<= 2` mutant would return nan here instead."""
    result = annualized_sharpe(np.array([3.0, 7.0]), periods_per_year=252.0)
    assert np.isfinite(result)


def test_p_value_boundary_counts_a_null_draw_tied_with_observed():
    """`valid_null >= observed`, not `>` -- the actual p-value numerator.
    Mutation testing (2026-07-12) found this untested since no prior test
    forced an exact tie between a null draw and the observed statistic.
    Found by direct search (not hand-derived): returns=[0, 1, -2] at
    n_permutations=3 has permutation index 2 (seed 2) reproduce the
    observed Sharpe bit-for-bit, while permutations 0/1 exceed it --
    valid_null=[+obs, +obs, observed], so >= counts all 3 (p=1.0) but the
    buggy `>` would count only 2 (p=0.75)."""
    returns = np.array([0.0, 1.0, -2.0])
    result = sign_permutation_mcpt(returns, periods_per_year=252.0, n_permutations=3)

    assert result is not None
    assert result.n_permutations == 3
    assert np.isclose(result.null_distribution[2], result.observed_sharpe)  # confirms the exact tie
    assert result.p_value == 1.0


def test_p_value_matches_the_documented_formula_for_a_single_permutation():
    # n_permutations=1 is fully deterministic (rng seed 0): independently
    # derived via plain numpy (see this test's inline recomputation, not
    # copied from the implementation) -- seed-0's rng.choice([1,-1], size=3)
    # flips all three signs, so the null Sharpe is exactly -observed and
    # p_value = (1 + count(null >= observed)) / (1 + 1) = 0.5.
    returns = np.array([10.0, -5.0, 8.0])
    observed = annualized_sharpe(returns, periods_per_year=3.0)

    result = sign_permutation_mcpt(returns, periods_per_year=3.0, n_permutations=1)

    assert result is not None
    assert np.isclose(result.observed_sharpe, observed)
    assert np.isclose(result.observed_sharpe, 0.921546, atol=1e-5)
    assert result.n_permutations == 1
    assert np.isclose(result.null_distribution[0], -result.observed_sharpe)
    assert result.p_value == 0.5


def test_p_value_is_low_for_a_strong_consistent_edge():
    # 20 observations, all strongly positive with modest noise -> real
    # Sharpe is large and positive; sign-flipping roughly half the
    # observations in most permutations should rarely produce a null Sharpe
    # that beats it.
    rng = np.random.default_rng(7)
    pnl = 100.0 + rng.normal(0, 5, size=20)

    result = sign_permutation_mcpt(pnl, periods_per_year=12.0, n_permutations=500)

    assert result is not None
    assert result.observed_sharpe > 0
    assert result.p_value is not None
    assert result.p_value < 0.05


def test_p_value_is_not_significant_for_genuinely_zero_edge_synthetic_data():
    # The core acceptance-criterion property: permuting a strategy with no
    # real edge must not manufacture significance. 300 draws of mean-zero
    # Gaussian noise -- no directional edge by construction -- should land
    # comfortably above a 0.05 (or even 0.10) significance bar.
    rng = np.random.default_rng(42)
    zero_edge_returns = rng.normal(0, 1, size=300)

    result = sign_permutation_mcpt(zero_edge_returns, periods_per_year=252.0, n_permutations=500)

    assert result is not None
    assert result.p_value is not None
    assert result.p_value > 0.10


def test_null_distribution_length_matches_n_permutations_when_all_valid():
    rng = np.random.default_rng(11)
    returns = rng.normal(1.0, 2.0, size=15)

    result = sign_permutation_mcpt(returns, periods_per_year=252.0, n_permutations=200)

    assert result is not None
    assert result.n_permutations == 200
    assert len(result.null_distribution) == 200


def test_reproducible_across_calls():
    # Each permutation draw is seeded by its own index
    # (np.random.default_rng(i)), so re-running with the same input must be
    # bit-for-bit reproducible -- required for a robustness result to be a
    # trustworthy audit artifact.
    returns = [12.0, -8.0, 15.0, -3.0, 9.0, -11.0, 4.0, -2.0]

    first = sign_permutation_mcpt(returns, periods_per_year=252.0, n_permutations=100)
    second = sign_permutation_mcpt(returns, periods_per_year=252.0, n_permutations=100)

    assert first is not None and second is not None
    assert first.observed_sharpe == second.observed_sharpe
    assert first.p_value == second.p_value
    assert np.array_equal(first.null_distribution, second.null_distribution)


def test_accepts_pandas_series_and_plain_list_identically():
    import pandas as pd

    values = [3.0, -1.0, 4.0, -1.5, 2.5]
    from_list = sign_permutation_mcpt(values, periods_per_year=252.0, n_permutations=50)
    from_series = sign_permutation_mcpt(pd.Series(values), periods_per_year=252.0, n_permutations=50)

    assert from_list is not None and from_series is not None
    assert from_list.observed_sharpe == from_series.observed_sharpe
    assert from_list.p_value == from_series.p_value

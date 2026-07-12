import numpy as np
import pandas as pd

from testing.viz.monte_carlo import mcpt_null_distribution_chart


def _strong_edge_pnl(n: int = 60, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    # Consistent positive drift with modest noise -- a real, permutation-detectable edge.
    return list(rng.normal(loc=50.0, scale=10.0, size=n))


def _equity(pnl: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(pnl), freq="B")
    return pd.Series(pnl, index=idx, dtype=float).cumsum()


def test_renders_histogram_and_observed_marker_for_a_real_edge() -> None:
    pnl = _strong_edge_pnl()
    fig = mcpt_null_distribution_chart("demo_strategy", pnl, _equity(pnl), n_permutations=300)

    assert fig is not None
    assert fig.data[0].type == "histogram"
    assert len(fig.data[0].x) == 300
    # observed Sharpe marked as a vline (shape), not folded into the histogram itself
    assert len(fig.layout.shapes) >= 1
    assert any("Observed Sharpe" in (a.text or "") for a in fig.layout.annotations)


def test_returns_none_for_no_trades() -> None:
    fig = mcpt_null_distribution_chart("empty", [], pd.Series(dtype=float))
    assert fig is None


def test_returns_none_for_zero_variance_pnl() -> None:
    pnl = [100.0] * 10
    fig = mcpt_null_distribution_chart("flat", pnl, _equity(pnl))
    assert fig is None

import numpy as np
import pandas as pd

from testing.engine.realized_vol import compute_realized_vol


def test_realized_vol_matches_manual_calculation_on_known_series():
    # constant daily log return of 1% -> rolling stdev of returns is 0 (no variance)
    spot = pd.Series([100 * (1.01 ** i) for i in range(10)])

    rv = compute_realized_vol(spot, window=5)

    assert np.isclose(rv.iloc[5:], 0.0, atol=1e-8).all()


def test_realized_vol_window_matches_numpy_std_directly():
    rng = np.random.default_rng(42)
    spot = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 50))))

    rv = compute_realized_vol(spot, window=21)

    log_returns = np.log(spot / spot.shift(1))
    expected_at_30 = log_returns.iloc[10:31].std() * np.sqrt(252)
    assert np.isclose(rv.iloc[30], expected_at_30)

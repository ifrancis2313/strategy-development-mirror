"""No real network calls -- `fetch_vix_history` is monkeypatched (same style
as tests/shadow/test_tastytrade_broker.py's `yf.download` mock), and the
cache round-trip is exercised against a real tmp_path CSV, not yfinance.
"""

from __future__ import annotations

import pandas as pd

from testing.engine import vix_source


def _fake_vix_series() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    return pd.Series(range(10, 20), index=idx, dtype=float, name="vix")


def test_load_or_fetch_vix_history_fetches_and_caches_on_miss(tmp_path, monkeypatch):
    cache_path = tmp_path / "vix_history.csv"
    fetched = []

    def fake_fetch():
        fetched.append(True)
        return _fake_vix_series()

    monkeypatch.setattr(vix_source, "fetch_vix_history", fake_fetch)

    result = vix_source.load_or_fetch_vix_history("2020-01-03", "2020-01-06", path=str(cache_path))

    assert fetched == [True]
    assert cache_path.exists()
    assert list(result.index) == list(pd.date_range("2020-01-03", "2020-01-06"))
    assert result.loc[pd.Timestamp("2020-01-03")] == 12.0


def test_load_or_fetch_vix_history_reads_cache_without_fetching_on_hit(tmp_path, monkeypatch):
    cache_path = tmp_path / "vix_history.csv"
    _fake_vix_series().rename_axis("date").to_frame("vix").to_csv(cache_path)

    def fail_fetch():
        raise AssertionError("should not fetch when cache already exists")

    monkeypatch.setattr(vix_source, "fetch_vix_history", fail_fetch)

    result = vix_source.load_or_fetch_vix_history("2020-01-01", "2020-01-10", path=str(cache_path))

    assert len(result) == 10
    assert result.iloc[0] == 10.0


def test_load_or_fetch_vix_history_slices_to_the_requested_window(tmp_path, monkeypatch):
    cache_path = tmp_path / "vix_history.csv"
    monkeypatch.setattr(vix_source, "fetch_vix_history", _fake_vix_series)

    result = vix_source.load_or_fetch_vix_history("2020-01-05", "2020-01-07", path=str(cache_path))

    assert len(result) == 3
    assert result.index.min() == pd.Timestamp("2020-01-05")
    assert result.index.max() == pd.Timestamp("2020-01-07")

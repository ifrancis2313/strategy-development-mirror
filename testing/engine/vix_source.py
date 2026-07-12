"""Free/public VIX index history via yfinance (`^VIX`) -- hedge-candidate
search option 3 (AGENTS.md 2026-07-10). Only the VIX *index* level, not
VIX options/futures -- those aren't in the existing `Options_Trading_Strategy
/data/all.parquet` OptionMetrics extract and would need separate sourcing
(a bigger, still-open decision) if a strategy ever wanted to actually trade
VIX itself rather than just use its level as a sizing/gating signal. Confirmed
free-data-first per standing preference: no paid vendor needed for the index
level, which is all `SizingEngine.size_vol_scaled` (see that method's
docstring -- originally VIX-only, generalized to any same-scale vol series
when RV-scaled sizing was wired up first) actually requires.

Same ingestion source (`yfinance`) `ingestion/sources/yfinance_source.py`
already uses for the shared Supabase project's live pulls (e.g.
`fetch_risk_free_rate`'s `^IRX`), but this module is Testing-side: a
historical-window fetch-and-cache for backtesting, not a live daily pull for
Ingestion's writer path. Deliberately not routed through `ingestion/` --
Testing has no dependency on Ingestion/Supabase (see `testing/CONTEXT.md`),
and a local CSV cache is all a closed 1996-2019 historical window needs.
"""

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_PATH = "data/vix_history.csv"


def fetch_vix_history() -> pd.Series:
    """Full available `^VIX` daily close history from yfinance (1990-present),
    in raw VIX points (e.g. 20.5, not 0.205) -- NOT the same scale as
    `testing.engine.realized_vol.compute_realized_vol`'s fractional output.
    `SizingEngine.size_vol_scaled` only cares about the ratio between
    `vol_by_date` and `reference_vol`, so this is fine as long as both are
    VIX-point-scaled together -- never mix a VIX-point series with an
    RV-fraction reference or vice versa (see that method's docstring).
    """
    import yfinance as yf

    hist = yf.Ticker("^VIX").history(period="max")
    if hist.empty:
        raise ValueError("yfinance returned no ^VIX history -- check network connectivity")
    series = hist["Close"].copy()
    series.index = pd.DatetimeIndex(series.index.tz_localize(None).normalize())
    return series.rename("vix").sort_index()


def load_or_fetch_vix_history(
    start_date: str, end_date: str, path: str = DEFAULT_CACHE_PATH,
) -> pd.Series:
    """Opt-in cache mirroring `testing.engine.chain_cache.
    load_or_build_spx_chain`'s pattern: cache-hit reads the local CSV
    directly (no network call), cache-miss fetches the full available
    history once and writes it, so repeated backtest runs (or multiple
    strategies' scripts) don't each re-hit yfinance. No invalidation logic
    for the same reason that cache has none -- a closed historical window
    (1996-2019) never changes; delete the file manually to force a refetch
    of more recent data.
    """
    cache_path = Path(path)
    if cache_path.exists():
        full = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")["vix"]
    else:
        full = fetch_vix_history()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        full.rename_axis("date").to_frame("vix").to_csv(cache_path)

    start_ts, end_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
    return full[(full.index >= start_ts) & (full.index <= end_ts)]

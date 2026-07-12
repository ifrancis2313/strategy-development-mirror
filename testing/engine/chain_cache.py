from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from testing.engine.backtest_runner import CHAIN_COLUMNS
from testing.engine.chain_repository import OptionChainRepository

DEFAULT_CACHE_DIR = "data/generated"


def _default_cache_path(secid: int, start_date: str, end_date: str) -> str:
    """Derives a cache filename from the actual query params, so two calls
    with different `(secid, start_date, end_date)` can never collide on the
    same file and silently return each other's data -- the bug this
    function used to have (a single fixed `DEFAULT_CACHE_PATH` regardless
    of what was asked for), found during the front-to-back review
    (2026-07-12) and fixed here rather than left as a documented landmine.
    `testing.strategies.dispersion.basket` already independently worked
    around this exact gap by building its own per-name cache instead of
    reusing this function -- this fix removes the reason it had to.
    """
    return f"{DEFAULT_CACHE_DIR}/spx_chain_cache_{secid}_{start_date}_{end_date}.parquet"


def load_or_build_spx_chain(
    repo: OptionChainRepository,
    secid: int = 108105,
    start_date: str = "1996-01-01",
    end_date: str = "2019-12-31",
    columns: Sequence[str] | None = None,
    path: str | None = None,
) -> pd.DataFrame:
    """Opt-in cache for the full historical chain query every backtest script
    currently repeats independently (`repo.get_chain(108105, "1996-01-01",
    "2019-12-31", columns=CHAIN_COLUMNS)` against the 14GB `all.parquet`).
    Exists so multiple strategies' scripts -- now potentially run by separate
    subagents, possibly concurrently -- don't each pay that cost separately.

    `path=None` (the default) derives a cache filename from `(secid,
    start_date, end_date)` via `_default_cache_path` -- pass `path` explicitly
    only to pin a specific file (e.g. a test's temp path); the auto-derived
    default is the safe choice for real callers and can't collide across
    different query params.

    Cache-miss: runs the same `OptionChainRepository.get_chain` call any
    existing script already makes, writes the (much smaller) result to
    `path`. Cache-hit: reads `path` directly, no repository query at all --
    but first verifies every requested `columns` name is actually present in
    the cached file, raising rather than silently returning a DataFrame
    missing columns the caller asked for (e.g. a cache built by an earlier,
    narrower-columns call at the same explicit `path`).

    No invalidation logic -- the 1996-2019 historical window this defaults to
    is closed and never changes, so a stale cache isn't a real risk the way
    it would be for a live/growing dataset. Delete the cache file manually if
    the underlying `all.parquet` is ever replaced.

    Deliberately opt-in: existing scripts keep calling `repo.get_chain(...)`
    directly and are unaffected. Only new/updated scripts adopt this.
    """
    resolved_columns = list(columns) if columns is not None else list(CHAIN_COLUMNS)
    cache_path = Path(path) if path is not None else Path(_default_cache_path(secid, start_date, end_date))

    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        missing = set(resolved_columns) - set(cached.columns)
        if missing:
            raise ValueError(
                f"{cache_path} is missing requested columns {sorted(missing)} -- "
                "it was likely built by an earlier call with a narrower `columns` "
                "list at this same path; delete the cache file or use a different `path`."
            )
        return cached

    chain = repo.get_chain(secid, start_date, end_date, columns=resolved_columns)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(cache_path, index=False)
    return chain

"""Free data source: underlying prices and option chains via yfinance.
No API key, no cost -- Yahoo's quotes are delayed ~15-20min, and yfinance
supplies its own implied vol per contract but no Greeks (those are computed
in ingestion/greeks.py). Ported verbatim from Options and Futures Pipeline's
`pipeline/block1_ingestion/sources/yfinance_source.py` -- this is Ingestion's
first real, working data source (per strategy-development's PRD, Ingestion
becomes the sole writer to the shared Supabase project; OFP's own copy of
this module is retired from writing once the cutover lands)."""

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd
import yfinance as yf

RISK_FREE_PROXY_TICKER = "^IRX"  # 13-week T-bill discount rate, in percent


@dataclass
class RawChain:
    symbol: str
    spot_price: float
    as_of: datetime
    # expiry (YYYY-MM-DD str, as yfinance reports it) -> (calls, puts)
    expiries: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = field(default_factory=dict)


def fetch_risk_free_rate() -> float:
    """Latest ^IRX close, converted from a percentage quote (e.g. 5.25) to a
    decimal rate (0.0525). Falls back to 0.0 if the ticker returns nothing --
    a stale-zero rate is a safer failure mode than crashing ingestion."""
    hist = yf.Ticker(RISK_FREE_PROXY_TICKER).history(period="5d")
    if hist.empty:
        return 0.0
    return float(hist["Close"].iloc[-1]) / 100


def fetch_raw_chain(symbol: str, max_expiries: int | None = None) -> RawChain:
    """Pulls every expiry's calls/puts for `symbol`. `max_expiries` caps how
    many expiries are pulled (each is a separate network call) -- useful for
    a fast smoke test against a symbol with many listed expiries."""
    ticker = yf.Ticker(symbol)
    spot = _fetch_spot(ticker)
    expiries = ticker.options
    if max_expiries is not None:
        expiries = expiries[:max_expiries]

    raw = RawChain(symbol=symbol, spot_price=spot, as_of=datetime.now(UTC))
    for expiry in expiries:
        chain = ticker.option_chain(expiry)
        raw.expiries[expiry] = (chain.calls, chain.puts)
    return raw


def _fetch_spot(ticker: yf.Ticker) -> float:
    fast_info = ticker.fast_info
    price = fast_info.get("lastPrice") if fast_info else None
    if price:
        return float(price)
    hist = ticker.history(period="1d")
    if hist.empty:
        raise ValueError(f"no spot price available for {ticker.ticker}")
    return float(hist["Close"].iloc[-1])

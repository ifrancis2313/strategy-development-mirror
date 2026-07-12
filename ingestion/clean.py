"""Raw vendor frames -> ChainSnapshot. This is the only place vendor-specific
column names/quirks are handled; everything past this point only ever sees
ChainSnapshot/ContractRow. Ported verbatim from Options and Futures
Pipeline's `pipeline/block1_ingestion/clean.py`."""

from datetime import date, datetime
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from ingestion import greeks
from ingestion.schema import ChainSnapshot, ContractRow
from ingestion.sources.yfinance_source import RawChain

_CALENDAR_DAYS_PER_YEAR = 365.0  # options T decays over real calendar time, not trading days -- 252 belongs to return annualization (realized_vol.py), not here
_NYSE = xcals.get_calendar("XNYS")


def clean_raw_chain(raw: RawChain, risk_free_rate: float, realized_vol_21d: float | None = None) -> ChainSnapshot:
    contracts: list[ContractRow] = []
    for expiry_str, (calls, puts) in raw.expiries.items():
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        time_to_expiry = _year_fraction(raw.as_of, expiry)
        if time_to_expiry <= 0:
            continue  # expiry date has fully passed, or (for a same-day
            # contract) the NYSE close for that date has already occurred
        contracts.extend(_clean_side(calls, "C", expiry, raw.spot_price, risk_free_rate, time_to_expiry))
        contracts.extend(_clean_side(puts, "P", expiry, raw.spot_price, risk_free_rate, time_to_expiry))

    return ChainSnapshot(
        symbol=raw.symbol,
        spot_price=raw.spot_price,
        as_of=raw.as_of,
        risk_free_rate=risk_free_rate,
        source="yfinance",
        contracts=contracts,
        realized_vol_21d=realized_vol_21d,
    )


def _year_fraction(as_of: datetime, expiry: date) -> float:
    """Time to expiry as a fraction of a year. A same-day (0DTE) contract
    is NOT expired just because the calendar-day delta is zero -- it still
    has real time value until the NYSE close on that date. Use the actual
    fraction of the session remaining for day-of expiries instead of
    flooring to zero, which previously caused every 0DTE contract to be
    silently dropped from every intraday snapshot."""
    days_between = (expiry - as_of.date()).days
    if days_between > 0:
        return days_between / _CALENDAR_DAYS_PER_YEAR
    if days_between < 0:
        return -1.0  # expiry date has already fully passed

    try:
        close = _NYSE.session_close(pd.Timestamp(expiry))
    except Exception:
        return -1.0  # expiry isn't a valid NYSE session; treat as expired

    remaining_seconds = (close - pd.Timestamp(as_of)).total_seconds()
    if remaining_seconds <= 0:
        return -1.0  # past today's close
    return (remaining_seconds / 86400.0) / _CALENDAR_DAYS_PER_YEAR


def _clean_side(df: pd.DataFrame, right: Literal["C", "P"], expiry: date, spot: float,
                 risk_free_rate: float, time_to_expiry: float) -> list[ContractRow]:
    if df.empty:
        return []
    df = df.copy()
    df["volume"] = df["volume"].fillna(0).astype(int)
    df["openInterest"] = df["openInterest"].fillna(0).astype(int)
    df["bid"] = df["bid"].fillna(0.0)
    df["ask"] = df["ask"].fillna(0.0)

    iv = _resolve_iv(df, right, spot, risk_free_rate, time_to_expiry)
    S = np.full(len(df), spot)
    K = df["strike"].to_numpy(dtype=float)
    T = np.full(len(df), time_to_expiry)
    r = np.full(len(df), risk_free_rate)
    rights = np.full(len(df), right)

    valid_iv = iv > 0
    g = greeks.all_greeks(S[valid_iv], K[valid_iv], T[valid_iv], r[valid_iv], iv[valid_iv], rights[valid_iv])

    rows = []
    g_idx = 0
    for pos, (_, row) in enumerate(df.iterrows()):
        row_iv = float(iv[pos])
        has_greeks = valid_iv[pos]
        rows.append(ContractRow(
            expiry=expiry,
            strike=float(row["strike"]),
            right=right,
            bid=float(row["bid"]),
            ask=float(row["ask"]),
            last=float(row["lastPrice"]) if pd.notna(row.get("lastPrice")) else None,
            volume=int(row["volume"]),
            open_interest=int(row["openInterest"]),
            implied_vol=row_iv if row_iv > 0 else None,
            delta=float(g["delta"][g_idx]) if has_greeks else None,
            gamma=float(g["gamma"][g_idx]) if has_greeks else None,
            theta=float(g["theta"][g_idx]) if has_greeks else None,
            vega=float(g["vega"][g_idx]) if has_greeks else None,
            source="yfinance",
        ))
        if has_greeks:
            g_idx += 1
    return rows


def _resolve_iv(df: pd.DataFrame, right: str, spot: float, risk_free_rate: float,
                 time_to_expiry: float) -> np.ndarray:
    """Prefer yfinance's own IV; back it out from the mid price via Brent's
    method only for the rows where Yahoo didn't supply one (rare, but happens
    on thin-liquidity strikes)."""
    iv = df["impliedVolatility"].fillna(0.0).to_numpy(dtype=float).copy()
    missing = np.where(iv <= 0)[0]
    for pos in missing:
        row = df.iloc[pos]
        mid = (row["bid"] + row["ask"]) / 2 or row.get("lastPrice", 0.0)
        if not mid or mid <= 0:
            continue
        solved = greeks.implied_vol(mid, spot, float(row["strike"]), time_to_expiry, risk_free_rate, right)
        if solved is not None:
            iv[pos] = solved
    return iv

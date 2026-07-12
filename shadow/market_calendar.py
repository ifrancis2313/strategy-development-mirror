"""NYSE trading-day guard (ADR 0009), ported from `Options_Trading_Strategy`'s
`core/market_calendar.py` -- Shadow's daily composition scripts (issue 13)
call this before doing anything, so they no-op cleanly on US market
holidays instead of marking open ledgers against a stale/missing quote.
Weekends are excluded by cron already, but holidays (and the occasional
mid-week close) need the authoritative exchange calendar.

Backed by `exchange_calendars` (XNYS), same library OTS already depends on
-- no new dependency introduced.
"""

from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

_NYSE = xcals.get_calendar("XNYS")


def is_trading_day(date) -> bool:
    """True iff `date` is a regular NYSE trading session (handles weekends
    and holidays). Early-close days (e.g. day after Thanksgiving) ARE
    sessions -- a mid-morning run on them still returns True."""
    return bool(_NYSE.is_session(pd.Timestamp(date).normalize()))

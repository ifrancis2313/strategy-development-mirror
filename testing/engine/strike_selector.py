from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrikeMatch:
    optionid: object
    strike_price: float
    delta: float
    best_bid: float
    best_offer: float
    impl_volatility: float | None = None

    @classmethod
    def from_row(cls, row: pd.Series) -> "StrikeMatch":
        """Build a match from a single chain row. Shared by StrikeSelector and
        any strategy-specific lookup (e.g. the calendar's exact-strike match)
        so the row->match mapping lives in one place.
        """
        return cls(
            optionid=row["optionid"],
            strike_price=row["strike_price"],
            delta=row["delta"],
            best_bid=row["best_bid"],
            best_offer=row["best_offer"],
            impl_volatility=row["impl_volatility"] if "impl_volatility" in row.index else None,
        )


def select_target_expiry(
    chain: pd.DataFrame, entry_date, min_dte: int, max_dte: int
) -> tuple[pd.Timestamp, pd.DataFrame] | None:
    """Among expiries within the [min_dte, max_dte] window, pick the one nearest
    the window's midpoint and return (chosen_exdate, that expiry's chain slice),
    or None if no expiry falls in the window.

    Generic chain mechanics shared by every strategy's trade selector -- the
    "which expiry do I trade on this day" decision is the same regardless of
    what structure or signal a strategy layers on top.
    """
    entry_date = pd.Timestamp(entry_date)
    dte = (chain["exdate"] - entry_date).dt.days
    eligible = chain[(dte >= min_dte) & (dte <= max_dte)]
    if eligible.empty:
        return None

    target_dte = (min_dte + max_dte) / 2
    eligible_dte = (eligible["exdate"] - entry_date).dt.days
    chosen_exdate = pd.Timestamp(eligible.loc[(eligible_dte - target_dte).abs().idxmin(), "exdate"])
    return chosen_exdate, eligible[eligible["exdate"] == chosen_exdate]


class StrikeSelector:
    """Picks the contract nearest a target delta, on one side (puts or calls)
    of a single-expiry chain slice. Shared by every leg of every structure
    rather than duplicated per leg/structure.

    Returns None -- rather than forcing a pick -- when nothing is within
    `max_delta_distance` of the target, so thin-liquidity days (more common
    pre-2000) don't silently corrupt trade economics with a wrong-delta fill.
    """

    def __init__(self, max_delta_distance: float = 0.05):
        self.max_delta_distance = max_delta_distance

    def select(self, chain: pd.DataFrame, target_delta: float, cp_flag: str) -> StrikeMatch | None:
        candidates = chain[chain["cp_flag"] == cp_flag].dropna(subset=["delta"])
        if candidates.empty:
            return None

        distances = (candidates["delta"] - target_delta).abs()
        best_idx = distances.idxmin()
        if distances.loc[best_idx] > self.max_delta_distance:
            return None

        return StrikeMatch.from_row(candidates.loc[best_idx])

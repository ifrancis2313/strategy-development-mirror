import pandas as pd


class SyntheticSpotEngine:
    """Derives a daily underlying spot proxy via put-call parity.

    `forward_price` is not populated in this data extract (confirmed zero across
    the entire file, all secids), so this falls back to deriving a synthetic
    forward directly from matched put/call quotes: for a strike K with mid
    prices C and P, K + C - P approximates the forward price F (parity, ignoring
    the short-dated discounting term, which is immaterial here since only
    day-over-day returns feed realized vol, not absolute levels). Noise is
    reduced by using the strikes nearest the C=P crossing point (closest to the
    money) and taking their median, per expiry, per day.
    """

    def __init__(self, target_dte: int = 30, n_strikes: int = 5):
        self.target_dte = target_dte
        self.n_strikes = n_strikes

    def _spot_for_day(self, day_chain: pd.DataFrame) -> float:
        dte = (day_chain["exdate"] - day_chain["date"]).dt.days
        nearest_exdate = day_chain.loc[(dte - self.target_dte).abs().idxmin(), "exdate"]
        expiry_chain = day_chain[day_chain["exdate"] == nearest_exdate]

        calls = expiry_chain[expiry_chain["cp_flag"] == "C"].set_index("strike_price")
        puts = expiry_chain[expiry_chain["cp_flag"] == "P"].set_index("strike_price")
        strikes = calls.index.intersection(puts.index)
        if len(strikes) == 0:
            return float("nan")

        mid_call = (calls.loc[strikes, "best_bid"] + calls.loc[strikes, "best_offer"]) / 2
        mid_put = (puts.loc[strikes, "best_bid"] + puts.loc[strikes, "best_offer"]) / 2
        synthetic_forward = pd.Series(strikes, index=strikes) + mid_call - mid_put

        crossing_distance = (mid_call - mid_put).abs().sort_values()
        nearest_strikes = crossing_distance.index[: self.n_strikes]
        return synthetic_forward.loc[nearest_strikes].median()

    def compute_spot_series(self, chain: pd.DataFrame) -> pd.Series:
        if chain.empty:
            return pd.Series(dtype=float)
        spots = {day: self._spot_for_day(day_chain) for day, day_chain in chain.groupby("date")}
        return pd.Series(spots, dtype=float).dropna().sort_index().rename("spot")

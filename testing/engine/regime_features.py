"""Causal (no-lookahead) daily leading-indicator features for regime-
transition risk -- parked research question #3 (AGENTS.md 2026-07-10).

Every feature here is computed using only data through day t (rolling/
trailing windows, forward-fill for calendar mismatches, never a full-sample
fit). This is the property the HMM regime label itself does NOT have --
`Isaiah's Summer Options project/src/regime.py`'s `RegimeClassifier.fit`
runs Baum-Welch smoothing over the entire 1996-2019 series at once, so a
label at day t can be informed by data after day t. `regime_labels.csv` is
therefore fine to use as a *target* (something you're trying to forecast
ahead of time) but must never be used, or have anything derived from it in
a lookahead way, as an *input feature*.

Feature design note: these are deliberately RATES OF CHANGE / trailing
percentiles rather than the raw levels (`vix`, `vrp`, `momentum`,
`opt_spread`) the HMM itself already conditions on -- the HMM's own level
features are, by construction, what defines the current state; a precursor
signal needs to characterize how those levels are *moving*, not restate the
state.
"""

from __future__ import annotations

import pandas as pd

from testing.engine.realized_vol import compute_realized_vol

#: Near-ATM, liquid delta band (both puts and calls, sign-agnostic) used for
#: the daily median bid-ask spread feature -- mirrors the sibling project's
#: `opt_spread` feature intent (options bid-ask spread widens as liquidity
#: dries up ahead of stress) without importing that project's code.
SPREAD_DELTA_BAND = (0.35, 0.65)


def compute_transition_distances(regime_df: pd.DataFrame) -> pd.DataFrame:
    """For each labeled trading day, trading days since the regime label
    last changed and trading days until it next changes. `days_since` is
    real-time knowable (purely backward-looking); `days_until` requires the
    future label sequence and is for target construction only, never as an
    input feature. Ported from `scripts/regime_transition_proximity_diagnostic.py`
    (first written there as a one-off diagnostic) into shared, reusable code.
    """
    df = regime_df.sort_values("odate").reset_index(drop=True)
    changed = df["regime"] != df["regime"].shift(1)
    change_idx = df.index[changed].tolist()

    since = []
    until = []
    for i in range(len(df)):
        prior_changes = [c for c in change_idx if c <= i]
        last_change = prior_changes[-1] if prior_changes else 0
        since.append(i - last_change)

        next_changes = [c for c in change_idx if c > i]
        until.append(next_changes[0] - i if next_changes else None)

    df["days_since_transition"] = since
    df["days_until_transition"] = until
    return df


def compute_daily_opt_spread(chain: pd.DataFrame, delta_band: tuple[float, float] = SPREAD_DELTA_BAND) -> pd.Series:
    """Daily median relative bid-ask spread across near-ATM, liquid
    contracts (both puts and calls, `|delta|` in `delta_band`). Purely a
    same-day cross-sectional statistic -- no rolling window, so it is
    trivially causal (uses only day-t quotes).
    """
    band = chain[chain["delta"].abs().between(delta_band[0], delta_band[1])].copy()
    mid = (band["best_bid"] + band["best_offer"]) / 2
    valid = mid > 0
    band = band[valid]
    mid = mid[valid]
    rel_spread = (band["best_offer"] - band["best_bid"]) / mid
    return rel_spread.groupby(band["date"]).median().rename("opt_spread")


def build_causal_features(
    vix: pd.Series,
    spot: pd.Series,
    chain: pd.DataFrame,
    *,
    momentum_window: int = 21,
    reversal_lookback: int = 10,
    rv_short_window: int = 5,
    rv_long_window: int = 20,
    vix_percentile_window: int = 252,
) -> pd.DataFrame:
    """Builds the causal feature panel on the chain's own trading-day
    calendar (VIX/spot are reindexed onto it with forward-fill -- carrying
    the last known value forward is causal; anything else would not be).

    Returns a DataFrame indexed by date with columns:
    `vix_chg_1d`, `vix_chg_5d`, `vix_chg_10d`, `vix_percentile_252d`,
    `vrp`, `vrp_chg_5d`, `rv_accel`, `momentum_21d`, `momentum_reversal`,
    `opt_spread`, `opt_spread_chg_5d`.
    """
    trading_days = pd.DatetimeIndex(sorted(chain["date"].unique()))
    vix = vix.reindex(trading_days, method="ffill")
    spot = spot.reindex(trading_days, method="ffill")

    vix_chg_1d = vix.pct_change(1).rename("vix_chg_1d")
    vix_chg_5d = vix.pct_change(5).rename("vix_chg_5d")
    vix_chg_10d = vix.pct_change(10).rename("vix_chg_10d")
    vix_percentile_252d = (
        vix.rolling(vix_percentile_window).rank(pct=True).rename("vix_percentile_252d")
    )

    rv_short = compute_realized_vol(spot, rv_short_window)
    rv_long = compute_realized_vol(spot, rv_long_window)
    rv_accel = (rv_short - rv_long).rename("rv_accel")

    # VIX is in raw points (e.g. 20.5); compute_realized_vol returns a
    # fractional annualized figure (e.g. 0.205) -- must divide VIX by 100
    # before comparing the two, per vix_source.py's own scale warning.
    vrp = (vix / 100 - rv_long).rename("vrp")
    vrp_chg_5d = vrp.diff(5).rename("vrp_chg_5d")

    momentum_21d = spot.pct_change(momentum_window).rename("momentum_21d")
    momentum_reversal = (momentum_21d - momentum_21d.shift(reversal_lookback)).rename("momentum_reversal")

    opt_spread = compute_daily_opt_spread(chain).reindex(trading_days, method="ffill")
    opt_spread_chg_5d = opt_spread.diff(5).rename("opt_spread_chg_5d")

    return pd.concat(
        [
            vix_chg_1d, vix_chg_5d, vix_chg_10d, vix_percentile_252d,
            vrp, vrp_chg_5d, rv_accel,
            momentum_21d, momentum_reversal,
            opt_spread.rename("opt_spread"), opt_spread_chg_5d,
        ],
        axis=1,
    )

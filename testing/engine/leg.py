from dataclasses import dataclass

import pandas as pd

from testing.engine.strike_selector import StrikeMatch


@dataclass(frozen=True)
class Leg:
    """One option leg of a (possibly multi-leg) position: a `StrikeMatch`
    tagged with which side of the market it's on (`right`: "P" or "C") and
    which side of the trade it's on (`side`: "short" or "long").

    This is what a `StrikeMatch` plus an implicit field name (e.g.
    `short_put`) becomes once positions are modeled as a flat list of legs
    instead of named fields -- `PositionSimulator`/`SizingEngine` operate on
    however many `Leg`s are present rather than assuming exactly 2 or 4.

    `exdate` is each leg's own expiry -- required, not derived from
    `StrikeMatch` (which doesn't carry it; the calling selector already
    knows which expiry slice a given match came from). Needed because not
    every structure shares one expiry across all legs (e.g. a calendar
    spread) -- see `PositionSpec.exdate`, which derives the position's
    overall exit boundary from these.

    `weight` is the leg's relative quantity within the position -- every
    strategy before `variance_swap` implicitly uses 1.0 (one leg = one
    contract-equivalent), which is why this defaults to 1.0 and changes
    nothing for them. `variance_swap`'s strike-strip replication is the
    first structure needing legs at different relative sizes (the standard
    1/K^2 log-contract weighting) rather than a handful of equal-size legs
    -- see `net_credit`/`max_loss_at_expiration`, both scaled by this field.
    """

    optionid: object
    exdate: pd.Timestamp
    strike_price: float
    delta: float
    best_bid: float
    best_offer: float
    right: str  # "P" or "C"
    side: str  # "short" or "long"
    impl_volatility: float | None = None
    weight: float = 1.0

    @classmethod
    def from_match(cls, match: StrikeMatch, right: str, side: str, exdate: pd.Timestamp, weight: float = 1.0) -> "Leg":
        return cls(
            optionid=match.optionid,
            exdate=exdate,
            strike_price=match.strike_price,
            delta=match.delta,
            best_bid=match.best_bid,
            best_offer=match.best_offer,
            right=right,
            side=side,
            impl_volatility=match.impl_volatility,
            weight=weight,
        )

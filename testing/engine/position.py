from dataclasses import dataclass

import pandas as pd

from testing.engine.leg import Leg


@dataclass(frozen=True)
class PositionSpec:
    """Generic replacement for the old VRP-specific `TradeSpec`: an entry
    date, an expiry, and a flat list of `Leg`s -- however many legs a given
    strategy's structure happens to need (a single put spread is 2, an iron
    condor is 4, a strategy with no symmetric structure at all could be any
    other count), rather than named `short_put`/`long_put`/`short_call`/
    `long_call` fields.

    The `*_legs` properties generalize what `TradeSpec.is_iron_condor`
    provided implicitly, so `PositionSimulator`/`SizingEngine` can be written
    against "however many legs of each kind are here" instead of 4 hardcoded
    names.
    """

    entry_date: pd.Timestamp
    legs: list[Leg]

    @property
    def exdate(self) -> pd.Timestamp:
        """The position's binding exit boundary -- the earliest expiry
        among its legs. For a vertical/condor (every leg shares one
        expiry) this trivially equals that shared expiry, unchanged from
        when `exdate` was a separate constructor field. For a calendar
        spread (legs at different expiries) this naturally resolves to the
        near leg's expiry -- the correct boundary, since the position's
        character fundamentally changes once that leg expires, with no
        special-casing needed anywhere else in `core/`.
        """
        return min(leg.exdate for leg in self.legs)

    @property
    def put_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.right == "P"]

    @property
    def call_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.right == "C"]

    @property
    def short_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.side == "short"]

    @property
    def long_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.side == "long"]

    @property
    def is_iron_condor(self) -> bool:
        """Retained for the VRP harvester's existing notion of "this position
        has both a put spread and a call spread" -- generic in terms of legs
        present, not a hardcoded structure type.
        """
        return len(self.put_legs) > 0 and len(self.call_legs) > 0

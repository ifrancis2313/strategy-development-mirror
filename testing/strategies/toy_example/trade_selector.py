import pandas as pd

from testing.engine.leg import Leg
from testing.engine.position import PositionSpec
from testing.engine.strike_selector import StrikeSelector, select_target_expiry
from testing.strategies.toy_example.config import ToyConfig


class ToyTradeSelector:
    """Minimal `select_trade` implementation: enters a two-leg put credit
    spread whenever no position is open and both target deltas resolve to
    real, correctly-ordered strikes. Demonstrates the selector contract
    described in testing/CONTEXT.md -- deliberately has no signal, no
    gating, no sizing logic beyond what `ExecutionConfig` already provides.
    """

    def __init__(self, config: ToyConfig, strike_selector: StrikeSelector):
        self.config = config
        self.strike_selector = strike_selector

    def select_trade(
        self,
        chain_snapshot: pd.DataFrame,
        entry_date,
        has_open_position: bool,
        **kwargs,
    ) -> PositionSpec | None:
        if has_open_position or chain_snapshot.empty:
            return None

        entry_date = pd.Timestamp(entry_date)
        target = select_target_expiry(chain_snapshot, entry_date, self.config.min_dte, self.config.max_dte)
        if target is None:
            return None
        chosen_exdate, expiry_chain = target

        short_put = self.strike_selector.select(expiry_chain, self.config.short_put_delta, cp_flag="P")
        long_put = self.strike_selector.select(expiry_chain, self.config.long_put_delta, cp_flag="P")
        if short_put is None or long_put is None:
            return None
        if long_put.strike_price >= short_put.strike_price:
            return None

        return PositionSpec(
            entry_date=entry_date,
            legs=[
                Leg.from_match(short_put, right="P", side="short", exdate=chosen_exdate),
                Leg.from_match(long_put, right="P", side="long", exdate=chosen_exdate),
            ],
        )

"""ChainSnapshot -- the Ingestion wing's output contract. Ported verbatim
from Options and Futures Pipeline's `pipeline/block1_ingestion/schema.py`
(same field names/types/constraints) so `chain_snapshots` in Supabase keeps
an identical shape across the cutover -- OFP's `db_sink.py` write logic and
`pipeline-dashboard`'s `ofp_adapter.py` read logic both depend on this exact
shape, not just the shape of `supabase_client.py`'s SQL below."""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ContractRow(BaseModel):
    expiry: date
    strike: float = Field(gt=0)
    right: Literal["C", "P"]
    bid: float = Field(ge=0)
    ask: float = Field(ge=0)
    last: float | None = None
    volume: int = Field(ge=0, default=0)
    open_interest: int = Field(ge=0, default=0)
    implied_vol: float | None = Field(default=None, ge=0)
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    source: str

    @field_validator("ask")
    @classmethod
    def ask_not_below_bid(cls, ask: float, info) -> float:
        # `bid is not None` guards a real case (bid missing from info.data
        # if bid's own Field(ge=0) validation already failed, not because
        # bid is Optional -- it isn't). The `and ask` that used to sit next
        # to it was different: ask is also required/non-nullable, so that
        # was accidentally testing ask's truthiness (0.0 is falsy), which
        # silently skipped this check whenever ask == 0.0 -- exactly the
        # degenerate crossed-market case (ask=0.0, bid>0) this validator
        # exists to catch. Ported verbatim from Options and Futures
        # Pipeline's schema.py, where the same gap exists; fixed here only,
        # found during the front-to-back review (2026-07-12).
        bid = info.data.get("bid")
        if bid is not None and ask < bid:
            raise ValueError(f"ask ({ask}) below bid ({bid})")
        return ask

    @property
    def mid(self) -> float:
        if self.bid and self.ask:
            return (self.bid + self.ask) / 2
        return self.last or 0.0


class ChainSnapshot(BaseModel):
    symbol: str
    spot_price: float = Field(gt=0)
    as_of: datetime
    risk_free_rate: float = Field(ge=0, le=1)
    source: str
    contracts: list[ContractRow]
    # 21-day rolling annualized realized vol on the underlying's daily
    # closes -- None if unavailable. Not part of the `chain_snapshots`
    # Postgres table (OFP never persisted this column there either); kept
    # here only so `ChainSnapshot` stays a drop-in match for OFP's schema
    # if a future consumer needs it.
    realized_vol_21d: float | None = None

    @property
    def n_contracts(self) -> int:
        return len(self.contracts)

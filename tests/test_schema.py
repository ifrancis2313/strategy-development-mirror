from datetime import date

import pytest
from pydantic import ValidationError

from ingestion.schema import ContractRow


def _row(**overrides):
    defaults = dict(expiry=date(2026, 1, 16), strike=500.0, right="C", bid=1.0, ask=1.2, source="test")
    return ContractRow(**{**defaults, **overrides})


def test_accepts_a_normal_quote():
    row = _row(bid=1.0, ask=1.2)
    assert row.ask == 1.2


def test_rejects_ask_below_bid():
    with pytest.raises(ValidationError, match="ask .* below bid"):
        _row(bid=5.0, ask=1.0)


def test_rejects_zero_ask_below_a_positive_bid():
    # A crossed/degenerate quote (ask=0.0, bid>0) must still be caught --
    # this used to slip through a truthiness bug (`and ask`, 0.0 is falsy).
    with pytest.raises(ValidationError, match="ask .* below bid"):
        _row(bid=5.0, ask=0.0)


def test_allows_both_bid_and_ask_zero():
    # A genuinely no-quote contract (both sides zero) is not a crossed
    # market and should pass.
    row = _row(bid=0.0, ask=0.0)
    assert row.ask == 0.0

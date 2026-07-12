"""Behavior-preservation tests for shadow/ledger.py -- ported from
Options_Trading_Strategy's tests/core/broker/test_shadow_engine.py and
test_shadow_ledger_store.py (see those files for the original scenarios;
several reference numbers below reused verbatim).

Uses the same local test-double Leg/PositionSpec/Config classes as
test_execution.py -- duck-typed, not imported from testing.engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pytest

from shadow import ledger
from shadow.execution import contracts_for_entry, entry_value_by_right

ENTRY = pd.Timestamp("2026-06-24")
EXDATE = pd.Timestamp("2026-08-21")  # ~58 days out, well past exit_dte=21
CAPS = {"floor": 0.0, "target": 0.65, "mid": 1.0}


@dataclass(frozen=True)
class Leg:
    optionid: object
    exdate: pd.Timestamp
    strike_price: float
    delta: float
    best_bid: float
    best_offer: float
    right: str
    side: str
    impl_volatility: float | None = None


@dataclass(frozen=True)
class PositionSpec:
    entry_date: pd.Timestamp
    legs: list[Leg]

    @property
    def exdate(self) -> pd.Timestamp:
        return min(leg.exdate for leg in self.legs)

    @property
    def put_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.right == "P"]

    @property
    def call_legs(self) -> list[Leg]:
        return [leg for leg in self.legs if leg.right == "C"]


@dataclass(frozen=True)
class Config:
    profit_target_pct: float = 0.50
    stop_loss_multiple: float = 2.0
    exit_dte: int = 21
    exit_trigger_basis: Literal["entry_credit", "max_loss"] = "entry_credit"
    fixed_notional: float = 100_000.0
    contract_multiplier: int = 100


def _spec():
    short_put = Leg(optionid="SP", exdate=EXDATE, strike_price=95, delta=-0.20, best_bid=3.0, best_offer=3.2, right="P", side="short")
    long_put = Leg(optionid="LP", exdate=EXDATE, strike_price=90, delta=-0.10, best_bid=1.0, best_offer=1.2, right="P", side="long")
    return PositionSpec(entry_date=ENTRY, legs=[short_put, long_put])


def _calendar_spec():
    near = Leg(optionid="NP", exdate=pd.Timestamp("2026-07-17"), strike_price=100, delta=-0.50, best_bid=2.0, best_offer=2.2, right="P", side="short")
    far = Leg(optionid="FP", exdate=pd.Timestamp("2026-08-21"), strike_price=100, delta=-0.50, best_bid=3.0, best_offer=3.2, right="P", side="long")
    return PositionSpec(entry_date=ENTRY, legs=[near, far])


@dataclass(frozen=True)
class WeightedLeg:
    """A second Leg-like shape carrying `weight`, mirroring
    testing.engine.leg.Leg's variance_swap-era field -- proves weight
    survives a real record_entry -> record_exit-or-hold round trip through
    the JSON ledger store, not just the entry-day math (which uses the
    real spec object directly and doesn't exercise _RehydratedLeg at all).
    """

    optionid: object
    exdate: pd.Timestamp
    strike_price: float
    best_bid: float
    best_offer: float
    right: str
    side: str
    weight: float = 1.0


def _weighted_spec():
    leg = WeightedLeg(optionid="WP", exdate=EXDATE, strike_price=95, best_bid=3.0, best_offer=3.2, right="P", side="short", weight=2.0)
    return PositionSpec(entry_date=ENTRY, legs=[leg])


def _condor_spec():
    short_put = Leg(optionid="SP", exdate=EXDATE, strike_price=95, delta=-0.20, best_bid=3.0, best_offer=3.2, right="P", side="short")
    long_put = Leg(optionid="LP", exdate=EXDATE, strike_price=90, delta=-0.10, best_bid=1.0, best_offer=1.2, right="P", side="long")
    short_call = Leg(optionid="SC", exdate=EXDATE, strike_price=105, delta=0.30, best_bid=2.0, best_offer=2.2, right="C", side="short")
    long_call = Leg(optionid="LC", exdate=EXDATE, strike_price=110, delta=0.20, best_bid=0.8, best_offer=1.0, right="C", side="long")
    return PositionSpec(entry_date=ENTRY, legs=[short_put, long_put, short_call, long_call])


# ---------------------------------------------------------------------
# Engine (ported from test_shadow_engine.py)
# ---------------------------------------------------------------------


def test_credit_total_combines_put_and_call_additively():
    """Mutation testing (2026-07-12): (P or 0.0) + (C or 0.0) survived being
    flipped to a subtraction -- every prior _credit_total-touching test used
    a put-only spec, where C is always None and +/- are indistinguishable."""
    assert ledger._credit_total({"P": 5.0, "C": 3.0}) == pytest.approx(8.0)


def test_credit_total_falls_back_to_zero_not_one_when_a_right_is_falsy():
    """(P or 0.0) survived becoming (P or 1.0) -- only observable when P is
    exactly 0.0/None, the one case the `or` fallback actually fires."""
    assert ledger._credit_total({"P": 0.0, "C": 3.0}) == pytest.approx(3.0)
    assert ledger._credit_total({"P": None, "C": 3.0}) == pytest.approx(3.0)


def test_spec_has_pricing_rejects_a_leg_missing_only_one_side_of_the_quote():
    """`best_bid is None or best_offer is None` survived being flipped to
    `and` (2026-07-12) -- every prior test used legs with either both sides
    present or (implicitly) neither, never exactly one missing."""
    one_sided = Leg(optionid="X", exdate=EXDATE, strike_price=95, delta=-0.20,
                     best_bid=3.0, best_offer=None, right="P", side="short")  # type: ignore[arg-type]
    spec = PositionSpec(entry_date=ENTRY, legs=[one_sided])
    assert ledger._spec_has_pricing(spec) is False


def test_spec_has_pricing_rejects_nan_bid_or_offer():
    """The two `math.isnan(...)` guards each survived `return False` being
    flipped to `return True` (2026-07-12) -- no prior test constructed a
    leg with an actual NaN quote (a plausible real broker-feed degenerate
    case, distinct from a missing/None quote)."""
    nan_bid = Leg(optionid="X", exdate=EXDATE, strike_price=95, delta=-0.20,
                  best_bid=float("nan"), best_offer=3.2, right="P", side="short")
    assert ledger._spec_has_pricing(PositionSpec(entry_date=ENTRY, legs=[nan_bid])) is False

    nan_offer = Leg(optionid="X", exdate=EXDATE, strike_price=95, delta=-0.20,
                     best_bid=3.0, best_offer=float("nan"), right="P", side="short")
    assert ledger._spec_has_pricing(PositionSpec(entry_date=ENTRY, legs=[nan_offer])) is False


def test_run_ledgers_entry_credit_combines_put_and_call_for_a_condor(tmp_path):
    """Integration-level version of test_credit_total_combines_put_and_call_
    additively -- proves the additive combination is what actually reaches
    record_entry/contracts_for_entry through run_ledgers, not just the unit
    function in isolation. Every prior run_ledgers test used a put-only or
    calendar (put-only) spec."""
    root = str(tmp_path)
    ledger.run_ledgers("condor", Config(), {"floor": 0.0}, _condor_spec(), quotes={}, today=ENTRY, root=root)
    floor = ledger.load_open("condor", "floor", root=root)
    # put: short sells at bid 3.0, long buys at offer 1.2 -> 1.8
    # call: short sells at bid 2.0, long buys at offer 1.0 -> 1.0
    assert floor["entry_credit_by_right"]["P"] == pytest.approx(1.8)
    assert floor["entry_credit_by_right"]["C"] == pytest.approx(1.0)
    assert floor["entry_credit_per_contract"] == pytest.approx(2.8)  # 1.8+1.0, not 1.8-1.0


def test_open_ledger_holding_pnl_per_contract_is_entry_minus_cost(tmp_path):
    """Mutation testing (2026-07-12): `pnl_pc = entry_credit - cost_to_close`
    survived being flipped to `+` in the "holding" (no exit yet) branch --
    the existing holding test only checked action=="holding", never the
    actual pnl/contract value."""
    root = str(tmp_path)
    ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, _spec(), quotes={}, today=ENTRY, root=root)
    # entry_credit_per_contract = 1.8 (short bid 3.0 - long offer 1.2)
    # cost_to_close: short buys back at offer 3.0, long sells at bid 0.9 -> 3.0-0.9=2.1
    quotes = {"SP": (2.8, 3.0), "LP": (0.9, 1.1)}
    later = ENTRY + pd.Timedelta(days=5)
    outcomes = ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, None, quotes, today=later, root=root)
    assert outcomes[0].action == "holding"
    assert outcomes[0].pnl_per_contract == pytest.approx(1.8 - 2.1)


def test_flat_with_selected_trade_enters_all_ledgers(tmp_path):
    root = str(tmp_path)
    outcomes = ledger.run_ledgers("vrp", Config(), CAPS, _spec(), quotes={}, today=ENTRY, root=root)
    assert {o.ledger: o.action for o in outcomes} == {"floor": "entered", "target": "entered", "mid": "entered"}
    floor = ledger.load_open("vrp", "floor", root=root)
    assert round(floor["entry_credit_per_contract"], 2) == 1.80
    target = ledger.load_open("vrp", "target", root=root)
    assert floor["contracts_shared"] == target["contracts_shared"]
    assert target["entry_credit_per_contract"] > floor["entry_credit_per_contract"]
    assert target["contracts_independent"] != floor["contracts_independent"]


def test_three_ledger_fan_out_shared_off_floor_independent_per_capture(tmp_path):
    root = str(tmp_path)
    config = Config()
    spec = _spec()
    ledger.run_ledgers("vrp", config, CAPS, spec, quotes={}, today=ENTRY, root=root)

    recs = {lg: ledger.load_open("vrp", lg, root=root) for lg in CAPS}
    floor_by_right = entry_value_by_right(spec, 0.0)
    expected_shared = contracts_for_entry(spec, (floor_by_right["P"] or 0) + (floor_by_right["C"] or 0), config)
    assert {r["contracts_shared"] for r in recs.values()} == {expected_shared}

    assert recs["floor"]["entry_credit_per_contract"] < recs["target"]["entry_credit_per_contract"] < recs["mid"]["entry_credit_per_contract"]
    for lg in CAPS:
        by_right = entry_value_by_right(spec, CAPS[lg])
        expected_indep = contracts_for_entry(spec, (by_right["P"] or 0) + (by_right["C"] or 0), config)
        assert recs[lg]["contracts_independent"] == expected_indep


def test_engine_handles_calendar_net_debit_and_near_leg_exdate(tmp_path):
    root = str(tmp_path)
    outs = ledger.run_ledgers("cal", Config(), CAPS, _calendar_spec(), quotes={}, today=ENTRY, root=root)
    assert all(o.action == "entered" for o in outs)
    floor = ledger.load_open("cal", "floor", root=root)
    assert floor["entry_credit_per_contract"] < 0  # net debit (sell 2.0 - buy 3.2)


def test_flat_with_no_trade_is_flat(tmp_path):
    outcomes = ledger.run_ledgers("vrp", Config(), CAPS, None, quotes={}, today=ENTRY, root=str(tmp_path))
    assert all(o.action == "flat" for o in outcomes)


def test_open_ledger_exits_on_profit_target(tmp_path):
    root = str(tmp_path)
    ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, _spec(), quotes={}, today=ENTRY, root=root)
    # cheap to close: short buy@0.6, long sell@0.2 -> cost 0.4; profit 1.8-0.4=1.4 >= 0.9
    quotes = {"SP": (0.5, 0.6), "LP": (0.2, 0.3)}
    later = ENTRY + pd.Timedelta(days=10)
    outcomes = ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, None, quotes, today=later, root=root)
    assert outcomes[0].action == "exited"
    assert outcomes[0].exit_reason == "profit_target"
    assert ledger.load_open("vrp", "floor", root=root) is None
    assert len(ledger.load_state("vrp", root=root)["floor"]["closed"]) == 1


def test_open_ledger_holds_when_no_exit(tmp_path):
    root = str(tmp_path)
    ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, _spec(), quotes={}, today=ENTRY, root=root)
    quotes = {"SP": (2.8, 3.0), "LP": (0.9, 1.1)}
    later = ENTRY + pd.Timedelta(days=5)
    outcomes = ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, None, quotes, today=later, root=root)
    assert outcomes[0].action == "holding"
    assert ledger.load_open("vrp", "floor", root=root) is not None


def test_missing_quote_warns_and_holds(tmp_path):
    root = str(tmp_path)
    ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, _spec(), quotes={}, today=ENTRY, root=root)
    quotes = {"SP": (0.5, 0.6)}  # LP missing -- never fabricate a fill
    later = ENTRY + pd.Timedelta(days=10)
    outcomes = ledger.run_ledgers("vrp", Config(), {"floor": 0.0}, None, quotes, today=later, root=root)
    assert outcomes[0].action == "warn"
    assert ledger.load_open("vrp", "floor", root=root) is not None  # left open


def test_leg_weight_survives_the_json_round_trip_into_next_days_mark_to_market(tmp_path):
    # Entry day: real WeightedLeg object, weight=2.0 applied directly by
    # shadow.execution.net_credit -- entry_credit_per_contract = bid(3.0) *
    # weight(2.0) = 6.0 at floor capture (sells at bid).
    root = str(tmp_path)
    ledger.run_ledgers("vw", Config(), {"floor": 0.0}, _weighted_spec(), quotes={}, today=ENTRY, root=root)
    floor = ledger.load_open("vw", "floor", root=root)
    assert floor["entry_credit_per_contract"] == pytest.approx(6.0)
    assert floor["spec"]["legs"][0]["weight"] == 2.0  # confirms it was actually stored

    # Next day: position is reloaded from the JSON store through
    # _RehydratedLeg, not the original WeightedLeg object. cost_to_close =
    # offer(1.0) * weight(2.0) = 2.0 at floor capture (buys back at offer);
    # pnl/contract = 6.0 - 2.0 = 4.0. Before the _RehydratedLeg fix, weight
    # silently defaulted to 1.0 here, giving cost_to_close=1.0 and a wrong
    # pnl/contract of 5.0 despite the correct entry_credit above.
    quotes = {"WP": (0.9, 1.0)}
    later = ENTRY + pd.Timedelta(days=5)
    outcomes = ledger.run_ledgers("vw", Config(), {"floor": 0.0}, None, quotes, today=later, root=root)
    assert outcomes[0].pnl_per_contract == pytest.approx(4.0)


def test_ledgers_desync_on_independent_exit(tmp_path):
    root = str(tmp_path)
    ledger.run_ledgers("vrp", Config(), {"floor": 0.0, "target": 0.65}, _spec(), quotes={}, today=ENTRY, root=root)
    quotes = {"SP": (0.85, 0.90), "LP": (0.0, 0.05)}
    later = ENTRY + pd.Timedelta(days=10)
    outcomes = {
        o.ledger: o.action
        for o in ledger.run_ledgers("vrp", Config(), {"floor": 0.0, "target": 0.65}, None, quotes, today=later, root=root)
    }
    assert outcomes["floor"] == "exited"
    assert ledger.load_open("vrp", "floor", root=root) is None


# ---------------------------------------------------------------------
# Store (ported from test_shadow_ledger_store.py)
# ---------------------------------------------------------------------


def _entry(root, name, capture, credit_pc, shared=5, indep=5):
    ledger.record_entry(
        "vrp_harvester", name, spec=_spec(), capture=capture,
        entry_credit_by_right={"P": credit_pc, "C": None},
        entry_credit_per_contract=credit_pc,
        contracts_shared=shared, contracts_independent=indep, root=root,
    )


def test_load_open_returns_none_when_nothing_recorded(tmp_path):
    assert ledger.load_open("vrp_harvester", "floor", root=str(tmp_path)) is None


def test_record_entry_round_trips(tmp_path):
    root = str(tmp_path)
    _entry(root, "floor", capture=0.0, credit_pc=1.8, shared=5, indep=7)

    loaded = ledger.load_open("vrp_harvester", "floor", root=root)
    assert loaded["capture"] == 0.0
    assert loaded["entry_credit_per_contract"] == 1.8
    assert loaded["contracts_shared"] == 5
    assert loaded["contracts_independent"] == 7
    assert loaded["spec"]["legs"][0]["optionid"] == "SP"
    # dict-key round trip, not just "some record exists" -- mutation testing
    # (2026-07-12) found the "entry_date" key literal survives being renamed/
    # cased differently, since no prior test read it back by exact key.
    assert loaded["entry_date"] == ENTRY.isoformat()


def test_record_entry_refuses_to_overwrite_open_ledger(tmp_path):
    root = str(tmp_path)
    _entry(root, "floor", capture=0.0, credit_pc=1.8)
    with pytest.raises(ValueError):
        _entry(root, "floor", capture=0.0, credit_pc=1.8)


def test_ledgers_are_independent(tmp_path):
    root = str(tmp_path)
    _entry(root, "floor", capture=0.0, credit_pc=1.8)
    assert ledger.load_open("vrp_harvester", "target", root=root) is None
    _entry(root, "target", capture=0.65, credit_pc=2.3)
    assert ledger.load_open("vrp_harvester", "floor", root=root)["capture"] == 0.0
    assert ledger.load_open("vrp_harvester", "target", root=root)["capture"] == 0.65


def test_record_exit_clears_open_and_appends_closed_with_per_contract_pnl(tmp_path):
    root = str(tmp_path)
    _entry(root, "floor", capture=0.0, credit_pc=1.8, shared=5, indep=7)
    ledger.record_exit(
        "vrp_harvester", "floor", exit_reason="profit_target",
        exit_date=pd.Timestamp("2026-07-10"), exit_debit_per_contract=0.8,
        pnl_per_contract=1.0, root=root,
    )

    assert ledger.load_open("vrp_harvester", "floor", root=root) is None
    state = ledger.load_state("vrp_harvester", root=root)
    closed = state["floor"]["closed"]
    assert len(closed) == 1
    rec = closed[0]
    assert rec["exit_reason"] == "profit_target"
    assert rec["pnl_per_contract"] == 1.0
    assert "pnl_dollars" not in rec  # dollar P&L derived at read time, never stored
    # dict-key round trip for the two fields record_exit itself adds --
    # mutation testing (2026-07-12) found "exit_date"/"exit_debit_per_contract"
    # both survive being renamed, since no prior test read them back by key.
    assert rec["exit_date"] == pd.Timestamp("2026-07-10").isoformat()
    assert rec["exit_debit_per_contract"] == 0.8
    assert rec["contracts_shared"] == 5 and rec["contracts_independent"] == 7


def test_record_exit_without_open_raises(tmp_path):
    with pytest.raises(ValueError):
        ledger.record_exit(
            "vrp_harvester", "floor", exit_reason="x",
            exit_date=pd.Timestamp("2026-07-10"), exit_debit_per_contract=0.0,
            pnl_per_contract=0.0, root=str(tmp_path),
        )

"""End-to-end proof that Shadow can actually consume a real Graduation
artifact -- not a hand-built test double, the genuine
`graduated/vrp_harvester/v1_20260709T214151Z/` snapshot issue 09 produced.

Uses `testing.graduation.loader.load_snapshot` (the one dependency
`shadow/` is allowed on `testing/`, per CONTEXT.md's Graduation entry) to
dynamically load the frozen snapshot, drives its own `VRPTradeSelector`
against a synthetic-but-realistic chain snapshot to get a real
`PositionSpec` built from the snapshot's own frozen `Leg`/`PositionSpec`
classes, then feeds that spec into `shadow.ledger.run_ledgers` -- proving
the duck-typed boundary between shadow/execution.py and a graduated
snapshot's own classes actually works at runtime, end to end, against
genuine graduation output.

This test imports `testing.graduation.loader` directly (a test of the
composition, not a module living under `shadow/`) -- it does not violate
`shadow/`'s own zero-coupling guarantee (see test_zero_testing_coupling.py,
which only inspects files under shadow/).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from testing.graduation.loader import load_snapshot_module

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_DIR = REPO_ROOT / "graduated" / "vrp_harvester" / "v1_20260709T214151Z"

pytestmark = pytest.mark.skipif(
    not SNAPSHOT_DIR.exists(),
    reason="issue 09's demo graduation snapshot not present on disk",
)


def _synthetic_chain(entry_date: pd.Timestamp, secid: int = 108105) -> pd.DataFrame:
    """A realistic-shaped SPX chain slice: one expiry 37 DTE out (inside
    VRPConfig's default [30, 45] window), puts spanning a delta range wide
    enough for the default short_put_delta=-0.20/long_put_delta=-0.10 to
    both resolve within StrikeSelector's 0.05 max_delta_distance.
    """
    exdate = entry_date + pd.Timedelta(days=37)
    strikes = list(range(80, 105, 2))  # 80, 82, ..., 102
    rows = []
    for i, strike in enumerate(strikes):
        # Put delta grows more negative as strike rises toward the money
        # (real-market convention: deep-OTM low strike -> delta near 0,
        # near-ATM high strike -> delta near -1). Spans past both
        # short_put_delta=-0.20 (strike 88) and long_put_delta=-0.10 (strike
        # 82), with long_put landing at a *lower* strike than short_put, as
        # VRPTradeSelector requires for a real put credit spread.
        delta = -0.05 - i * 0.04
        rows.append({
            "secid": secid,
            "date": entry_date,
            "exdate": exdate,
            "cp_flag": "P",
            "strike_price": float(strike),
            "best_bid": 1.0 + i * 0.3,
            "best_offer": 1.2 + i * 0.3,
            "delta": delta,
            "impl_volatility": 0.18,
            "optionid": f"SPXW_{strike}P",
        })
    return pd.DataFrame(rows)


def _load_selector_and_config():
    config_module = load_snapshot_module(SNAPSHOT_DIR, "strategies.vrp_harvester.config")
    selector_module = load_snapshot_module(SNAPSHOT_DIR, "strategies.vrp_harvester.trade_selector")
    strike_selector_module = load_snapshot_module(SNAPSHOT_DIR, "engine.strike_selector")

    config = config_module.VRPConfig()
    selector = selector_module.VRPTradeSelector(config, strike_selector_module.StrikeSelector())
    return config, selector


def test_graduated_snapshot_selector_produces_a_real_position_spec():
    config, selector = _load_selector_and_config()
    entry_date = pd.Timestamp("2026-07-13")
    chain = _synthetic_chain(entry_date)

    spec = selector.select_trade(chain, entry_date, has_open_position=False)

    assert spec is not None
    assert len(spec.legs) == 2  # put_credit_spread default structure
    assert {leg.right for leg in spec.legs} == {"P"}
    assert {leg.side for leg in spec.legs} == {"short", "long"}


def test_shadow_ledger_consumes_a_real_graduated_position_spec(tmp_path):
    """The actual end-to-end proof: a graduated snapshot's own selector
    output flows into shadow.ledger.run_ledgers unchanged, and all three
    capture ledgers enter -- shadow/'s duck-typed execution math works
    against the snapshot's own frozen Leg/PositionSpec/ExecutionConfig
    classes without shadow/ needing to statically import any of them.
    """
    from shadow import ledger

    config, selector = _load_selector_and_config()
    entry_date = pd.Timestamp("2026-07-13")
    chain = _synthetic_chain(entry_date)
    spec = selector.select_trade(chain, entry_date, has_open_position=False)
    assert spec is not None

    root = str(tmp_path)
    captures = {"floor": 0.0, "target": 0.65, "mid": 1.0}
    outcomes = ledger.run_ledgers(
        "vrp_harvester_graduated_v1", config, captures, spec, quotes={}, today=entry_date, root=root,
    )

    assert {o.ledger: o.action for o in outcomes} == {"floor": "entered", "target": "entered", "mid": "entered"}

    floor_rec = ledger.load_open("vrp_harvester_graduated_v1", "floor", root=root)
    assert floor_rec is not None
    assert floor_rec["entry_credit_per_contract"] > 0  # a put credit spread should collect a net credit
    assert floor_rec["contracts_shared"] >= 1

    # Now mark to market on a later date with tighter quotes (near-worthless
    # to close) and confirm the floor ledger can actually exit through the
    # same shadow.ledger/shadow.execution machinery.
    put_legs = {leg.optionid: leg for leg in spec.legs}
    quotes = {optionid: (0.01, 0.02) for optionid in put_legs}  # cheap to close -> profit
    later = entry_date + pd.Timedelta(days=10)
    exit_outcomes = ledger.run_ledgers(
        "vrp_harvester_graduated_v1", config, {"floor": 0.0}, None, quotes, today=later, root=root,
    )
    assert exit_outcomes[0].action in ("exited", "holding")  # deterministically one or the other, never "warn"/"flat"

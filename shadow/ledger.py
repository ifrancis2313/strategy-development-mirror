"""The parallel capture-fraction shadow ledger (ADR 0008, ported from
`Options_Trading_Strategy`'s `core/broker/shadow_ledger_store.py` +
`core/broker/shadow_engine.py`).

A strategy runs three **parallel shadow ledgers**, one per spread-capture
assumption: floor (0.0, always-achievable), target (0.65, the
empirically-validated break-even capture), mid (1.0, optimistic ceiling).
Each ledger is **self-governed**: it enters at its own credit and exits
when *its own* cost-to-close trips a threshold. Entries are identical
across ledgers when all are flat (the strategy's selector doesn't depend on
capture); divergent exits desync them.

**Per-contract economics are the stored source of truth.** Each trade
stores both a shared/pinned contract count (sized off the floor credit,
identical across every ledger) and the ledger's own independent count
(sized off its own capture-dependent credit) -- dollar P&L is derived at
read time (`pnl_per_contract * contract_multiplier * <chosen count>`),
never stored, so the clean per-capture comparison is never confounded by a
sizing artifact.

Pure of any broker dependency: quotes and an already-selected `PositionSpec`
(built by a graduated snapshot's own selector against live broker data) come
in; ledger outcomes and JSON-file store writes go out. Uses `shadow.execution`
for the fill/mark-to-market/exit math -- zero import from `testing.engine` or
`testing.strategies` anywhere in this module (see
`tests/shadow/test_zero_testing_coupling.py`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from shadow.execution import check_exit, contracts_for_entry, entry_value_by_right, mark_to_market

SHADOW_ROOT = "data/generated/shadow"
FLOOR_CAPTURE = 0.0
DEFAULT_CAPTURES = {"floor": 0.0, "target": 0.65, "mid": 1.0}


# ---------------------------------------------------------------------
# Persistence (ported from core/broker/shadow_ledger_store.py)
# ---------------------------------------------------------------------


def _path_for(strategy: str, root: str = SHADOW_ROOT) -> Path:
    return Path(root) / f"{strategy}.json"


def _load_file(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_file(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2)


def _ledger(state: dict, ledger: str) -> dict:
    return state.setdefault(ledger, {"open": None, "closed": []})


def _leg_to_dict(leg: Any) -> dict:
    d = dict(leg.__dict__) if hasattr(leg, "__dict__") else {}
    if not d:
        # frozen dataclasses without __dict__ (slots) -- fall back to dataclasses.asdict
        import dataclasses

        d = dataclasses.asdict(leg)
    d["exdate"] = pd.Timestamp(leg.exdate).isoformat()
    return d


def _spec_to_dict(spec: Any) -> dict:
    return {
        "entry_date": pd.Timestamp(spec.entry_date).isoformat(),
        "legs": [_leg_to_dict(leg) for leg in spec.legs],
    }


def load_open(strategy: str, ledger: str, root: str = SHADOW_ROOT) -> dict | None:
    """Return the open record for one ledger. `spec` is left as a plain
    dict (unlike the source repo's `load_open`, which rehydrates a
    `core.position.PositionSpec`) -- Shadow has no static `PositionSpec`
    class of its own to rehydrate into (see module docstring); callers that
    need a live object should reconstruct one from the graduated snapshot's
    own frozen `PositionSpec`/`Leg` classes.
    """
    state = _load_file(_path_for(strategy, root))
    rec = state.get(ledger, {}).get("open")
    if rec is None:
        return None
    return dict(rec)


def record_entry(
    strategy: str,
    ledger: str,
    *,
    spec: Any,
    capture: float,
    entry_credit_by_right: dict,
    entry_credit_per_contract: float,
    contracts_shared: int,
    contracts_independent: int,
    root: str = SHADOW_ROOT,
) -> None:
    """Record a simulated entry for one ledger. Enforces one-open-per-ledger."""
    path = _path_for(strategy, root)
    state = _load_file(path)
    led = _ledger(state, ledger)
    if led["open"] is not None:
        raise ValueError(f"Refusing to record_entry over an open {strategy}/{ledger} ledger -- close it first")
    led["open"] = {
        "spec": _spec_to_dict(spec),
        "capture": capture,
        "entry_credit_by_right": entry_credit_by_right,
        "entry_credit_per_contract": entry_credit_per_contract,
        "contracts_shared": contracts_shared,
        "contracts_independent": contracts_independent,
        "entry_date": pd.Timestamp(spec.entry_date).isoformat(),
    }
    _save_file(path, state)


def record_exit(
    strategy: str,
    ledger: str,
    *,
    exit_reason: str,
    exit_date,
    exit_debit_per_contract: float,
    pnl_per_contract: float,
    root: str = SHADOW_ROOT,
) -> None:
    """Close the open record for one ledger. Dollar P&L is NOT stored --
    derive it from `pnl_per_contract * contract_multiplier * <chosen count>`.
    """
    path = _path_for(strategy, root)
    state = _load_file(path)
    led = _ledger(state, ledger)
    if led["open"] is None:
        raise ValueError(f"No open {strategy}/{ledger} position to record_exit for")
    led["closed"].append({
        **led["open"],
        "exit_reason": exit_reason,
        "exit_date": pd.Timestamp(exit_date).isoformat(),
        "exit_debit_per_contract": exit_debit_per_contract,
        "pnl_per_contract": pnl_per_contract,
    })
    led["open"] = None
    _save_file(path, state)


def load_state(strategy: str, root: str = SHADOW_ROOT) -> dict:
    """Raw all-ledgers state for a strategy (for reporting/comparison)."""
    return _load_file(_path_for(strategy, root))


# ---------------------------------------------------------------------
# Engine (ported from core/broker/shadow_engine.py)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerOutcome:
    ledger: str
    capture: float
    action: str  # "entered" | "exited" | "holding" | "flat" | "warn"
    detail: str
    pnl_per_contract: float | None = None
    exit_reason: str | None = None


def _credit_total(by_right: dict) -> float:
    return (by_right["P"] or 0.0) + (by_right["C"] or 0.0)


def _missing_quote_legs(spec: Any, quotes: dict) -> list:
    return [leg.optionid for leg in spec.legs if leg.optionid not in quotes]


def _spec_has_pricing(spec: Any) -> bool:
    for leg in spec.legs:
        if leg.best_bid is None or leg.best_offer is None:
            return False
        if isinstance(leg.best_bid, float) and math.isnan(leg.best_bid):
            return False
        if isinstance(leg.best_offer, float) and math.isnan(leg.best_offer):
            return False
    return True


def run_ledgers(
    strategy: str,
    config: Any,
    captures: dict,
    selected_spec: Any | None,
    quotes: dict,
    today,
    root: str = SHADOW_ROOT,
) -> list:
    """Process every ledger once for `today`, recording entries/exits to the
    store and returning a `LedgerOutcome` per ledger. `config` and
    `selected_spec` come from a graduated snapshot's own `ExecutionConfig`/
    `PositionSpec` classes (duck-typed here, never statically imported --
    see module docstring).
    """
    today = pd.Timestamp(today)

    shared_contracts = None
    if selected_spec is not None and _spec_has_pricing(selected_spec):
        floor_by_right = entry_value_by_right(selected_spec, FLOOR_CAPTURE)
        shared_contracts = contracts_for_entry(selected_spec, _credit_total(floor_by_right), config)

    outcomes = []
    for ledger, capture in captures.items():
        outcomes.append(
            _run_one(strategy, config, ledger, capture, selected_spec, shared_contracts, quotes, today, root)
        )
    return outcomes


def _run_one(strategy, config, ledger, capture, selected_spec, shared_contracts, quotes, today, root):
    open_rec = load_open(strategy, ledger, root=root)

    if open_rec is not None:
        return _handle_open(strategy, config, ledger, capture, open_rec, quotes, today, root)

    if selected_spec is None:
        return LedgerOutcome(ledger, capture, "flat", "no trade selected")
    if not _spec_has_pricing(selected_spec):
        return LedgerOutcome(ledger, capture, "warn", "selected trade missing leg pricing -- no entry")

    by_right = entry_value_by_right(selected_spec, capture)
    credit_pc = _credit_total(by_right)
    independent = contracts_for_entry(selected_spec, credit_pc, config)
    record_entry(
        strategy, ledger, spec=selected_spec, capture=capture,
        entry_credit_by_right=by_right, entry_credit_per_contract=credit_pc,
        contracts_shared=shared_contracts, contracts_independent=independent, root=root,
    )
    return LedgerOutcome(
        ledger, capture, "entered",
        f"credit/contract={credit_pc:+.2f} shared={shared_contracts} indep={independent}",
    )


def _handle_open(strategy, config, ledger, capture, open_rec, quotes, today, root):
    spec = _RehydratedSpec(open_rec["spec"])
    missing = _missing_quote_legs(spec, quotes)
    if missing:
        return LedgerOutcome(ledger, capture, "warn", f"missing quote for {missing} -- holding, no exit evaluated")

    cost_to_close, put_pnl, call_pnl = mark_to_market(spec, quotes, open_rec["entry_credit_by_right"], capture)
    entry_credit = open_rec["entry_credit_per_contract"]
    dte_remaining = (spec.exdate - today).days
    result = check_exit(spec, today, cost_to_close, entry_credit, dte_remaining, put_pnl, call_pnl, config)

    if result is None:
        pnl_pc = entry_credit - cost_to_close
        return LedgerOutcome(ledger, capture, "holding", f"dte={dte_remaining} pnl/contract={pnl_pc:+.2f}", pnl_per_contract=pnl_pc)

    pnl_pc = entry_credit - cost_to_close
    record_exit(
        strategy, ledger, exit_reason=result.exit_reason, exit_date=today,
        exit_debit_per_contract=cost_to_close, pnl_per_contract=pnl_pc, root=root,
    )
    return LedgerOutcome(
        ledger, capture, "exited", f"{result.exit_reason} pnl/contract={pnl_pc:+.2f}",
        pnl_per_contract=pnl_pc, exit_reason=result.exit_reason,
    )


class _RehydratedLeg:
    """Rehydrates one leg dict (as stored in a ledger's `open` record) into
    an object with the attributes `shadow.execution`'s duck-typed functions
    expect. Kept private/minimal -- Shadow doesn't own a real `Leg` class
    (see module docstring); this is just enough structure to mark an
    already-recorded position to market.
    """

    def __init__(self, d: dict):
        self.optionid = d["optionid"]
        self.exdate = pd.Timestamp(d["exdate"])
        self.strike_price = d["strike_price"]
        self.delta = d.get("delta")
        self.best_bid = d.get("best_bid")
        self.best_offer = d.get("best_offer")
        self.right = d["right"]
        self.side = d["side"]
        self.impl_volatility = d.get("impl_volatility")
        # d.get, not d["weight"] -- an older stored ledger entry (recorded
        # before Leg.weight existed) has no "weight" key at all, same
        # backward-compat concern as shadow.execution's _leg_weight helper.
        # `_leg_to_dict` DOES serialize a real Leg's weight (verified: it's
        # captured generically via __dict__/dataclasses.asdict, not a
        # hand-picked field list) -- this was the compounding half of the
        # variance_swap correctness bug: shadow.execution's net_credit fix
        # alone wasn't sufficient, since every day after entry, mark-to-
        # market runs against a _RehydratedLeg reloaded from the JSON
        # store, and this constructor was silently dropping the weight that
        # WAS on disk. Found immediately after the execution.py fix, same
        # front-to-back review (2026-07-12).
        self.weight = d.get("weight", 1.0)


class _RehydratedSpec:
    """Rehydrates a stored `spec` dict (entry_date + legs) into an object
    with `legs`/`put_legs`/`call_legs`/`exdate`/`entry_date`, mirroring
    `PositionSpec`'s read-only surface -- see `_RehydratedLeg`."""

    def __init__(self, d: dict):
        self.entry_date = pd.Timestamp(d["entry_date"])
        self.legs = [_RehydratedLeg(leg) for leg in d["legs"]]

    @property
    def exdate(self) -> pd.Timestamp:
        return min(leg.exdate for leg in self.legs)

    @property
    def put_legs(self) -> list:
        return [leg for leg in self.legs if leg.right == "P"]

    @property
    def call_legs(self) -> list:
        return [leg for leg in self.legs if leg.right == "C"]

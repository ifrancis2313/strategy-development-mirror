import math

import pytest

from shadow.graduation_gate import (
    ShadowTrackRecord,
    check_shadow_graduation_gate,
    shadow_track_record,
)


def _closed_trade(pnl: float, entry: str, exit_: str) -> dict:
    return {
        "spec": {},
        "capture": 0.65,
        "entry_credit_by_right": {"P": 1.0, "C": 0.0},
        "entry_credit_per_contract": 1.0,
        "contracts_shared": 1,
        "contracts_independent": 1,
        "entry_date": entry,
        "exit_reason": "profit_target",
        "exit_date": exit_,
        "exit_debit_per_contract": 0.5,
        "pnl_per_contract": pnl,
    }


def _state_with(pnls: list[float]) -> dict:
    closed = [_closed_trade(p, f"2026-01-{i + 1:02d}", f"2026-01-{i + 2:02d}") for i, p in enumerate(pnls)]
    return {"floor": {"open": None, "closed": closed}, "target": {"open": None, "closed": closed}, "mid": {"open": None, "closed": []}}


def test_shadow_track_record_extracts_closed_trades() -> None:
    state = _state_with([10.0, -5.0, 20.0])
    tr = shadow_track_record(state, "vrp_harvester", "target")
    assert tr.n_closed_trades == 3
    assert tr.pnl_per_contract == (10.0, -5.0, 20.0)
    assert tr.first_entry_date == "2026-01-01"
    assert tr.last_exit_date == "2026-01-04"


def test_shadow_track_record_empty_ledger() -> None:
    state = _state_with([])
    tr = shadow_track_record(state, "vrp_harvester", "mid")
    assert tr.n_closed_trades == 0
    assert tr.first_entry_date is None
    assert tr.last_exit_date is None
    assert math.isnan(tr.live_sharpe)


def test_live_sharpe_matches_hand_computed_value() -> None:
    tr = ShadowTrackRecord("s", "target", 3, (10.0, -5.0, 20.0), "d1", "d2")
    pnls = [10.0, -5.0, 20.0]
    mean = sum(pnls) / 3
    variance = sum((x - mean) ** 2 for x in pnls) / 2
    expected = mean / math.sqrt(variance)
    assert tr.live_sharpe == pytest.approx(expected)


def test_track_record_not_yet_defined_when_no_target_set() -> None:
    tr = ShadowTrackRecord("s", "target", 5, (1.0,) * 5, "d1", "d2")
    report = check_shadow_graduation_gate(tr)
    assert report.track_record_verdict == "not_yet_defined"
    assert report.ready_to_graduate is True  # not_yet_defined never blocks


def test_track_record_fails_below_target() -> None:
    tr = ShadowTrackRecord("s", "target", 3, (1.0, 2.0, 3.0), "d1", "d2")
    report = check_shadow_graduation_gate(tr, min_trades_target=10)
    assert report.track_record_verdict == "fail"
    assert report.ready_to_graduate is False


def test_track_record_passes_at_target() -> None:
    tr = ShadowTrackRecord("s", "target", 10, tuple(range(10)), "d1", "d2")
    report = check_shadow_graduation_gate(tr, min_trades_target=10)
    assert report.track_record_verdict == "pass"


def test_divergence_not_yet_defined_without_reference() -> None:
    tr = ShadowTrackRecord("s", "target", 5, (10.0, -5.0, 20.0, 3.0, 8.0), "d1", "d2")
    report = check_shadow_graduation_gate(tr)
    assert report.divergence_verdict == "not_yet_defined"


def test_divergence_passes_within_tolerance() -> None:
    tr = ShadowTrackRecord("s", "target", 5, (10.0, -5.0, 20.0, 3.0, 8.0), "d1", "d2")
    live_sharpe = tr.live_sharpe
    report = check_shadow_graduation_gate(tr, reference_sharpe=live_sharpe, max_divergence=0.01)
    assert report.divergence_verdict == "pass"
    assert report.ready_to_graduate is True


def test_divergence_fails_outside_tolerance() -> None:
    tr = ShadowTrackRecord("s", "target", 5, (10.0, -5.0, 20.0, 3.0, 8.0), "d1", "d2")
    report = check_shadow_graduation_gate(tr, reference_sharpe=999.0, max_divergence=0.01)
    assert report.divergence_verdict == "fail"
    assert report.ready_to_graduate is False


def test_divergence_not_yet_defined_when_too_few_trades_for_a_real_sharpe() -> None:
    tr = ShadowTrackRecord("s", "target", 1, (10.0,), "d1", "d2")
    report = check_shadow_graduation_gate(tr, reference_sharpe=1.0, max_divergence=0.5)
    assert report.divergence_verdict == "not_yet_defined"
    assert report.live_sharpe is None


def test_divergence_not_yet_defined_when_live_sharpe_is_nan_from_zero_variance() -> None:
    # >= 2 trades, but identical pnl_per_contract -> zero variance -> NaN
    # live_sharpe, distinct from the too-few-trades case above. Must not
    # fall through to the pass/fail comparison, where `nan <= max_divergence`
    # is always False and would silently produce "fail".
    tr = ShadowTrackRecord("s", "target", 2, (5.0, 5.0), "d1", "d2")
    assert math.isnan(tr.live_sharpe)
    report = check_shadow_graduation_gate(tr, reference_sharpe=1.0, max_divergence=0.5)
    assert report.divergence_verdict == "not_yet_defined"
    assert report.ready_to_graduate is True


def test_live_sharpe_is_computed_at_exactly_two_closed_trades() -> None:
    """`n_closed_trades >= 2 else None` -- mutation testing (2026-07-12)
    found the boundary survives being tightened to `> 2`. Two trades with
    distinct pnl (non-zero variance) must produce a real live_sharpe, not
    None -- the one-trade case above can't distinguish >= from >."""
    tr = ShadowTrackRecord("s", "target", 2, (10.0, -5.0), "d1", "d2")
    report = check_shadow_graduation_gate(tr, reference_sharpe=1.0, max_divergence=0.5)
    assert report.live_sharpe is not None
    assert report.divergence_verdict in ("pass", "fail")  # actually evaluated, not not_yet_defined


def test_divergence_boundary_is_inclusive() -> None:
    """`abs(live_sharpe - reference_sharpe) <= max_divergence` -- mutation
    testing (2026-07-12) found the boundary survives being tightened to `<`.
    reference_sharpe is derived from the real live_sharpe so the difference
    lands bit-exact at max_divergence, not just approximately."""
    tr = ShadowTrackRecord("s", "target", 5, (10.0, -5.0, 20.0, 3.0, 8.0), "d1", "d2")
    max_divergence = 0.5
    reference_sharpe = tr.live_sharpe - max_divergence
    report = check_shadow_graduation_gate(tr, reference_sharpe=reference_sharpe, max_divergence=max_divergence)
    assert abs(report.live_sharpe - report.reference_sharpe) == max_divergence  # confirms exact boundary
    assert report.divergence_verdict == "pass"


def test_divergence_not_yet_defined_when_only_max_divergence_is_missing() -> None:
    """The three-way `A is None or B is None or C is None` guard survived
    being flipped to `(A is None and B is None) or C is None` (2026-07-12)
    -- with reference_sharpe and live_sharpe both real but max_divergence
    None, the buggy version skips straight to the pass/fail comparison and
    crashes on `abs(x) <= None`, instead of reporting not_yet_defined."""
    tr = ShadowTrackRecord("s", "target", 5, (10.0, -5.0, 20.0, 3.0, 8.0), "d1", "d2")
    report = check_shadow_graduation_gate(tr, reference_sharpe=1.0, max_divergence=None)
    assert report.divergence_verdict == "not_yet_defined"
    assert report.ready_to_graduate is True


def test_never_imports_testing_engine_or_strategies() -> None:
    import ast
    from pathlib import Path

    tree = ast.parse(Path("shadow/graduation_gate.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("testing.engine")
            assert not node.module.startswith("testing.strategies")


def test_summary_mentions_every_criterion() -> None:
    tr = ShadowTrackRecord("s", "target", 5, (10.0, -5.0, 20.0, 3.0, 8.0), "d1", "d2")
    report = check_shadow_graduation_gate(tr, min_trades_target=10, reference_sharpe=1.0, max_divergence=0.5)
    text = report.summary()
    for keyword in ("Track record", "divergence", "Ready to graduate"):
        assert keyword in text

"""Tests `shadow.etf_ledger.run_daily`'s day-by-day live process against a
synthetic fake config -- position-decision boundaries, cost accounting, and
entry-price bookkeeping.

(This mirror omits an additional cross-check this file used to carry,
proving the day-by-day replay reproduces a vectorized reference backtest's
math exactly -- dropped here because that reference implementation lived in
an excluded strategy package, not because the property itself isn't worth
testing.)
"""

from __future__ import annotations

from dataclasses import dataclass

from shadow import etf_ledger


@dataclass
class _FakeConfig:
    fixed_notional: float = 100_000.0
    round_trip_cost_bps: float = 10.0
    contango_threshold: float | None = None
    signal_tenor: str = "vix3m"
    reference_tenor: str = "vix"


def test_first_run_is_flat_and_costs_nothing(tmp_path):
    config = _FakeConfig()
    outcome = etf_ledger.run_daily("vix_carry_test", config, signal_today=2.0, price_today=50.0,
                                     today="2026-01-05", root=str(tmp_path))
    assert outcome.action == "first_run"
    assert outcome.position == 0
    assert outcome.pnl_dollars == 0.0


def test_decide_position_boundary_is_strict_not_inclusive():
    """`last_signal > config.contango_threshold` -- mutation testing
    (2026-07-12) found the boundary survives being loosened to `>=`. A
    signal exactly at the threshold must stay flat (0), not enter (1) --
    this is vix_carry's live position-decision signal, one of only 3
    strategies actually running on the droplet."""
    config = _FakeConfig(contango_threshold=1.5)
    assert etf_ledger._decide_position(config, last_signal=1.5) == 0


def test_entry_price_survives_a_holding_day_unchanged(tmp_path):
    """`position_today == 1 and (prev_position != 1)` -- mutation testing
    (2026-07-12) found this survives being flipped to `prev_position == 1`,
    which would silently overwrite entry_price with today's price on every
    *holding* day instead of only on a genuine new entry."""
    config = _FakeConfig(contango_threshold=None)
    root = str(tmp_path)
    strategy = "vix_carry_test"

    etf_ledger.run_daily(strategy, config, signal_today=2.0, price_today=50.0, today="2026-01-05", root=root)
    etf_ledger.run_daily(strategy, config, signal_today=2.1, price_today=51.0, today="2026-01-06", root=root)  # entered @ 51.0
    etf_ledger.run_daily(strategy, config, signal_today=2.0, price_today=55.0, today="2026-01-07", root=root)  # holding, price moved

    state = etf_ledger.load_state(strategy, root=root)
    assert state["entry_price"] == 51.0  # unchanged by the holding day's different price


def test_entry_price_resets_to_none_when_exiting_to_flat(tmp_path):
    """`elif position_today == 0: entry_price = None` -- mutation testing
    (2026-07-12) found this survives being flipped to `!= 0`, which would
    clear entry_price on every day the position IS 1 (the opposite of
    intended) instead of only when flattening out."""
    config = _FakeConfig(contango_threshold=1.5)
    root = str(tmp_path)
    strategy = "vix_carry_test"

    etf_ledger.run_daily(strategy, config, signal_today=2.0, price_today=50.0, today="2026-01-05", root=root)  # first_run, flat
    etf_ledger.run_daily(strategy, config, signal_today=2.0, price_today=51.0, today="2026-01-06", root=root)  # last_signal=2.0>1.5 -> entered
    state_after_entry = etf_ledger.load_state(strategy, root=root)
    assert state_after_entry["entry_price"] == 51.0

    etf_ledger.run_daily(strategy, config, signal_today=1.0, price_today=52.0, today="2026-01-07", root=root)  # last_signal=2.0>1.5 -> still long
    etf_ledger.run_daily(strategy, config, signal_today=1.0, price_today=53.0, today="2026-01-08", root=root)  # last_signal=1.0<=1.5 -> flat now
    state_after_exit = etf_ledger.load_state(strategy, root=root)
    assert state_after_exit["position"] == 0
    assert state_after_exit["entry_price"] is None


def test_naive_mode_enters_and_holds_after_first_run(tmp_path):
    config = _FakeConfig(contango_threshold=None)
    root = str(tmp_path)
    strategy = "vix_carry_test"

    day1 = etf_ledger.run_daily(strategy, config, signal_today=2.0, price_today=50.0, today="2026-01-05", root=root)
    day2 = etf_ledger.run_daily(strategy, config, signal_today=2.1, price_today=51.0, today="2026-01-06", root=root)
    day3 = etf_ledger.run_daily(strategy, config, signal_today=2.0, price_today=50.5, today="2026-01-07", root=root)

    assert day1.position == 0
    assert day2.action == "entered" and day2.position == 1
    assert day3.action == "holding" and day3.position == 1

    state = etf_ledger.load_state(strategy, root=root)
    assert len(state["history"]) == 3

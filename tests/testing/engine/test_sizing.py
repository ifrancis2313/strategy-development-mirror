import numpy as np
import pandas as pd

from testing.engine.execution_config import ExecutionConfig
from testing.engine.leg import Leg
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import TradeResult
from testing.engine.sizing import SizingEngine, contracts_for_entry

_EXDATE = pd.Timestamp("2020-02-15")


def _spread(short_strike=95, long_strike=90):
    short_put = Leg(optionid="short", exdate=_EXDATE, strike_price=short_strike, delta=-0.20, best_bid=3.0, best_offer=3.2, right="P", side="short")
    long_put = Leg(optionid="long", exdate=_EXDATE, strike_price=long_strike, delta=-0.10, best_bid=1.0, best_offer=1.2, right="P", side="long")
    return PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[short_put, long_put])


def _condor(short_put=95, long_put=90, short_call=105, long_call=115):
    legs = [
        Leg(optionid="sp", exdate=_EXDATE, strike_price=short_put, delta=-0.20, best_bid=3.0, best_offer=3.2, right="P", side="short"),
        Leg(optionid="lp", exdate=_EXDATE, strike_price=long_put, delta=-0.10, best_bid=1.0, best_offer=1.2, right="P", side="long"),
        Leg(optionid="sc", exdate=_EXDATE, strike_price=short_call, delta=0.30, best_bid=2.0, best_offer=2.2, right="C", side="short"),
        Leg(optionid="lc", exdate=_EXDATE, strike_price=long_call, delta=0.20, best_bid=0.8, best_offer=1.0, right="C", side="long"),
    ]
    return PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=legs)


def _config(**overrides):
    defaults = dict(fixed_notional=100_000.0, contract_multiplier=100)
    defaults.update(overrides)
    return ExecutionConfig(**defaults)


def test_contracts_for_entry_uses_put_width_for_a_put_spread():
    spec = _spread(short_strike=95, long_strike=90)  # width 5
    config = _config()
    # max_loss_per_contract = (5 - 1.8) * 100 = 320 -> 100000 // 320 = 312
    assert contracts_for_entry(spec, entry_credit=1.8, config=config) == 312


def test_contracts_for_entry_uses_wider_side_for_iron_condor():
    # put width 5, call width 10 -- max loss must use the wider call side
    spec = _condor(short_put=95, long_put=90, short_call=105, long_call=115)
    config = _config()
    # max_loss_per_contract = (10 - 2.8) * 100 = 720 -> 100000 // 720 = 138
    assert contracts_for_entry(spec, entry_credit=2.8, config=config) == 138


def test_contracts_for_entry_respects_explicit_notional_override():
    spec = _spread()
    config = _config()
    assert contracts_for_entry(spec, entry_credit=1.8, config=config, notional=50_000.0) == 156


def test_contracts_for_entry_floors_at_one_when_max_loss_exceeds_notional():
    spec = _spread(short_strike=995, long_strike=90)  # huge width, tiny notional
    config = _config(fixed_notional=100.0)
    assert contracts_for_entry(spec, entry_credit=1.8, config=config) == 1


def test_size_fixed_notional_matches_direct_contracts_for_entry_call():
    spec = _spread()
    config = _config()
    result = TradeResult(entry_date=spec.entry_date, exit_date=pd.Timestamp("2020-02-01"),
                          exit_reason="profit_target", entry_credit=1.8, exit_debit=0.8, pnl=1.0)

    sized = SizingEngine().size_fixed_notional([(spec, result)], config)

    assert len(sized) == 1
    assert sized[0].contracts == contracts_for_entry(spec, result.entry_credit, config)
    assert np.isclose(sized[0].pnl_dollars, result.pnl * config.contract_multiplier * sized[0].contracts)


def test_size_vol_scaled_passes_adjusted_notional_through():
    spec = _spread()
    config = _config()
    result = TradeResult(entry_date=spec.entry_date, exit_date=pd.Timestamp("2020-02-01"),
                          exit_reason="profit_target", entry_credit=1.8, exit_debit=0.8, pnl=1.0)
    vol_by_date = {spec.entry_date: 40.0}  # double reference vol -> half notional

    sized = SizingEngine().size_vol_scaled([(spec, result)], config, vol_by_date, reference_vol=20.0)

    expected_contracts = contracts_for_entry(spec, result.entry_credit, config, notional=50_000.0)
    assert sized[0].contracts == expected_contracts


def test_size_vol_scaled_is_scale_agnostic_between_vix_points_and_fractional_rv():
    # Same 2x-elevated-vol-halves-notional ratio, expressed in VIX-point units
    # (20 -> 40) and in fractional annualized-RV units (0.15 -> 0.30) -- only
    # the ratio should matter, confirming size_vol_scaled works unmodified
    # for testing.engine.realized_vol.compute_realized_vol's output, not just
    # VIX, per its docstring's "deliberately vol-source-agnostic" claim.
    spec = _spread()
    config = _config()
    result = TradeResult(entry_date=spec.entry_date, exit_date=pd.Timestamp("2020-02-01"),
                          exit_reason="profit_target", entry_credit=1.8, exit_debit=0.8, pnl=1.0)

    vix_sized = SizingEngine().size_vol_scaled(
        [(spec, result)], config, {spec.entry_date: 40.0}, reference_vol=20.0,
    )
    rv_sized = SizingEngine().size_vol_scaled(
        [(spec, result)], config, {spec.entry_date: 0.30}, reference_vol=0.15,
    )

    assert vix_sized[0].contracts == rv_sized[0].contracts

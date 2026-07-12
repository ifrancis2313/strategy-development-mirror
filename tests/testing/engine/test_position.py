import pandas as pd

from testing.engine.leg import Leg
from testing.engine.position import PositionSpec


def _leg(optionid, exdate, strike, side, right="P"):
    return Leg(optionid=optionid, exdate=pd.Timestamp(exdate), strike_price=strike, delta=-0.20,
               best_bid=1.0, best_offer=1.2, right=right, side=side)


def test_exdate_derives_to_shared_expiry_when_every_leg_matches():
    short_leg = _leg("short", "2020-02-15", 95, "short")
    long_leg = _leg("long", "2020-02-15", 90, "long")
    spec = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[short_leg, long_leg])

    assert spec.exdate == pd.Timestamp("2020-02-15")


def test_exdate_derives_to_the_earliest_leg_expiry_for_a_calendar():
    near_leg = _leg("near", "2020-02-15", 100, "short")
    far_leg = _leg("far", "2020-03-16", 100, "long")
    spec = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[near_leg, far_leg])

    assert spec.exdate == pd.Timestamp("2020-02-15")


def test_exdate_is_order_independent():
    near_leg = _leg("near", "2020-02-15", 100, "short")
    far_leg = _leg("far", "2020-03-16", 100, "long")

    forward = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[near_leg, far_leg])
    backward = PositionSpec(entry_date=pd.Timestamp("2020-01-15"), legs=[far_leg, near_leg])

    assert forward.exdate == backward.exdate == pd.Timestamp("2020-02-15")

import math

import pytest

from testing.engine.black_scholes import black_scholes_price


def test_matches_textbook_reference_values():
    # Hull, "Options, Futures, and Other Derivatives" -- the standard
    # worked example: S=42, K=40, r=0.10, vol=0.20, T=0.5.
    call = black_scholes_price(spot=42, strike=40, time_to_expiry_years=0.5, vol=0.20, right="C", rate=0.10)
    put = black_scholes_price(spot=42, strike=40, time_to_expiry_years=0.5, vol=0.20, right="P", rate=0.10)
    assert call == pytest.approx(4.76, abs=0.01)
    assert put == pytest.approx(0.81, abs=0.01)


def test_put_call_parity_holds():
    spot, strike, t, vol, rate = 2000.0, 1950.0, 0.1, 0.18, 0.02
    call = black_scholes_price(spot, strike, t, vol, "C", rate)
    put = black_scholes_price(spot, strike, t, vol, "P", rate)
    # C - P = S - K*e^(-rT)
    assert (call - put) == pytest.approx(spot - strike * math.exp(-rate * t), abs=1e-6)


def test_zero_time_collapses_to_intrinsic_value():
    assert black_scholes_price(spot=110, strike=100, time_to_expiry_years=0.0, vol=0.2, right="C") == 10.0
    assert black_scholes_price(spot=90, strike=100, time_to_expiry_years=0.0, vol=0.2, right="P") == 10.0
    assert black_scholes_price(spot=90, strike=100, time_to_expiry_years=0.0, vol=0.2, right="C") == 0.0


def test_zero_vol_collapses_to_intrinsic_value():
    assert black_scholes_price(spot=110, strike=100, time_to_expiry_years=0.5, vol=0.0, right="C") == 10.0


def test_higher_vol_increases_option_value():
    low_vol = black_scholes_price(spot=2000, strike=2000, time_to_expiry_years=0.1, vol=0.15, right="C")
    high_vol = black_scholes_price(spot=2000, strike=2000, time_to_expiry_years=0.1, vol=0.30, right="C")
    assert high_vol > low_vol


def test_price_is_never_negative():
    price = black_scholes_price(spot=1000, strike=2000, time_to_expiry_years=0.05, vol=0.10, right="C")
    assert price >= 0.0

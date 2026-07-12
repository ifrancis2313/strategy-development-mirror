from shadow.market_calendar import is_trading_day


def test_regular_weekday_is_a_trading_day():
    assert is_trading_day("2026-07-13") is True  # a Monday


def test_weekend_is_not_a_trading_day():
    assert is_trading_day("2026-07-11") is False  # a Saturday


def test_nyse_holiday_is_not_a_trading_day():
    assert is_trading_day("2026-01-01") is False  # New Year's Day


def test_day_after_thanksgiving_is_a_trading_day():
    # An early-close session (market open at 10 AM ET), still a real session.
    assert is_trading_day("2026-11-27") is True

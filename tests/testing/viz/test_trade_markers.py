import pandas as pd

from testing.engine.leg import Leg
from testing.engine.position import PositionSpec
from testing.engine.position_simulator import TradeResult
from testing.viz.trade_markers import entry_exit_markers


def _price() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    return pd.Series(range(100, 160), index=idx, dtype=float)


def _leg(exdate: pd.Timestamp) -> Leg:
    return Leg(
        optionid=1, exdate=exdate, strike_price=100.0, delta=-0.2, best_bid=1.0, best_offer=1.2,
        right="P", side="short",
    )


def _trade(entry_date: pd.Timestamp, exit_date: pd.Timestamp, pnl: float) -> tuple[PositionSpec, TradeResult]:
    spec = PositionSpec(entry_date=entry_date, legs=[_leg(exit_date)])
    result = TradeResult(entry_date, exit_date, "profit_target", 10.0, 10.0 - pnl, pnl)
    return spec, result


def test_train_and_test_trades_get_distinct_traces() -> None:
    price = _price()
    train_end = price.index[30]
    trades = [
        _trade(price.index[5], price.index[9], 5.0),  # train
        _trade(price.index[45], price.index[49], -15.0),  # test
    ]
    fig = entry_exit_markers(price, trades, train_end=train_end)

    names = {trace.name for trace in fig.data}
    assert "Entry (train)" in names
    assert "Exit (train)" in names
    assert "Entry (test)" in names
    assert "Exit (test)" in names
    # price line + 4 marker traces
    assert len(fig.data) == 5


def test_all_trades_in_train_period_omits_test_traces() -> None:
    price = _price()
    train_end = price.index[55]
    trades = [_trade(price.index[5], price.index[9], 5.0)]
    fig = entry_exit_markers(price, trades, train_end=train_end)

    names = {trace.name for trace in fig.data}
    assert "Entry (train)" in names
    assert "Entry (test)" not in names


def test_entry_and_exit_use_distinct_marker_symbols() -> None:
    price = _price()
    trades = [_trade(price.index[5], price.index[9], 5.0)]
    fig = entry_exit_markers(price, trades, train_end=price.index[30])

    entry_trace = next(t for t in fig.data if t.name == "Entry (train)")
    exit_trace = next(t for t in fig.data if t.name == "Exit (train)")
    assert entry_trace.marker.symbol != exit_trace.marker.symbol
    # color still differs by train/test even though symbol carries entry/exit --
    # neither channel is doing the whole job alone.
    assert entry_trace.marker.color == exit_trace.marker.color


def test_no_trades_still_renders_price_line() -> None:
    price = _price()
    fig = entry_exit_markers(price, [], train_end=price.index[30])
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"

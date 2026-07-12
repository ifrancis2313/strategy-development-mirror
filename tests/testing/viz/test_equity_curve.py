import pandas as pd

from testing.viz.equity_curve import equity_curve_and_drawdown


def _equity() -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    # up, then a drawdown, then a partial recovery -- exercises cummax/drawdown math
    values = [0, 100, 200, 300, 250, 150, 100, 200, 300, 400] + [400] * 10
    return pd.Series(values, index=idx, dtype=float).rename("equity")


def test_renders_two_single_axis_subplots() -> None:
    fig = equity_curve_and_drawdown(_equity(), fixed_notional=1_000.0)
    assert len(fig.data) == 2
    assert fig.data[0].type == "scatter"
    assert fig.data[1].type == "scatter"
    # one y-axis per subplot (never a dual-axis single chart)
    assert fig.layout.yaxis.title.text == "Cumulative P&L ($)"
    assert fig.layout.yaxis2.title.text == "Drawdown (%)"


def test_drawdown_matches_performance_analyzer_convention() -> None:
    equity = _equity()
    fixed_notional = 1_000.0
    fig = equity_curve_and_drawdown(equity, fixed_notional=fixed_notional)
    drawdown_trace = fig.data[1]

    running_max = equity.cummax()
    expected = ((equity - running_max) / fixed_notional * 100).values
    assert list(drawdown_trace.y) == list(expected)


def test_train_end_adds_a_split_marker() -> None:
    equity = _equity()
    without_split = equity_curve_and_drawdown(equity, fixed_notional=1_000.0)
    with_split = equity_curve_and_drawdown(equity, fixed_notional=1_000.0, train_end=equity.index[10])
    assert len(with_split.layout.shapes) > len(without_split.layout.shapes)
    assert len(with_split.layout.annotations) > len(without_split.layout.annotations)


def test_empty_equity_curve_does_not_raise() -> None:
    fig = equity_curve_and_drawdown(pd.Series(dtype=float), fixed_notional=1_000.0)
    assert len(fig.data) == 2

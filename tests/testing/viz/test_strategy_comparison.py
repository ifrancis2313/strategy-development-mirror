import pandas as pd
import pytest

from testing.engine.performance import PerformanceReport
from testing.viz.strategy_comparison import (
    StrategySeries,
    multi_strategy_comparison_chart,
    multi_strategy_metrics_table,
)


def _series(name: str, offset: float = 0.0) -> StrategySeries:
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    equity = pd.Series(range(10), index=idx, dtype=float) + offset
    metrics = PerformanceReport(
        num_trades=10, win_rate=0.6, avg_pnl_dollars=12.5, sharpe=0.4, max_drawdown_pct=-0.05, calmar=1.2
    )
    return StrategySeries(name=name, equity_curve=equity, metrics=metrics)


def test_single_strategy_has_no_legend() -> None:
    fig = multi_strategy_comparison_chart([_series("strategy_a")])
    assert len(fig.data) == 1
    assert fig.layout.showlegend is False


def test_multiple_strategies_get_a_legend_and_fixed_slot_colors() -> None:
    series = [_series("strategy_a"), _series("strategy_b", offset=5)]
    fig = multi_strategy_comparison_chart(series)
    assert len(fig.data) == 2
    assert fig.layout.showlegend is True
    # fixed categorical order: first strategy always gets slot 0's color
    from testing.viz.theme import CATEGORICAL_LIGHT

    assert fig.data[0].line.color == CATEGORICAL_LIGHT[0]
    assert fig.data[1].line.color == CATEGORICAL_LIGHT[1]


def test_extensible_to_many_strategies_still_within_ceiling() -> None:
    series = [_series(f"strategy_{i}") for i in range(8)]
    fig = multi_strategy_comparison_chart(series)
    assert len(fig.data) == 8


def test_raises_past_categorical_ceiling_instead_of_cycling() -> None:
    series = [_series(f"strategy_{i}") for i in range(9)]
    with pytest.raises(ValueError):
        multi_strategy_comparison_chart(series)


def test_metrics_table_has_one_row_per_strategy() -> None:
    series = [_series("strategy_a"), _series("strategy_b")]
    table = multi_strategy_metrics_table(series)
    assert list(table["strategy"]) == ["strategy_a", "strategy_b"]
    assert set(table.columns) == {
        "strategy", "num_trades", "win_rate", "avg_pnl_dollars", "sharpe", "max_drawdown_pct", "calmar",
    }

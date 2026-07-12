"""Side-by-side multi-strategy comparison: overlaid equity curves + a metrics
table. Works with however many strategies currently exist (just
`vrp_harvester` today) and is extensible -- issue 06 (porting the other 6
`Options_Trading_Strategy` strategies) doesn't need any change here, it just
means `render_comparison`'s caller passes a longer `series` list.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go

from testing.engine.performance import PerformanceReport
from testing.viz.theme import Mode, apply_layout, category_color, get_theme

# dataviz skill's series-count ladder: past 8 hues, fold the tail into "Other" or a
# small-multiples facet instead of cycling a 9th generated hue back onto an earlier slot.
_CATEGORICAL_CEILING = 8


@dataclass(frozen=True)
class StrategySeries:
    """One named strategy's already-computed equity curve + performance
    report -- the same objects `PortfolioAggregator.build_equity_curve` and
    `PerformanceAnalyzer.analyze` already produce, so a caller running the
    engine end to end has zero conversion work before calling this module.
    """

    name: str
    equity_curve: pd.Series
    metrics: PerformanceReport


def multi_strategy_comparison_chart(
    series: list[StrategySeries],
    *,
    mode: Mode = "light",
    title: str | None = "Strategy comparison -- equity curves",
) -> go.Figure:
    """Overlaid equity curves, one fixed categorical hue per strategy in the
    order given (never reassigned if the list is later filtered -- "color
    follows the entity, never its rank"). Raises if given more than the
    8-hue categorical ceiling, rather than silently cycling a hue back onto
    an earlier strategy (which would make two different strategies visually
    indistinguishable).
    """
    if len(series) > _CATEGORICAL_CEILING:
        raise ValueError(
            f"multi_strategy_comparison_chart: {len(series)} strategies exceeds the "
            f"{_CATEGORICAL_CEILING}-hue categorical ceiling -- fold the tail into a "
            "small-multiples facet instead of cycling a hue (see the dataviz skill's "
            "series-count ladder)."
        )

    theme = get_theme(mode)
    fig = go.Figure()
    show_legend = len(series) > 1  # a single series needs no legend -- the title names it

    for slot, s in enumerate(series):
        equity = s.equity_curve.sort_index()
        fig.add_trace(
            go.Scatter(
                x=equity.index,
                y=equity.values,
                mode="lines",
                line={"color": category_color(theme, slot), "width": 2},
                name=s.name,
                hovertemplate=f"{s.name}<br>" + "%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            )
        )

    apply_layout(fig, theme, title=title, show_legend=show_legend)
    fig.update_yaxes(title_text="Cumulative P&L ($)")
    fig.update_xaxes(title_text="Date")
    return fig


def multi_strategy_metrics_table(series: list[StrategySeries]) -> pd.DataFrame:
    """The tabular "relief" side of the comparison -- per the dataviz skill,
    a chart with more than a handful of distinguishable classes wants a table
    alongside it, and this repo's rigor bar wants the underlying numbers
    inspectable, not just a visual read of curve height. One row per
    strategy, `PerformanceReport`'s own fields as columns.
    """
    rows = [
        {
            "strategy": s.name,
            "num_trades": s.metrics.num_trades,
            "win_rate": s.metrics.win_rate,
            "avg_pnl_dollars": s.metrics.avg_pnl_dollars,
            "sharpe": s.metrics.sharpe,
            "max_drawdown_pct": s.metrics.max_drawdown_pct,
            "calmar": s.metrics.calmar,
        }
        for s in series
    ]
    return pd.DataFrame(rows)

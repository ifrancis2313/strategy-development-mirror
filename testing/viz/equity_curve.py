"""Equity curve + drawdown chart.

Two measures on different scales (cumulative dollars vs. drawdown percent) --
per the `dataviz` skill's form guidance ("two measures of different scale ->
two charts / small multiples," never a dual-axis chart), this is two stacked
subplots sharing one x-axis, each with exactly one y-axis, not one chart with
two y-scales.

Both series are single-series charts (one line, one filled area) so neither
needs a legend box per the skill's rule ("a single series needs no legend --
the title/axis label names it"); each subplot's y-axis title does that
naming instead.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from testing.engine.performance import drawdown_series
from testing.viz.theme import Mode, apply_layout, category_color, get_theme


def equity_curve_and_drawdown(
    equity: pd.Series,
    fixed_notional: float,
    *,
    train_end: pd.Timestamp | str | None = None,
    mode: Mode = "light",
    title: str | None = "Equity curve & drawdown",
) -> go.Figure:
    """`equity` is `PortfolioAggregator.build_equity_curve()`'s output:
    cumulative dollar P&L indexed by trade exit date. `fixed_notional` must
    be the same notional the backtest was sized against (`config.
    fixed_notional`), so the drawdown percentage matches `PerformanceAnalyzer`
    exactly.

    `train_end`, if given, draws a vertical marker splitting the chart into
    the strategy's tuning (train) period and everything after (test) -- the
    same boundary `testing.viz.trade_markers.entry_exit_markers` uses, so the
    two charts read consistently when shown together.
    """
    theme = get_theme(mode)
    equity = equity.sort_index()
    drawdown = drawdown_series(equity, fixed_notional)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Cumulative P&L ($)", "Drawdown (% of notional)"),
    )

    equity_color = category_color(theme, 0)
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            mode="lines",
            line={"color": equity_color, "width": 2},
            name="Cumulative P&L",
            hovertemplate="%{x|%Y-%m-%d}<br>$%{y:,.0f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    drawdown_color = category_color(theme, 5)  # red -- severity, not a second series identity
    r, g, b = int(drawdown_color[1:3], 16), int(drawdown_color[3:5], 16), int(drawdown_color[5:7], 16)
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode="lines",
            line={"color": drawdown_color, "width": 2},
            fill="tozeroy",
            fillcolor=f"rgba({r},{g},{b},0.15)",
            name="Drawdown",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    if train_end is not None:
        train_end_ts = pd.Timestamp(train_end)
        for row in (1, 2):
            fig.add_vline(
                x=train_end_ts,
                line={"color": theme.text_muted, "width": 1, "dash": "dash"},
                row=row,
                col=1,
            )
        fig.add_annotation(
            x=train_end_ts,
            y=1.0,
            yref="y domain",
            row=1,
            col=1,
            text="train | test",
            showarrow=False,
            font={"color": theme.text_secondary, "size": 11},
            yanchor="bottom",
        )

    apply_layout(fig, theme, title=title, show_legend=False)
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    return fig

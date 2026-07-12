"""Entry/exit markers overlaid on the underlying price series, split by
training vs. test period.

Per the PRD's own user story (#26): "I want to see exactly where a strategy
enters and exits trades, overlaid on the underlying price series, split by
training vs. test period, so that I can sanity-check the strategy isn't only
'working' on the period it was tuned on." The price series is context (an
"emphasis" framing per the `dataviz` skill -- the line is the backdrop, the
markers are the point), so it's drawn in muted ink, not a categorical hue.
Markers carry two encodings: color for train vs. test (2-series categorical,
comfortable per the skill's series-count ladder) and marker symbol for entry
vs. exit -- color is never the only channel identifying a point.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from testing.engine.position import PositionSpec
from testing.engine.position_simulator import TradeResult
from testing.viz.theme import Mode, apply_layout, category_color, get_theme


def _nearest_price(price: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Looks up each date's price, falling back to the nearest available
    trading day -- entry/exit dates come from the option chain's own `date`
    column, which should already coincide with the synthetic spot series'
    index (both are derived from the same chain), but `nearest` keeps this
    robust to the rare day where one side has a gap the other doesn't.
    """
    return price.reindex(dates, method="nearest")


def entry_exit_markers(
    price: pd.Series,
    trades: list[tuple[PositionSpec, TradeResult]],
    train_end: pd.Timestamp | str,
    *,
    mode: Mode = "light",
    title: str | None = "Entries/exits on underlying price, train vs. test",
) -> go.Figure:
    """`price` is a daily underlying price/proxy series indexed by date (e.g.
    `SyntheticSpotEngine.compute_spot_series`'s output). `trades` is
    `walk_forward`'s own output shape -- no separate marker-building step
    needed by the caller. `train_end` splits every trade by its entry date:
    on/before `train_end` is "train" (the period the strategy's params were
    presumably tuned against), after is "test" -- so a strategy that only
    "works" in-sample shows a visibly different marker pattern (win/loss mix,
    density) on either side of the line.
    """
    theme = get_theme(mode)
    train_end_ts = pd.Timestamp(train_end)
    price = price.sort_index()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price.index,
            y=price.values,
            mode="lines",
            line={"color": theme.text_muted, "width": 1.5},
            name="Underlying (synthetic spot)",
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:,.1f}<extra></extra>",
        )
    )

    train_color = category_color(theme, 0)  # blue
    test_color = category_color(theme, 1)  # aqua

    entry_dates = pd.DatetimeIndex([result.entry_date for _, result in trades])
    exit_dates = pd.DatetimeIndex([result.exit_date for _, result in trades])
    is_train = entry_dates <= train_end_ts

    label = {"train": "train", "test": "test"}
    symbol = {"entry": "triangle-up", "exit": "triangle-down"}
    color = {"train": train_color, "test": test_color}

    # One trace per (period, entry|exit) combination -- 4 legend entries max,
    # rather than one trace per trade -- so each combination is exactly one
    # legend entry and stays within the skill's "comfortable" series count.
    for period in ("train", "test"):
        mask = is_train if period == "train" else ~is_train
        period_entry_dates = entry_dates[mask]
        period_exit_dates = exit_dates[mask]

        if len(period_entry_dates):
            fig.add_trace(
                go.Scatter(
                    x=period_entry_dates,
                    y=_nearest_price(price, period_entry_dates).values,
                    mode="markers",
                    marker={"symbol": symbol["entry"], "size": 9, "color": color[period], "line": {"width": 0}},
                    name=f"Entry ({label[period]})",
                    hovertemplate="%{x|%Y-%m-%d}<br>entry, " + label[period] + "<extra></extra>",
                )
            )
        if len(period_exit_dates):
            fig.add_trace(
                go.Scatter(
                    x=period_exit_dates,
                    y=_nearest_price(price, period_exit_dates).values,
                    mode="markers",
                    marker={
                        "symbol": symbol["exit"],
                        "size": 9,
                        "color": color[period],
                        "line": {"width": 1, "color": theme.surface},
                    },
                    name=f"Exit ({label[period]})",
                    hovertemplate="%{x|%Y-%m-%d}<br>exit, " + label[period] + "<extra></extra>",
                )
            )

    fig.add_vline(
        x=train_end_ts,
        line={"color": theme.text_muted, "width": 1, "dash": "dash"},
    )
    fig.add_annotation(
        x=train_end_ts,
        y=1.0,
        yref="y domain",
        text="train | test",
        showarrow=False,
        font={"color": theme.text_secondary, "size": 11},
        yanchor="bottom",
    )

    apply_layout(fig, theme, title=title, show_legend=True)
    fig.update_yaxes(title_text="Underlying price")
    fig.update_xaxes(title_text="Date")
    return fig

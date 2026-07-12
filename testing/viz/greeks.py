"""Greek exposure over time -- real chart, wired against issue 04's
`testing.metrics.options.portfolio_greek_exposure` now that it's merged.

Four small multiples (delta/gamma/vega/theta), one row each sharing a date
x-axis, rather than one chart with four overlaid lines: the four Greeks live
on genuinely different scales/units (delta is roughly O(shares), gamma is
small, vega/theta are dollar-per-unit-move-ish), so a shared y-axis would be
misleading -- same "different scale -> small multiples, never dual-axis"
rule `equity_curve.py`'s equity/drawdown split already follows, applied here
because it's four series instead of two.

Availability detection (`is_greek_exposure_available`) is kept even though
issue 04 is now merged -- a defensive guard, not the primary code path, in
case this module is ever checked out against an older commit or a stripped
build. `unavailable_placeholder_chart` is kept for the same reason and is
what the render script falls back to if `is_greek_exposure_available()` is
ever `False` again.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from testing.viz.theme import Mode, apply_layout, category_color, get_theme

_UNAVAILABLE_MESSAGE = (
    "Greek exposure chart unavailable: testing/metrics/options.py (issue 04) "
    "is not present in this worktree yet."
)


def is_greek_exposure_available() -> bool:
    """Detects whether issue 04's options-metrics module exists in this
    checkout, without importing it eagerly (importlib.util.find_spec doesn't
    execute the module, so this is safe to call even mid-build elsewhere).

    `find_spec` raises `ModuleNotFoundError` (rather than returning `None`)
    when an *intermediate* package in the dotted path doesn't exist at all --
    which is exactly this worktree's current state (`testing/metrics/` has no
    `__init__.py` yet, issue 04 hasn't landed) -- so that case is caught
    alongside the more obvious "package exists, submodule doesn't" case.
    """
    try:
        return importlib.util.find_spec("testing.metrics.options") is not None
    except ModuleNotFoundError:
        return False


_GREEK_TITLES = {
    "delta": "Delta (shares-equivalent)",
    "gamma": "Gamma",
    "vega": "Vega ($ / 1 vol pt)",
    "theta": "Theta ($ / day)",
}


def greek_exposure_chart(
    exposure: pd.DataFrame | None = None,
    *,
    mode: Mode = "light",
    title: str | None = "Portfolio Greek exposure over time",
) -> go.Figure | None:
    """`exposure` is `testing.metrics.options.portfolio_greek_exposure`'s
    output: a `pd.DataFrame` indexed by date with `delta`/`gamma`/`vega`/
    `theta` columns. Returns `None` (skip cleanly, same convention as
    `monte_carlo.mcpt_null_distribution_chart`) if issue 04's module isn't
    importable, or if `exposure` is `None`/empty -- never renders a
    misleading empty chart.
    """
    if not is_greek_exposure_available() or exposure is None or exposure.empty:
        return None

    from testing.metrics.options import GREEK_COLUMNS

    theme = get_theme(mode)
    exposure = exposure.sort_index()

    fig = make_subplots(
        rows=len(GREEK_COLUMNS),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=tuple(_GREEK_TITLES[g] for g in GREEK_COLUMNS),
    )

    for slot, greek in enumerate(GREEK_COLUMNS):
        color = category_color(theme, slot)
        fig.add_trace(
            go.Scatter(
                x=exposure.index,
                y=exposure[greek].values,
                mode="lines",
                line={"color": color, "width": 2},
                name=greek,
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>{greek}: %{{y:,.4f}}<extra></extra>",
                showlegend=False,
            ),
            row=slot + 1,
            col=1,
        )

    apply_layout(fig, theme, title=title, show_legend=False)
    fig.update_layout(height=180 * len(GREEK_COLUMNS))
    return fig


def unavailable_placeholder_chart(mode: Mode = "light") -> go.Figure:
    """A small, honest "not available" figure -- used by the render script in
    place of the real chart so a missing issue-04 dependency is visible in
    the output directory rather than a silently absent file.
    """
    theme = get_theme(mode)
    fig = go.Figure()
    fig.add_annotation(
        text=_UNAVAILABLE_MESSAGE,
        showarrow=False,
        font={"color": theme.text_secondary, "size": 13},
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    apply_layout(fig, theme, title="Greek exposure over time", show_legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

"""Monte Carlo null-distribution histogram with the observed Sharpe marked.

**Judgment call, documented per this worktree's instructions:** issue 03
(`testing/monte_carlo/mcpt.py`/`pbo.py`, an asset-class-agnostic dedicated
module) is not ported into this worktree yet. But issue 02's port already
shipped a real, validated sign-permutation MCPT implementation --
`testing.engine.strategy_comparison.evaluate_strategy_pnl` -- reproducing
`Options_Trading_Strategy` ADR 0011's cited regression number exactly (see
this repo's `AGENTS.md`, "Verified against real data"). That function
already returns exactly what this chart needs (a null distribution +
observed Sharpe + p-value), so this chart calls it directly rather than
stubbing the whole feature -- the dataviz/rigor bar for this repo is "render
against real data, don't fake it," and faking a null distribution when a
real one is one function call away would be the wrong trade. Once issue 03's
dedicated module lands, whoever merges can repoint this import at the new
module -- the output shape (`{"sharpe", "mcpt": {"p_value",
"n_permutations", "null_distribution"}}`) is what this chart already expects,
so no chart-code changes should be needed.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

from testing.engine.strategy_comparison import evaluate_strategy_pnl
from testing.viz.theme import Mode, apply_layout, category_color, get_theme


def mcpt_null_distribution_chart(
    name: str,
    trade_pnl: Sequence[float],
    equity_curve: pd.Series,
    *,
    n_permutations: int = 1000,
    mode: Mode = "light",
    title: str | None = None,
) -> go.Figure | None:
    """`trade_pnl`/`equity_curve` are sized-trade dollar P&L and the matching
    cumulative equity curve (`SizingEngine`/`PortfolioAggregator`'s outputs)
    -- the same inputs `evaluate_strategy_pnl` already takes, engine-agnostic
    per that function's own docstring.

    Returns `None` on a degenerate input (no trades, or zero-variance P&L --
    the same case `evaluate_strategy_pnl` reports as `sharpe: None`) rather
    than rendering an empty/misleading chart; the caller should skip this
    chart cleanly in that case, same treatment as the issue-03/04 stubs.
    """
    evaluation = evaluate_strategy_pnl(name, trade_pnl, equity_curve, n_permutations=n_permutations)
    if evaluation["sharpe"] is None or not evaluation["mcpt"]["null_distribution"]:
        return None

    theme = get_theme(mode)
    observed = evaluation["sharpe"]
    null_distribution = evaluation["mcpt"]["null_distribution"]
    p_value = evaluation["mcpt"]["p_value"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=null_distribution,
            name="Null distribution (sign-permuted)",
            marker={"color": category_color(theme, 0)},
            opacity=0.85,
            nbinsx=40,
        )
    )
    observed_color = category_color(theme, 5)  # red -- a marked threshold, not a second series
    fig.add_vline(
        x=observed,
        line={"color": observed_color, "width": 2, "dash": "dash"},
    )
    fig.add_annotation(
        x=observed,
        y=1.0,
        yref="y domain",
        text=f"Observed Sharpe {observed:.3f}" + (f"  (p={p_value:.3f})" if p_value is not None else ""),
        showarrow=True,
        arrowhead=2,
        arrowcolor=observed_color,
        font={"color": theme.text_primary, "size": 12},
        yanchor="bottom",
    )

    apply_layout(
        fig,
        theme,
        title=title or f"{name} -- MCPT null distribution ({evaluation['mcpt']['n_permutations']} permutations)",
        show_legend=False,
    )
    fig.update_xaxes(title_text="Sharpe under random sign permutation")
    fig.update_yaxes(title_text="Count")
    return fig

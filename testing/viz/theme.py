"""Design tokens for the Testing wing's visualization layer, following the
`dataviz` skill's method (see `~/.agents/skills/dataviz` or the bundled skill
docs): color is assigned by job (categorical/sequential/status), never
eyeballed, and every hex value in this package flows through here -- no chart
module should reach for a raw hex string.

**Palette provenance.** These are the `dataviz` skill's own reference-instance
values from `references/palette.md`, copied verbatim (this repo has no brand
palette of its own to substitute). Independently re-validated for this repo,
not just trusted from the skill doc:

    node scripts/validate_palette.js \
      "#2a78d6,#1baf7a,#eda100,#008300,#4a3aa7,#e34948,#e87ba4,#eb6834" \
      --mode light
    -> ALL CHECKS PASS (worst adjacent CVD ΔE 24.2; contrast WARN on
       aqua/yellow/magenta -- mitigated everywhere in this package by never
       using color as the only channel: a legend + direct labels/annotations
       always accompany color, per the skill's "relief rule").

    node scripts/validate_palette.js \
      "#3987e5,#199e70,#c98500,#008300,#9085e9,#e66767,#d55181,#d95926" \
      --mode dark --surface "#1a1a19"
    -> ALL CHECKS PASS (CVD WARN in the 8-12 floor band -- same relief-rule
       mitigation applies).

This repo is a personal research tool (single-user, no hosted dark-mode
toggle), so `get_theme()` defaults to "light" and dark is opt-in, but both
are implemented and validated so a future notebook/dashboard embedding isn't
blocked on redoing this work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Mode = Literal["light", "dark"]

# Categorical palette -- fixed hue order, never cycled, never reordered per
# chart. Index 0 is always "series 1" regardless of how many series a given
# chart happens to show.
CATEGORICAL_LIGHT: list[str] = [
    "#2a78d6",  # 1 blue
    "#1baf7a",  # 2 aqua
    "#eda100",  # 3 yellow
    "#008300",  # 4 green
    "#4a3aa7",  # 5 violet
    "#e34948",  # 6 red
    "#e87ba4",  # 7 magenta
    "#eb6834",  # 8 orange
]
CATEGORICAL_DARK: list[str] = [
    "#3987e5",
    "#199e70",
    "#c98500",
    "#008300",
    "#9085e9",
    "#e66767",
    "#d55181",
    "#d95926",
]

# Fixed, reserved -- never reused for "series N" (color-formula.md's "status
# is fixed" rule). Always paired with an icon/label wherever used, never
# color alone.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

_CHROME_LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
}
_CHROME_DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#898781",
    "gridline": "#2c2c2a",
    "baseline": "#383835",
}


@dataclass(frozen=True)
class ChartTheme:
    mode: Mode
    surface: str
    page: str
    text_primary: str
    text_secondary: str
    text_muted: str
    gridline: str
    baseline: str
    categorical: list[str] = field(default_factory=list)


def get_theme(mode: Mode = "light") -> ChartTheme:
    chrome = _CHROME_LIGHT if mode == "light" else _CHROME_DARK
    categorical = CATEGORICAL_LIGHT if mode == "light" else CATEGORICAL_DARK
    return ChartTheme(mode=mode, categorical=list(categorical), **chrome)


def category_color(theme: ChartTheme, slot: int) -> str:
    """Fixed-order categorical color for series index `slot` (0-based).
    Never cycles silently past the 8-hue ceiling -- per the skill's
    series-count ladder, a 9th distinct series should fold into "Other" or a
    small-multiples facet rather than repeat a hue that would then be
    indistinguishable from an earlier series under CVD.
    """
    if slot >= len(theme.categorical):
        raise ValueError(
            f"category_color: slot {slot} exceeds the {len(theme.categorical)}-hue categorical "
            "ceiling -- fold extra series into 'Other' or a small-multiples facet instead of "
            "cycling a hue (see the dataviz skill's series-count ladder)."
        )
    return theme.categorical[slot]


def apply_layout(fig, theme: ChartTheme, *, title: str | None = None, show_legend: bool = True) -> None:
    """Applies the shared chrome (surface, ink, gridlines, hover style) to a
    plotly figure in place. Deliberately does not touch axis titles/ranges --
    each chart function owns its own single-axis semantics; this only
    supplies the theme-level tokens common to every chart in this package.
    """
    fig.update_layout(
        template="plotly_white" if theme.mode == "light" else "plotly_dark",
        paper_bgcolor=theme.page,
        plot_bgcolor=theme.surface,
        font={"family": 'system-ui, -apple-system, "Segoe UI", sans-serif', "color": theme.text_primary},
        title={"text": title, "font": {"color": theme.text_primary, "size": 16}} if title else None,
        showlegend=show_legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hoverlabel={"bgcolor": theme.surface, "font": {"color": theme.text_primary}},
        margin={"l": 60, "r": 30, "t": 60 if title else 30, "b": 40},
    )
    fig.update_xaxes(
        gridcolor=theme.gridline, linecolor=theme.baseline, tickfont={"color": theme.text_muted}, zeroline=False
    )
    fig.update_yaxes(
        gridcolor=theme.gridline, linecolor=theme.baseline, tickfont={"color": theme.text_muted}, zeroline=False
    )

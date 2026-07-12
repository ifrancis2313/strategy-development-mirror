import re

from testing.viz.theme import CATEGORICAL_DARK, CATEGORICAL_LIGHT, STATUS, category_color, get_theme

_HEX_RE = re.compile(r"^#[0-9a-f]{6}$")


def test_categorical_palettes_are_valid_hex_and_fixed_length() -> None:
    for palette in (CATEGORICAL_LIGHT, CATEGORICAL_DARK):
        assert len(palette) == 8
        assert all(_HEX_RE.match(c) for c in palette)


def test_status_palette_distinct_from_categorical() -> None:
    # Status colors are reserved -- never reused for "series N" (color-formula.md).
    assert set(STATUS.values()).isdisjoint(CATEGORICAL_LIGHT)
    assert set(STATUS.values()).isdisjoint(CATEGORICAL_DARK)


def test_get_theme_light_and_dark_use_distinct_surfaces() -> None:
    light = get_theme("light")
    dark = get_theme("dark")
    assert light.surface != dark.surface
    assert light.categorical == CATEGORICAL_LIGHT
    assert dark.categorical == CATEGORICAL_DARK


def test_category_color_fixed_order_never_cycles() -> None:
    theme = get_theme("light")
    assert category_color(theme, 0) == CATEGORICAL_LIGHT[0]
    assert category_color(theme, 7) == CATEGORICAL_LIGHT[7]


def test_category_color_raises_past_ceiling_instead_of_cycling() -> None:
    theme = get_theme("light")
    try:
        category_color(theme, 8)
        raise AssertionError("expected ValueError past the 8-hue ceiling")
    except ValueError:
        pass

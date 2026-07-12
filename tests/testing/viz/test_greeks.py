import pandas as pd

from testing.viz.greeks import (
    greek_exposure_chart,
    is_greek_exposure_available,
    unavailable_placeholder_chart,
)


def test_greek_exposure_available_now_that_issue_04_is_merged() -> None:
    assert is_greek_exposure_available() is True


def test_greek_exposure_chart_returns_none_when_exposure_is_none() -> None:
    assert greek_exposure_chart(None) is None


def test_greek_exposure_chart_returns_none_on_empty_exposure() -> None:
    empty = pd.DataFrame(columns=["delta", "gamma", "vega", "theta"], dtype=float)
    assert greek_exposure_chart(empty) is None


def test_greek_exposure_chart_renders_four_small_multiples() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    exposure = pd.DataFrame(
        {
            "delta": [10.0, 12.0, 8.0, 5.0, 0.0],
            "gamma": [0.1, 0.15, 0.12, 0.08, 0.0],
            "vega": [50.0, 55.0, 40.0, 20.0, 0.0],
            "theta": [-30.0, -32.0, -28.0, -15.0, 0.0],
        },
        index=dates,
    )
    fig = greek_exposure_chart(exposure)
    assert fig is not None
    # one trace per Greek, in GREEK_COLUMNS order
    assert len(fig.data) == 4
    assert [trace.name for trace in fig.data] == ["delta", "gamma", "vega", "theta"]
    assert list(fig.data[0].y) == [10.0, 12.0, 8.0, 5.0, 0.0]
    # four stacked rows, one y-axis per Greek (never a shared/dual axis)
    assert len(fig.layout.annotations) == 4  # subplot titles


def test_unavailable_placeholder_chart_renders_a_clear_message() -> None:
    fig = unavailable_placeholder_chart()
    assert len(fig.layout.annotations) == 1
    assert "issue 04" in fig.layout.annotations[0].text

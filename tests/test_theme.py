"""The two dashboard designs: complete tokens, readable text, sensible colour ramps and server environment."""

from __future__ import annotations

from pathlib import Path

import pytest

from dashboards.components import theme

ROOT = Path(__file__).resolve().parents[1]
NAMES = sorted(theme.THEMES)


def _luminance(hex_colour: str) -> float:
    """WCAG relative luminance of ``#rrggbb``."""
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize("name", NAMES)
def test_every_design_has_all_tokens_and_a_stylesheet(name):
    spec = theme.THEMES[name]
    assert set(spec["tokens"]) == set(theme.FIELD)
    assert (ROOT / "dashboards" / "assets" / spec["css"]).is_file()
    assert len(spec["seq_risk"]) == 8
    assert spec["simulator"] in {"light", "dark"}


@pytest.mark.parametrize("name", NAMES)
def test_text_and_interface_colours_are_readable(name):
    tokens = theme.THEMES[name]["tokens"]
    for surface in ("paper", "card"):
        assert _contrast(tokens["ink"], tokens[surface]) >= 7
        assert _contrast(tokens["ink-2"], tokens[surface]) >= 4.5
        assert _contrast(tokens["muted"], tokens[surface]) >= 4.5
    assert _contrast(tokens["primary"], tokens["paper"]) >= 4.5
    # the two chart categories must stand out from the chart surface (graphics need 3:1)
    assert _contrast(tokens["heat"], tokens["card"]) >= 3
    assert _contrast(tokens["rain"], tokens["card"]) >= 3


@pytest.mark.parametrize("name", NAMES)
def test_risk_ramp_is_monotonic_in_lightness(name):
    ramp = theme.THEMES[name]["seq_risk"]
    lum = [_luminance(c) for c in ramp]
    # light surface: higher risk is darker; dark surface: higher risk is brighter
    if _luminance(theme.THEMES[name]["tokens"]["paper"]) > 0.5:
        assert lum == sorted(lum, reverse=True)
    else:
        assert lum == sorted(lum)


def test_server_environment_uses_streamlit_option_names():
    env = theme.streamlit_environment("almanac")
    assert env["CR_THEME"] == "almanac"
    assert env["STREAMLIT_THEME_PRIMARY_COLOR"] == "#e9b44c"
    assert env["STREAMLIT_THEME_SIDEBAR_BACKGROUND_COLOR"] == "#12110f"
    assert theme.streamlit_environment("field") == {"CR_THEME": "field"}
    with pytest.raises(ValueError):
        theme.streamlit_environment("missing")


def test_css_variables_cover_every_token():
    block = theme.css_variables()
    for name, value in theme.TOKENS.items():
        assert f"--{name}: {value};" in block

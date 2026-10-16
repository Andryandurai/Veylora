"""The dashboard palette: readable text, sensible colour ramp and complete CSS variables."""

from __future__ import annotations

from pathlib import Path

from dashboards.components import theme

ROOT = Path(__file__).resolve().parents[1]


def _luminance(hex_colour: str) -> float:
    """WCAG relative luminance of ``#rrggbb``."""
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_stylesheet_exists():
    assert (ROOT / "dashboards" / "assets" / theme.CSS_FILE).is_file()
    assert len(theme.SEQ_RISK) == 8


def test_text_and_interface_colours_are_readable():
    tokens = theme.TOKENS
    for surface in ("paper", "card"):
        assert _contrast(tokens["ink"], tokens[surface]) >= 7
        assert _contrast(tokens["ink-2"], tokens[surface]) >= 4.5
        assert _contrast(tokens["muted"], tokens[surface]) >= 4.5
    assert _contrast(tokens["primary"], tokens["paper"]) >= 4.5
    # the two chart categories must stand out from the chart surface (graphics need 3:1)
    assert _contrast(tokens["heat"], tokens["chart"]) >= 3
    assert _contrast(tokens["rain"], tokens["chart"]) >= 3


def test_risk_ramp_gets_darker_with_higher_risk():
    lum = [_luminance(c) for c in theme.SEQ_RISK]
    assert lum == sorted(lum, reverse=True)


def test_css_variables_cover_every_token():
    block = theme.css_variables()
    for name, value in theme.TOKENS.items():
        assert f"--{name}: {value};" in block


def test_streamlit_config_uses_the_same_page_colours():
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    for token, key in (
        ("paper", "backgroundColor"),
        ("card", "secondaryBackgroundColor"),
        ("ink", "textColor"),
        ("primary", "primaryColor"),
        ("line", "borderColor"),
    ):
        assert f'{key} = "{theme.TOKENS[token]}"' in config

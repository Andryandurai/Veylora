"""Colour tokens of the dashboards: one palette shared by the stylesheet, the Altair charts and the maps.

The design is "paper and field": a warm paper page, white cards, a deep green brand colour and a terracotta accent
for heat and risk. Two-series charts use ``RAIN`` and ``HEAT`` (checked for colour-vision separation); the brand
green is used for single series and the interface, never as a second category next to terracotta.
``.streamlit/config.toml`` repeats the page colours for Streamlit's own widgets, and ``scripts/make_site.py`` reuses the
tokens for the static site, so the two stay in step.
"""

from __future__ import annotations

TOKENS = {
    "paper": "#f4f1e8",
    "card": "#ffffff",
    "chart": "#ffffff",
    "line": "#e2dccb",
    "rule": "#cfc8b3",
    "ink": "#1a2420",
    "ink-2": "#4f5a54",
    "muted": "#5f6b64",
    "primary": "#1f6f5a",
    "primary-dark": "#123d33",
    "primary-soft": "#e3eee8",
    "heat": "#d65a2b",
    "rain": "#2a86c0",
    "amber": "#9a6a12",
    "amber-soft": "#f8efd6",
    "red": "#b3402a",
    "red-soft": "#f8e4de",
}

# Single-hue ramp (light to dark) for probabilities and vulnerability on the maps.
SEQ_RISK = [
    "#fbefe0",
    "#f6d9ba",
    "#efbd91",
    "#e69d6c",
    "#d97a4a",
    "#c25a2e",
    "#9c4020",
    "#6f2c15",
]

CSS_FILE = "minimal.css"

PAPER = TOKENS["paper"]
CARD = TOKENS["card"]
CHART = TOKENS["chart"]
LINE = TOKENS["line"]
RULE = TOKENS["rule"]
INK = TOKENS["ink"]
INK_2 = TOKENS["ink-2"]
MUTED = TOKENS["muted"]
PRIMARY = TOKENS["primary"]
PRIMARY_DARK = TOKENS["primary-dark"]
PRIMARY_SOFT = TOKENS["primary-soft"]
HEAT = TOKENS["heat"]
RAIN = TOKENS["rain"]
AMBER = TOKENS["amber"]
AMBER_SOFT = TOKENS["amber-soft"]
RED = TOKENS["red"]
RED_SOFT = TOKENS["red-soft"]


def css_variables() -> str:
    """The tokens as a ``:root`` block of CSS custom properties (the stylesheet refers to them by name)."""
    body = " ".join(f"--{name}: {value};" for name, value in TOKENS.items())
    return f":root {{ {body} }}"

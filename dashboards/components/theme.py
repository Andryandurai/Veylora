"""Colour tokens of the dashboards: one palette shared by the stylesheet, the Altair charts and the maps.

The palette is "paper and field": a warm paper page, white cards, a deep green brand colour and a terracotta
accent for heat and risk. Two-series charts use ``RAIN`` and ``HEAT`` (checked for colour-vision separation);
green is used for single series and for the interface, never next to terracotta as two categories.
"""

from __future__ import annotations

PAPER = "#f4f1e8"
CARD = "#ffffff"
LINE = "#e2dccb"
INK = "#1a2420"
INK_2 = "#4f5a54"
MUTED = "#66726b"

PRIMARY = "#1f6f5a"
PRIMARY_DARK = "#123d33"
PRIMARY_SOFT = "#e3eee8"

HEAT = "#d65a2b"
RAIN = "#2a86c0"
AMBER = "#9a6a12"
AMBER_SOFT = "#f8efd6"
RED = "#b3402a"
RED_SOFT = "#f8e4de"

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

TOKENS = {
    "paper": PAPER,
    "card": CARD,
    "line": LINE,
    "ink": INK,
    "ink-2": INK_2,
    "muted": MUTED,
    "primary": PRIMARY,
    "primary-dark": PRIMARY_DARK,
    "primary-soft": PRIMARY_SOFT,
    "heat": HEAT,
    "rain": RAIN,
    "amber": AMBER,
    "amber-soft": AMBER_SOFT,
    "red": RED,
    "red-soft": RED_SOFT,
}


def css_variables() -> str:
    """The tokens as a ``:root`` block of CSS custom properties (the stylesheet refers to them by name)."""
    body = " ".join(f"--{name}: {value};" for name, value in TOKENS.items())
    return f":root {{ {body} }}"

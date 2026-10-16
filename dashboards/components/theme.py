"""Colour tokens of the dashboards: one palette shared by the stylesheet, the Altair charts and the maps.

Two designs ship so they can be compared side by side; the environment variable ``CR_THEME`` picks one when
the server starts (``python -m dashboards.serve --theme almanac``):

* ``field`` (default): warm paper page, white cards, deep green brand colour, terracotta for heat and risk.
* ``almanac``: flat, editorial, warm charcoal page; serif display type, hairline rules instead of boxes, one saffron accent.

Two-series charts use ``RAIN`` and ``HEAT`` (checked for colour-vision separation against each design's
surface); the brand green or teal is used for single series and the interface, never as a second category.
"""

from __future__ import annotations

import os
import re

FIELD = {
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
FIELD_SEQ_RISK = [
    "#fbefe0",
    "#f6d9ba",
    "#efbd91",
    "#e69d6c",
    "#d97a4a",
    "#c25a2e",
    "#9c4020",
    "#6f2c15",
]

ALMANAC = {
    "paper": "#161513",
    "card": "#1e1c19",
    "chart": "#161513",
    "line": "#34302a",
    "rule": "#5b5549",
    "ink": "#efe9dc",
    "ink-2": "#bdb5a4",
    "muted": "#968e7c",
    "primary": "#e9b44c",
    "primary-dark": "#f3cd7e",
    "primary-soft": "#2b2412",
    "heat": "#e8643a",
    "rain": "#3399dd",
    "amber": "#e9b44c",
    "amber-soft": "#2b2412",
    "red": "#ef7a6b",
    "red-soft": "#33201d",
}
# Dark-surface ramp: low values sink into the page, high values brighten toward amber.
ALMANAC_SEQ_RISK = [
    "#2b261f",
    "#463524",
    "#664426",
    "#8a5427",
    "#b0672a",
    "#d4802f",
    "#eca047",
    "#ffc56e",
]

THEMES = {
    "field": {
        "tokens": FIELD,
        "seq_risk": FIELD_SEQ_RISK,
        "css": "minimal.css",
        "simulator": "light",
        "streamlit": {},
    },
    "almanac": {
        "tokens": ALMANAC,
        "seq_risk": ALMANAC_SEQ_RISK,
        "css": "almanac.css",
        "simulator": "dark",
        # Streamlit's own widgets read these at server start (they override .streamlit/config.toml).
        "streamlit": {
            "theme.base": "dark",
            "theme.primaryColor": "#e9b44c",
            "theme.backgroundColor": "#161513",
            "theme.secondaryBackgroundColor": "#1e1c19",
            "theme.textColor": "#efe9dc",
            "theme.linkColor": "#e9b44c",
            "theme.borderColor": "#34302a",
            "theme.baseRadius": "4px",
            "theme.showSidebarBorder": "true",
            "theme.sidebar.backgroundColor": "#12110f",
            "theme.sidebar.secondaryBackgroundColor": "#1e1c19",
            "theme.sidebar.textColor": "#efe9dc",
            "theme.sidebar.primaryColor": "#e9b44c",
            "theme.sidebar.linkColor": "#e9b44c",
            "theme.sidebar.borderColor": "#2a2722",
        },
    },
}

NAME = os.environ.get("CR_THEME", "field")
if NAME not in THEMES:
    NAME = "field"
_ACTIVE = THEMES[NAME]
TOKENS: dict[str, str] = _ACTIVE["tokens"]
CSS_FILE: str = _ACTIVE["css"]
SIMULATOR_THEME: str = _ACTIVE["simulator"]
SEQ_RISK: list[str] = _ACTIVE["seq_risk"]

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


def streamlit_environment(name: str) -> dict[str, str]:
    """Environment variables that start a server in design ``name`` (theme choice plus Streamlit theme options)."""
    if name not in THEMES:
        raise ValueError(f"unknown theme {name!r}; choose one of {sorted(THEMES)}")
    env = {"CR_THEME": name}
    for option, value in THEMES[name]["streamlit"].items():
        snake = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", option.replace(".", "_"))
        env["STREAMLIT_" + snake.upper()] = value
    return env

"""Shared look of the dashboards: one stylesheet and a few helpers (minimal design, see ``assets/minimal.css``)."""

from __future__ import annotations

import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from dashboards.components import theme

CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "minimal.css"
WIDE_CSS = (
    "[data-testid='stMainBlockContainer'], .block-container { max-width: 1480px !important; }"
)


@lru_cache(maxsize=1)
def _stylesheet() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def inject_css(wide: bool = False) -> None:
    """Add the stylesheet to the page; ``wide`` lifts the width limit (maps and the 3D simulator)."""
    extra = WIDE_CSS if wide else ""
    st.markdown(
        f"<style>{theme.css_variables()}{_stylesheet()}{extra}</style>", unsafe_allow_html=True
    )


def eyebrow(text: str) -> None:
    """A small upper-case label above a title."""
    st.markdown(f'<div class="eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def brand() -> None:
    """The product name with a small mark, shown at the top of the sidebar."""
    st.markdown(
        '<div class="brand"><span class="brand-mark"></span>'
        '<span class="brand-name">ClimateResilience <b>India</b></span></div>',
        unsafe_allow_html=True,
    )


def page_header(
    title: str, intro: str | None = None, label: str | None = None, hero: bool = False
) -> None:
    """Label, title and one line of introduction: the same top of every page.

    With ``hero`` the header sits in a tinted band (used on the home page).
    """
    if hero:
        with st.container(key="hero"):
            page_header(title, intro, label)
        return
    if label:
        eyebrow(label)
    st.title(title)
    if intro:
        st.caption(intro)


def rule() -> None:
    """A hairline divider."""
    st.markdown('<hr class="hair">', unsafe_allow_html=True)


def note(text: str) -> None:
    """A quiet one-line note with an amber rule on the left (used for data caveats that must stay visible)."""
    st.markdown(f'<div class="note-warn">{text}</div>', unsafe_allow_html=True)


def tile(page: str, title: str, text: str) -> None:
    """A linked tile: bold link, one line of description, hairline border."""
    with st.container(border=True, key=f"tile_{title.lower().replace(' ', '_')}"):
        st.page_link(page, label=title)
        st.caption(text)

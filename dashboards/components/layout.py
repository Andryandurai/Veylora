"""Common page set-up: page config, stylesheet, header, data guards, synthetic-data note and the shared sidebar."""

from __future__ import annotations

import streamlit as st

from dashboards.components import assets, guards, state, ui


def start(
    title: str,
    icon: str = "",
    needs: tuple[str, ...] = ("processed data",),
    crop: bool = True,
    intro: str | None = None,
    label: str | None = None,
    wide: bool = False,
    hero: bool = False,
) -> tuple[str, str]:
    """Configure the page and return the selected ``(district_id, crop)``.

    ``st.set_page_config`` must be the first Streamlit call of a page, so every page calls this first.
    ``icon`` is accepted for compatibility with the earlier design and is not shown; ``label`` is the small
    upper-case line above the title; ``wide`` lifts the width limit of the page column; ``hero`` puts the
    header in a tinted band (home page).
    """
    page_title = ui.SITE_NAME if title == ui.SITE_NAME else f"{title} · {ui.SITE_NAME}"
    st.set_page_config(page_title=page_title, page_icon=assets.favicon(), layout="wide")
    ui.inject_css(wide=wide)
    ui.page_header(title, intro, label, hero=hero)
    guards.require(*needs)
    guards.synthetic_banner()
    return state.sidebar_selection(show_crop=crop)

"""Common page set-up: page config, title, data guards, synthetic-data banner and the shared sidebar."""

from __future__ import annotations

import streamlit as st

from dashboards.components import guards, state


def start(
    title: str,
    icon: str,
    needs: tuple[str, ...] = ("processed data",),
    crop: bool = True,
    intro: str | None = None,
) -> tuple[str, str]:
    """Configure the page and return the selected ``(district_id, crop)``.

    ``st.set_page_config`` must be the first Streamlit call of a page, so every page calls this first.
    """
    st.set_page_config(
        page_title=f"{title} · ClimateResilience India", page_icon=icon, layout="wide"
    )
    st.title(f"{icon} {title}")
    if intro:
        st.caption(intro)
    guards.require(*needs)
    guards.synthetic_banner()
    return state.sidebar_selection(show_crop=crop)

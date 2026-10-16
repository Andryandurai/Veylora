"""Sidebar with the global district / crop selection stored in ``st.session_state``."""

from __future__ import annotations

import streamlit as st

from dashboards.components import data


def sidebar_selection(show_crop: bool = True) -> tuple[str, str]:
    """Render the shared selectors and return ``(district_id, crop)``; the choice persists across pages."""
    districts = data.load_districts()
    ids = districts["district_id"].tolist()
    labels = dict(
        zip(
            districts["district_id"],
            districts["district"] + " (" + districts["state"] + ")",
            strict=True,
        )
    )
    crops = data.crops_list()
    st.session_state.setdefault(
        "district", "tamil_nadu__thanjavur" if "tamil_nadu__thanjavur" in ids else ids[0]
    )
    st.session_state.setdefault("crop", "rice" if "rice" in crops else crops[0])
    with st.sidebar:
        st.header("Selection")
        st.selectbox("District", ids, format_func=lambda d: labels[d], key="district")
        if show_crop:
            st.selectbox("Crop", crops, format_func=str.capitalize, key="crop")
        st.caption("District units are reconstructed ICRISAT parent districts, see Methodology.")
    return st.session_state["district"], st.session_state["crop"]

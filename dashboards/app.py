"""ClimateResilience India dashboards: navigation entry point (``streamlit run dashboards/app.py``)."""

from __future__ import annotations

import streamlit as st

PAGES = [
    st.Page("home.py", title="Home", icon="🏠", url_path="home", default=True),
    st.Page(
        "pages/1_Weather_Forecast.py",
        title="Weather and early warning",
        icon="🌦️",
        url_path="weather",
    ),
    st.Page("pages/2_Yield_Predictor.py", title="Yield predictor", icon="📈", url_path="yield"),
    st.Page("pages/3_Risk_Map.py", title="Risk map", icon="🗺️", url_path="risk-map"),
    st.Page("pages/4_Advisory.py", title="Farmer advisory", icon="📱", url_path="advisory"),
    st.Page("pages/5_3D_Simulator.py", title="3D simulator", icon="🧊", url_path="simulator"),
    st.Page("pages/6_Methodology.py", title="Methodology", icon="📚", url_path="methodology"),
]

st.navigation(PAGES).run()

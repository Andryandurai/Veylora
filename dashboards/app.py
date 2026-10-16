"""ClimateResilience India dashboards: navigation entry point (``streamlit run dashboards/app.py``)."""

from __future__ import annotations

import streamlit as st

PAGES = [
    st.Page("home.py", title="Home", url_path="home", default=True),
    st.Page("pages/1_Weather_Forecast.py", title="Early warning", url_path="weather"),
    st.Page("pages/2_Yield_Predictor.py", title="Yield", url_path="yield"),
    st.Page("pages/3_Risk_Map.py", title="Risk map", url_path="risk-map"),
    st.Page("pages/4_Advisory.py", title="Advisory", url_path="advisory"),
    st.Page("pages/5_3D_Simulator.py", title="Simulator", url_path="simulator"),
    st.Page("pages/6_Methodology.py", title="Methodology", url_path="methodology"),
]

st.navigation(PAGES, position="top").run()

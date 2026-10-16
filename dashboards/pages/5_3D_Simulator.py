"""3D simulator: the Three.js crop-climate scene embedded from the static folder."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboards.components import assets, data, guards, layout


def main() -> None:
    district, crop = layout.start(
        "Simulator",
        needs=("processed data", "simulator data"),
        label="Explain",
        wide=True,
        intro="A field-scale scene of the 16-week season. Inside the scene, move the rainfall, temperature and week sliders and compare irrigation strategies; the district and crop follow the sidebar where the simulator has them.",
    )
    if assets.ensure_simulator() is None:
        st.error("The simulator is not available right now. Please try again later.")
        return
    centroids, clusters = data.centroids_and_clusters()
    choice = assets.simulator_choice(district, crop, centroids, clusters)
    if choice.note:
        st.info(choice.note)
    url = assets.simulator_url(choice.district_id, choice.crop)
    st.iframe(url, height=980)
    st.markdown(f"[Open the simulator in its own tab]({url})")
    st.caption(
        "The scene shows results calculated in advance for a grid of rainfall, temperature and irrigation choices; it is an illustration, not a live crop model. "
        "The terrain is real (exaggerated so it is visible); the fields, plants and lighting are illustrative."
    )


guards.run_guarded(main)

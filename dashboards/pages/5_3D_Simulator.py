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
        st.error("The simulator file is missing: run `make export-sim embed-sim`.")
        return
    centroids, clusters = data.centroids_and_clusters()
    choice = assets.simulator_choice(district, crop, centroids, clusters)
    if choice.note:
        st.info(choice.note)
    url = assets.simulator_url(choice.district_id, choice.crop)
    st.iframe(url, height=900)
    st.markdown(f"[Open the simulator in its own tab]({url})")
    st.caption(
        "The scene interpolates a precomputed grid (7 rainfall x 6 temperature x 3 irrigation strategies per district and crop; FAO-56 water balance plus the yield response, with paddy rice as a ponded field). It does not run a crop model live. "
        "The relief is real (open terrain tiles, exaggerated); fields, plants, soil classes and the time-of-day light are schematic. "
        "Rendering speed was checked only with software rendering; verify the frame rate on your own GPU."
    )


guards.run_guarded(main)

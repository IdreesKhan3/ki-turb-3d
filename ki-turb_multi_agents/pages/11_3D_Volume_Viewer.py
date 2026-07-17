"""3D volume viewer (Streamlit). Logic in pages/VolumeViewer3D/.

Reads *.vti velocity fields; volume/slice/isosurface views for |u|, vorticity, Q/R.
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from pages.VolumeViewer3D import load_volume_data, render_main_view

st.set_page_config(page_icon="⚫")


def main():
    inject_theme_css()
    st.title("3D Volume Viewer")
    st.markdown("**Interactive 3D Volume Visualization with ParaView-like Controls**")

    # Apply agent-synced options before any widgets run (fixes sidebar showing defaults)
    if st.session_state.get("vol3d_from_agent"):
        if "file_index" in st.session_state:
            st.session_state["slider_index"] = st.session_state["file_index"]
        st.session_state["vol3d_from_agent"] = False

    state = load_volume_data()
    if state is None:
        return

    render_main_view(state)


if __name__ == "__main__":
    main()

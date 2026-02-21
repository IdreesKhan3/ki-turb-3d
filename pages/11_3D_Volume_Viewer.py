"""
3D Volume Viewer Page (Streamlit)
Interactive 3D volume visualization with ParaView-like features

Features:
- Reads *.vti velocity fields
- Field choices: |u|, ux, uy, uz, |ω|, individual vorticity components, Q_S^S, Q, R invariants
- Interactive Plotly 3D: volume rendering, orthogonal slices, clipping box, isosurface
- Fast downsampling, time series animation, export

Refactored: logic in pages/VolumeViewer3D/ (data_helpers, file_loading, plot_style, views).
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

    state = load_volume_data()
    if state is None:
        return

    render_main_view(state)


if __name__ == "__main__":
    main()

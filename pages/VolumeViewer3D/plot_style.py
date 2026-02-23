"""
3D Volume Viewer — Plot style sidebar (persistent 3D-specific settings).
"""

import streamlit as st
from typing import Dict, Any

from utils.theme_config import apply_theme_to_plot_style
from utils.plot_style import (
    default_plot_style,
    render_figure_size_ui,
    render_plot_title_ui,
)


def get_plot_style_3d() -> Dict[str, Any]:
    """
    Get 3D plot style: theme defaults + user overrides from session state.
    Does not render UI; use render_plot_style_sidebar for that.
    """
    current_theme = st.session_state.get("theme", "Light Scientific")
    ps = default_plot_style()
    ps = apply_theme_to_plot_style(ps, current_theme)
    user_settings = st.session_state.get("plot_style_3d", {})
    for key, value in user_settings.items():
        if isinstance(value, dict) and isinstance(ps.get(key), dict):
            ps[key] = ps[key].copy()
            ps[key].update(value)
        else:
            ps[key] = value
    return ps


def render_plot_style_sidebar() -> Dict[str, Any]:
    """
    Render the Plot Style expander in sidebar and return the updated plot style.
    """
    st.sidebar.markdown("---")
    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        if "plot_style_3d" not in st.session_state:
            st.session_state.plot_style_3d = {}

        current_theme = st.session_state.get("theme", "Light Scientific")
        ps = default_plot_style()
        ps = apply_theme_to_plot_style(ps, current_theme)
        user_settings = st.session_state.plot_style_3d
        for key, value in user_settings.items():
            if isinstance(value, dict) and isinstance(ps.get(key), dict):
                ps[key] = ps[key].copy()
                ps[key].update(value)
            else:
                ps[key] = value

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker(
            "Scene background", ps.get("plot_bgcolor", "#FFFFFF"), key="3d_plot_bgcolor"
        )
        ps["paper_bgcolor"] = st.color_picker(
            "Paper background", ps.get("paper_bgcolor", "#FFFFFF"), key="3d_paper_bgcolor"
        )

        st.markdown("---")
        st.markdown("**Grid**")
        ps["grid_color"] = st.color_picker(
            "Grid color", ps.get("grid_color", "#B0B0B0"), key="3d_grid_color"
        )

        st.markdown("---")
        st.markdown("**Fonts**")
        ps["title_size"] = st.slider(
            "Plot title size", 10, 32, int(ps.get("title_size", 16)), key="3d_title_size"
        )
        ps["axis_title_size"] = st.slider(
            "Axis title size", 8, 28, int(ps.get("axis_title_size", 14)), key="3d_axis_title_size"
        )

        st.markdown("---")
        render_figure_size_ui(ps, key_prefix="3d")

        st.markdown("---")
        render_plot_title_ui(ps, key_prefix="3d")

        st.session_state.plot_style_3d = ps

    return get_plot_style_3d()

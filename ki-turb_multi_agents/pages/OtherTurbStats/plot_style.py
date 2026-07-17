"""
Other Turbulence Stats — Plot styling (wraps utils.plot_style).
"""

import streamlit as st
from typing import Dict, Any

from utils.plot_style import (
    default_plot_style,
    apply_plot_style as apply_plot_style_base,
    plot_style_sidebar as shared_plot_style_sidebar,
    ensure_per_sim_defaults,
    convert_superscript,
)
from utils.theme_config import template_selector


def _get_title_dict(ps: Dict[str, Any], title_text: str):
    """Get title dict with font color for dark theme compatibility."""
    if not title_text:
        return None
    font_color = ps.get("font_color")
    if font_color is None:
        template = ps.get("template", "plotly_white")
        if "dark" in template.lower():
            font_color = "#d4d4d4"
        else:
            font_color = "#000000"
    return dict(
        text=convert_superscript(title_text),
        font=dict(
            family=ps.get("font_family", "Arial"),
            size=ps.get("title_size", 16),
            color=font_color,
        ),
    )


def apply_plot_style(fig, ps: Dict[str, Any]):
    """Apply plot style to figure with Other Turbulence Stats defaults."""
    if ps.get("show_plot_title", False) and (not ps.get("plot_title") or ps.get("plot_title") == ""):
        ps["plot_title"] = "Custom Multi-Trace Plot"
    original_plot_title = ps.get("plot_title", "")
    if not ps.get("show_plot_title", False):
        ps["plot_title"] = ""
    fig = apply_plot_style_base(fig, ps)
    ps["plot_title"] = original_plot_title
    if not ps.get("show_plot_title", False):
        fig.update_layout(title=None)
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    return fig


def plot_style_sidebar(sim_groups: Dict[str, list]):
    """Plot style sidebar using shared utilities."""
    if "plot_style" not in st.session_state:
        st.session_state.plot_style = default_plot_style()
    ps = st.session_state.plot_style
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_turb_stats", include_marker=True)

    def _reset_style():
        st.session_state.plot_style = default_plot_style()

    return shared_plot_style_sidebar(
        sim_groups=sim_groups,
        style_key="per_sim_style_turb_stats",
        key_prefix="turb_stats",
        include_marker=True,
        reset_callback=_reset_style,
        theme_selector=template_selector,
    )

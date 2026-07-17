"""
Flatness — Plot styling (fonts, grids, per-sim overrides, reference line).
"""

import streamlit as st
from pathlib import Path
from typing import Any, Dict, List

from utils.theme_config import apply_theme_to_plot_style
from utils.plot_style import (
    default_plot_style,
    apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui,
    render_figure_size_ui,
    render_axis_scale_ui,
    render_tick_format_ui,
    render_axis_borders_ui,
    render_plot_title_ui,
    render_legend_position_ui,
    _normalize_plot_name,
    render_per_sim_style_ui,
    ensure_per_sim_defaults,
    convert_superscript,
)


def _get_title_dict(ps: Dict[str, Any], title_text: str) -> Dict[str, Any] | None:
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
    """Apply plot style to figure with Flatness defaults."""
    original_plot_title = ps.get("plot_title", "")
    if not ps.get("show_plot_title", False):
        ps["plot_title"] = ""
    fig = apply_plot_style_base(fig, ps)
    ps["plot_title"] = original_plot_title
    if not ps.get("show_plot_title", False):
        fig.update_layout(title=None)
    if any(k in ps for k in ["margin_left", "margin_right", "margin_top", "margin_bottom"]):
        fig.update_layout(
        margin=dict(
            l=ps.get("margin_left", 50),
            r=max(int(ps.get("margin_right", 30)), 80),
            t=ps.get("margin_top", 30),
            b=ps.get("margin_bottom", 50),
            )
        )
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    return fig


def get_plot_style(plot_name: str) -> Dict[str, Any]:
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    default = default_plot_style()
    default.update({
        "line_width": 2.4,
        "marker_size": 7,
        "margin_left": 50,
        "margin_right": 30,
        "margin_top": 30,
        "margin_bottom": 50,
        "std_alpha": 0.18,
        "reference_color": "#000000",
        "reference_dash": "dot",
        "reference_width": 1.5,
        "per_sim_style_flatness": {},
        "x_axis_type": "log",
        "y_axis_type": "linear",
    })

    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = default.copy()
    merged = apply_theme_to_plot_style(merged, current_theme)
    theme_plot_bgcolor = merged["plot_bgcolor"]
    theme_paper_bgcolor = merged["paper_bgcolor"]

    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value

    if "plot_bgcolor" in plot_style:
        stored_bg = plot_style["plot_bgcolor"]
        if stored_bg in ["#1e1e1e", "#FFFFFF", "#F5F5F5"]:
            merged["plot_bgcolor"] = theme_plot_bgcolor
    else:
        merged["plot_bgcolor"] = theme_plot_bgcolor

    if "paper_bgcolor" in plot_style:
        stored_bg = plot_style["paper_bgcolor"]
        if stored_bg in ["#1e1e1e", "#FFFFFF", "#F5F5F5"]:
            merged["paper_bgcolor"] = theme_paper_bgcolor
    else:
        merged["paper_bgcolor"] = theme_paper_bgcolor

    if "Dark" in current_theme:
        if merged.get("reference_color") == "#000000":
            merged["reference_color"] = "#dcdcaa"
    return merged


def plot_style_sidebar(data_dir: Path, sim_groups: Dict[str, list], plot_names: List[str]):
    """Render plot style configuration in sidebar."""
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure", plot_names, key="flatness_plot_selector"
    )
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}

    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name(selected_plot)
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_flatness", include_marker=True)
    key_prefix = f"flatness_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        font_idx = fonts.index(ps.get("font_family", "Arial")) if ps.get("font_family", "Arial") in fonts else 0
        ps["font_family"] = st.selectbox("Font family", fonts, index=font_idx, key=f"{key_prefix}_font_family")
        ps["font_size"] = st.slider("Base/global font size", 8, 26, int(ps.get("font_size", 14)), key=f"{key_prefix}_font_size")
        ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)), key=f"{key_prefix}_title_size")
        ps["legend_size"] = st.slider("Legend font size", 8, 24, int(ps.get("legend_size", 12)), key=f"{key_prefix}_legend_size")
        ps["show_legend"] = st.checkbox("Show legend", bool(ps.get("show_legend", True)), help="Display legend on the plot", key=f"{key_prefix}_show_legend")
        render_legend_position_ui(ps, key_prefix)
        ps["tick_font_size"] = st.slider("Tick label font size", 6, 24, int(ps.get("tick_font_size", 12)), key=f"{key_prefix}_tick_font_size")
        ps["axis_title_size"] = st.slider("Axis title font size", 8, 28, int(ps.get("axis_title_size", 14)), key=f"{key_prefix}_axis_title_size")
        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background (inside axes)", ps.get("plot_bgcolor", "#FFFFFF"), key=f"{key_prefix}_plot_bgcolor")
        ps["paper_bgcolor"] = st.color_picker("Paper background (outside axes)", ps.get("paper_bgcolor", "#FFFFFF"), key=f"{key_prefix}_paper_bgcolor")
        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)), key=f"{key_prefix}_tick_len")
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)), key=f"{key_prefix}_tick_w")
        ps["ticks_outside"] = st.checkbox("Ticks outside", bool(ps.get("ticks_outside", True)), key=f"{key_prefix}_ticks_outside")
        st.markdown("---")
        render_axis_scale_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_tick_format_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_axis_borders_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox("Show major grid", bool(ps.get("show_grid", True)), key=f"{key_prefix}_show_grid")
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)), key=f"{key_prefix}_grid_on_x")
        with gcol2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)), key=f"{key_prefix}_grid_on_y")
        ps["grid_w"] = st.slider("Major grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)), key=f"{key_prefix}_grid_w")
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        grid_dash_idx = grid_styles.index(ps.get("grid_dash", "dot")) if ps.get("grid_dash", "dot") in grid_styles else 1
        ps["grid_dash"] = st.selectbox("Major grid type", grid_styles, index=grid_dash_idx, key=f"{key_prefix}_grid_dash")
        ps["grid_color"] = st.color_picker("Major grid color", ps.get("grid_color", "#B0B0B0"), key=f"{key_prefix}_grid_color")
        ps["grid_opacity"] = st.slider("Major grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)), key=f"{key_prefix}_grid_opacity")
        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)), key=f"{key_prefix}_show_minor_grid")
        ps["minor_grid_w"] = st.slider("Minor grid width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)), key=f"{key_prefix}_minor_grid_w")
        minor_grid_dash_idx = grid_styles.index(ps.get("minor_grid_dash", "dot")) if ps.get("minor_grid_dash", "dot") in grid_styles else 1
        ps["minor_grid_dash"] = st.selectbox("Minor grid type", grid_styles, index=minor_grid_dash_idx, key=f"{key_prefix}_minor_grid_dash")
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"), key=f"{key_prefix}_minor_grid_color")
        ps["minor_grid_opacity"] = st.slider("Minor grid opacity", 0.0, 1.0, float(ps.get("minor_grid_opacity", 0.45)), key=f"{key_prefix}_minor_grid_opacity")
        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.4)), key=f"{key_prefix}_line_width")
        ps["marker_size"] = st.slider("Global marker size", 0, 18, int(ps.get("marker_size", 7)), key=f"{key_prefix}_marker_size")
        ps["std_alpha"] = st.slider("Std band opacity", 0.05, 0.6, float(ps.get("std_alpha", 0.18)), key=f"{key_prefix}_std_alpha")
        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2", "Pastel1", "Bold", "Prism", "Custom"]
        palette_idx = palettes.index(ps.get("palette", "Plotly")) if ps.get("palette", "Plotly") in palettes else 0
        ps["palette"] = st.selectbox("Palette", palettes, index=palette_idx, key=f"{key_prefix}_palette")
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"{key_prefix}_cust_color_{i}"))
            ps["custom_colors"] = new_cols
        st.markdown("---")
        st.markdown("**Reference line (Gaussian F=3)**")
        ps["reference_color"] = st.color_picker("Reference line color", ps.get("reference_color", "#000000"), key=f"{key_prefix}_reference_color")
        ps["reference_width"] = st.slider("Reference line width", 0.5, 4.0, float(ps.get("reference_width", 1.5)), key=f"{key_prefix}_reference_width")
        ps["reference_dash"] = st.selectbox("Reference line dash", grid_styles, index=grid_styles.index(ps.get("reference_dash", "dot")) if ps.get("reference_dash", "dot") in grid_styles else 1, key=f"{key_prefix}_reference_dash")
        st.markdown("---")
        st.markdown("**Theme**")
        old_template = ps.get("template", "plotly_white")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox("Template", templates, index=templates.index(old_template) if old_template in templates else 0, key=f"{key_prefix}_template")
        if ps["template"] != old_template:
            if ps["template"] == "plotly_dark":
                ps["plot_bgcolor"] = "#1e1e1e"
                ps["paper_bgcolor"] = "#1e1e1e"
            else:
                ps["plot_bgcolor"] = "#FFFFFF"
                ps["paper_bgcolor"] = "#FFFFFF"
        st.markdown("---")
        render_plot_title_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_axis_limits_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_figure_size_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        st.markdown("**Frame/Margin Size**")
        col1, col2 = st.columns(2)
        with col1:
            ps["margin_left"] = st.number_input("Left margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_left", 50)), step=5, key=f"{key_prefix}_margin_left")
            ps["margin_top"] = st.number_input("Top margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_top", 30)), step=5, key=f"{key_prefix}_margin_top")
        with col2:
            ps["margin_right"] = st.number_input("Right margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_right", 30)), step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_bottom", 50)), step=5, key=f"{key_prefix}_margin_bottom")
        st.markdown("---")
        render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_flatness", key_prefix=f"{key_prefix}_sim", include_marker=True, show_enable_checkbox=True)
        st.markdown("---")
        if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset"):
            st.session_state.plot_styles[selected_plot] = {}
            widget_keys = [
                f"{key_prefix}_font_family", f"{key_prefix}_font_size", f"{key_prefix}_title_size",
                f"{key_prefix}_legend_size", f"{key_prefix}_show_legend", f"{key_prefix}_tick_font_size",
                f"{key_prefix}_axis_title_size", f"{key_prefix}_plot_bgcolor", f"{key_prefix}_paper_bgcolor",
                f"{key_prefix}_tick_len", f"{key_prefix}_tick_w", f"{key_prefix}_ticks_outside",
                f"{key_prefix}_x_axis_type", f"{key_prefix}_y_axis_type",
                f"{key_prefix}_x_tick_format", f"{key_prefix}_x_tick_decimals",
                f"{key_prefix}_y_tick_format", f"{key_prefix}_y_tick_decimals",
                f"{key_prefix}_show_axis_lines", f"{key_prefix}_axis_line_width",
                f"{key_prefix}_axis_line_color", f"{key_prefix}_mirror_axes",
                f"{key_prefix}_show_grid", f"{key_prefix}_grid_on_x", f"{key_prefix}_grid_on_y",
                f"{key_prefix}_grid_w", f"{key_prefix}_grid_dash", f"{key_prefix}_grid_color",
                f"{key_prefix}_grid_opacity", f"{key_prefix}_show_minor_grid", f"{key_prefix}_minor_grid_w",
                f"{key_prefix}_minor_grid_dash", f"{key_prefix}_minor_grid_color", f"{key_prefix}_minor_grid_opacity",
                f"{key_prefix}_line_width", f"{key_prefix}_marker_size", f"{key_prefix}_std_alpha",
                f"{key_prefix}_palette", f"{key_prefix}_reference_color", f"{key_prefix}_reference_width",
                f"{key_prefix}_reference_dash", f"{key_prefix}_template",
                f"{key_prefix}_show_plot_title", f"{key_prefix}_plot_title",
                f"{key_prefix}_enable_x_limits", f"{key_prefix}_x_min", f"{key_prefix}_x_max",
                f"{key_prefix}_enable_y_limits", f"{key_prefix}_y_min", f"{key_prefix}_y_max",
                f"{key_prefix}_enable_custom_size", f"{key_prefix}_figure_width", f"{key_prefix}_figure_height",
                f"{key_prefix}_margin_left", f"{key_prefix}_margin_right", f"{key_prefix}_margin_top",
                f"{key_prefix}_margin_bottom", f"{key_prefix}_enable_per_sim",
            ]
            for i in range(10):
                widget_keys.append(f"{key_prefix}_cust_color_{i}")
            if sim_groups:
                for sim_prefix in sim_groups.keys():
                    for suffix in ["over_on", "over_color", "over_width", "over_dash", "over_marker", "over_msize"]:
                        widget_keys.append(f"{key_prefix}_sim_{suffix}_{sim_prefix}")
            for k in widget_keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.toast(f"Reset style for '{selected_plot}'.")
            st.rerun()

    st.session_state.plot_styles[selected_plot] = ps

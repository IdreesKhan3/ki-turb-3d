"""
Real Isotropy — Plot styling (fonts, grids, per-curve overrides, reference line colors).
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
    _normalize_plot_name as _normalize_base,
    convert_superscript,
)


def _normalize_plot_name(plot_name: str) -> str:
    """Normalize plot name; extends utils version to handle '/'."""
    normalized = _normalize_base(plot_name)
    return normalized.replace("/", "_")


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
    """Apply plot style to figure with Real Isotropy defaults."""
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
                l=ps.get("margin_left", 60),
                r=ps.get("margin_right", 20),
                t=ps.get("margin_top", 40),
                b=ps.get("margin_bottom", 50),
            )
        )
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    return fig


def _ensure_curve_defaults(ps: Dict[str, Any], curves: List[str], plot_name: str) -> str:
    plot_key = _normalize_plot_name(plot_name)
    style_key = f"per_curve_style_{plot_key}"
    ps.setdefault(style_key, {})
    for c in curves:
        ps[style_key].setdefault(
            c,
            {"enabled": False, "color": None, "width": None, "dash": "solid", "marker": "circle", "msize": None},
        )
    return style_key


def get_plot_style(plot_name: str) -> Dict[str, Any]:
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    default = default_plot_style()
    default.update(
        {
            "enable_per_curve_style": False,
            "margin_left": 60,
            "margin_right": 20,
            "margin_top": 40,
            "margin_bottom": 50,
            "line_width": 1.6,
            "isotropic_1_3_color": "#ff0000",
            "isotropic_0_color": "#000000",
            "stationary_line_color": "#800080",
        }
    )
    if plot_name in ["Cross-correlations (D)", "Deviations (E)", "Convergence (F)"]:
        default["y_axis_type"] = "log"
    if plot_name == "Deviations (E)":
        default["y_tick_format"] = "normal"
        default["y_tick_decimals"] = 3

    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = default.copy()
    merged = apply_theme_to_plot_style(merged, current_theme)
    theme_props = {
        "plot_bgcolor": merged["plot_bgcolor"],
        "paper_bgcolor": merged["paper_bgcolor"],
        "font_color": merged.get("font_color"),
        "axis_line_color": merged.get("axis_line_color"),
        "grid_color": merged.get("grid_color"),
        "template": merged.get("template"),
    }
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    for prop_key, theme_value in theme_props.items():
        if prop_key in plot_style:
            stored_value = plot_style[prop_key]
            if prop_key in ["plot_bgcolor", "paper_bgcolor"]:
                if stored_value in ["#1e1e1e", "#FFFFFF", "#F5F5F5"]:
                    merged[prop_key] = theme_value
            elif prop_key == "font_color":
                if stored_value in [None, "#000000", "#d4d4d4", "#FFFFFF"]:
                    merged[prop_key] = theme_value
            elif prop_key == "axis_line_color":
                if stored_value in ["#000000", "#FFFFFF", "#d4d4d4"]:
                    merged[prop_key] = theme_value
            elif prop_key == "grid_color":
                if stored_value in ["#B0B0B0", "#404040", "#D0D0D0"]:
                    merged[prop_key] = theme_value
            elif prop_key == "template":
                if stored_value in ["plotly_white", "simple_white", "plotly_dark"]:
                    merged[prop_key] = theme_value
        else:
            merged[prop_key] = theme_value
    if "Dark" in current_theme:
        if merged.get("isotropic_1_3_color") == "#ff0000":
            merged["isotropic_1_3_color"] = "#f48771"
        if merged.get("isotropic_0_color") == "#000000":
            merged["isotropic_0_color"] = "#d4d4d4"
        if merged.get("stationary_line_color") == "#800080":
            merged["stationary_line_color"] = "#c586c0"
    return merged


def resolve_curve_style(curve: str, idx: int, colors: List[str], ps: Dict[str, Any], plot_name: str) -> tuple:
    """Resolve per-curve style; delegates to utils.plot_style.resolve_curve_style."""
    from utils.plot_style import resolve_curve_style as _resolve
    plot_key = _normalize_plot_name(plot_name)
    return _resolve(curve, idx, colors, ps, plot_key)


def plot_style_sidebar(data_dir: Path, curves: List[str], plot_names: List[str]):
    """Render plot style configuration in sidebar."""
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure", plot_names, key="realiso_plot_selector"
    )
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}
    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name(selected_plot)
    style_key = _ensure_curve_defaults(ps, curves, selected_plot)
    key_prefix = f"realiso_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        saved_font = ps.get("font_family", "Arial")
        font_idx = fonts.index(saved_font) if saved_font in fonts else 0
        ps["font_family"] = st.selectbox(
            "Font family", fonts, index=font_idx, key=f"{key_prefix}_font_family"
        )
        ps["font_size"] = st.slider(
            "Base font size", 8, 26, int(ps.get("font_size", 14)), key=f"{key_prefix}_font_size"
        )
        ps["title_size"] = st.slider(
            "Title size", 10, 32, int(ps.get("title_size", 16)), key=f"{key_prefix}_title_size"
        )
        ps["legend_size"] = st.slider(
            "Legend size", 8, 24, int(ps.get("legend_size", 12)), key=f"{key_prefix}_legend_size"
        )
        ps["tick_font_size"] = st.slider(
            "Tick label size", 6, 24, int(ps.get("tick_font_size", 12)),
            key=f"{key_prefix}_tick_font_size"
        )
        ps["axis_title_size"] = st.slider(
            "Axis title size", 8, 28, int(ps.get("axis_title_size", 14)),
            key=f"{key_prefix}_axis_title_size"
        )
        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker(
            "Plot background", ps.get("plot_bgcolor", "#FFFFFF"), key=f"{key_prefix}_plot_bgcolor"
        )
        ps["paper_bgcolor"] = st.color_picker(
            "Paper background", ps.get("paper_bgcolor", "#FFFFFF"), key=f"{key_prefix}_paper_bgcolor"
        )
        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)), key=f"{key_prefix}_tick_len")
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)), key=f"{key_prefix}_tick_w")
        ps["ticks_outside"] = st.checkbox(
            "Ticks outside", bool(ps.get("ticks_outside", True)), key=f"{key_prefix}_ticks_outside"
        )
        st.markdown("---")
        render_axis_scale_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_tick_format_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_axis_borders_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox(
            "Show major grid", bool(ps.get("show_grid", True)), key=f"{key_prefix}_show_grid"
        )
        c1, c2 = st.columns(2)
        with c1:
            ps["grid_on_x"] = st.checkbox(
                "Grid on X", bool(ps.get("grid_on_x", True)), key=f"{key_prefix}_grid_on_x"
            )
        with c2:
            ps["grid_on_y"] = st.checkbox(
                "Grid on Y", bool(ps.get("grid_on_y", True)), key=f"{key_prefix}_grid_on_y"
            )
        ps["grid_w"] = st.slider(
            "Grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)), key=f"{key_prefix}_grid_w"
        )
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        ps["grid_dash"] = st.selectbox(
            "Grid type",
            grid_styles,
            index=grid_styles.index(ps.get("grid_dash", "dot")),
            key=f"{key_prefix}_grid_dash",
        )
        ps["grid_color"] = st.color_picker(
            "Grid color", ps.get("grid_color", "#B0B0B0"), key=f"{key_prefix}_grid_color"
        )
        ps["grid_opacity"] = st.slider(
            "Grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)), key=f"{key_prefix}_grid_opacity"
        )
        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox(
            "Show minor grid", bool(ps.get("show_minor_grid", False)), key=f"{key_prefix}_show_minor_grid"
        )
        ps["minor_grid_w"] = st.slider(
            "Minor width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)), key=f"{key_prefix}_minor_grid_w"
        )
        ps["minor_grid_dash"] = st.selectbox(
            "Minor type",
            grid_styles,
            index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
            key=f"{key_prefix}_minor_grid_dash",
        )
        ps["minor_grid_color"] = st.color_picker(
            "Minor color", ps.get("minor_grid_color", "#D0D0D0"), key=f"{key_prefix}_minor_grid_color"
        )
        ps["minor_grid_opacity"] = st.slider(
            "Minor opacity", 0.0, 1.0, float(ps.get("minor_grid_opacity", 0.4)),
            key=f"{key_prefix}_minor_grid_opacity"
        )
        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider(
            "Global line width", 0.5, 7.0, float(ps.get("line_width", 1.6)), key=f"{key_prefix}_line_width"
        )
        ps["marker_size"] = st.slider(
            "Global marker size", 0, 14, int(ps.get("marker_size", 6)), key=f"{key_prefix}_marker_size"
        )
        ps["raw_data_opacity"] = st.slider(
            "Raw data opacity", 0.0, 1.0, float(ps.get("raw_data_opacity", 0.5)),
            key=f"{key_prefix}_raw_data_opacity",
            help="Opacity for raw fluctuation lines and markers",
        )
        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2", "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox(
            "Palette", palettes, index=palettes.index(ps.get("palette", "Plotly")), key=f"{key_prefix}_palette"
        )
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#ff7f0e", "#2ca02c"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"{key_prefix}_cust_color_{i}"))
            ps["custom_colors"] = new_cols
        st.markdown("---")
        st.markdown("**Theme**")
        old_template = ps.get("template", "plotly_white")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox(
            "Template", templates, index=templates.index(old_template), key=f"{key_prefix}_template"
        )
        if ps["template"] != old_template:
            if ps["template"] == "plotly_dark":
                ps["plot_bgcolor"] = "#1e1e1e"
                ps["paper_bgcolor"] = "#1e1e1e"
            else:
                ps["plot_bgcolor"] = "#FFFFFF"
                ps["paper_bgcolor"] = "#FFFFFF"
        st.markdown("---")
        render_plot_title_ui(ps, key_prefix=key_prefix)
        if selected_plot == "Energy Fractions (A)":
            ps["isotropic_1_3_color"] = st.color_picker(
                "Isotropic (1/3) line color", ps.get("isotropic_1_3_color", "#ff0000"),
                key=f"{key_prefix}_isotropic_1_3_color"
            )
            ps["stationary_line_color"] = st.color_picker(
                "Statistical stationarity line color", ps.get("stationary_line_color", "#800080"),
                key=f"{key_prefix}_stationary_line_color"
            )
        elif selected_plot == "Diagonal b_ii (C)":
            ps["isotropic_0_color"] = st.color_picker(
                "Isotropic (0) line color", ps.get("isotropic_0_color", "#000000"),
                key=f"{key_prefix}_isotropic_0_color"
            )
        elif selected_plot == "Deviations (E)":
            ps["stationary_line_color"] = st.color_picker(
                "Statistical stationarity line color", ps.get("stationary_line_color", "#800080"),
                key=f"{key_prefix}_stationary_line_color"
            )
        st.markdown("---")
        render_axis_limits_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        render_figure_size_ui(ps, key_prefix=key_prefix)
        st.markdown("---")
        st.markdown("**Frame/Margin Size**")
        col1, col2 = st.columns(2)
        with col1:
            ps["margin_left"] = st.number_input(
                "Left margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_left", 60)),
                step=5, key=f"{key_prefix}_margin_left"
            )
            ps["margin_top"] = st.number_input(
                "Top margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_top", 40)),
                step=5, key=f"{key_prefix}_margin_top"
            )
        with col2:
            ps["margin_right"] = st.number_input(
                "Right margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_right", 20)),
                step=5, key=f"{key_prefix}_margin_right"
            )
            ps["margin_bottom"] = st.number_input(
                "Bottom margin (px)", min_value=0, max_value=200, value=int(ps.get("margin_bottom", 50)),
                step=5, key=f"{key_prefix}_margin_bottom"
            )
        st.markdown("---")
        st.markdown("**Per-curve overrides (optional)**")
        ps["enable_per_curve_style"] = st.checkbox(
            "Enable per-curve overrides", bool(ps.get("enable_per_curve_style", False)),
            key=f"{key_prefix}_enable_per_curve"
        )
        if ps["enable_per_curve_style"]:
            dash_opts = ["solid", "dot", "dash", "dashdot", "longdash"]
            marker_opts = ["circle", "square", "diamond", "cross", "x", "triangle-up", "triangle-down", "star"]
            with st.container(border=True):
                for c in curves:
                    s = ps[style_key][c]
                    st.markdown(f"`{c}`")
                    o1, o2, o3, o4, o5 = st.columns([1, 1, 1, 1, 1])
                    with o1:
                        s["enabled"] = st.checkbox("Override", value=s["enabled"], key=f"{key_prefix}_over_on_{c}")
                    with o2:
                        s["color"] = st.color_picker(
                            "Color", value=s["color"] or "#000000", key=f"{key_prefix}_over_color_{c}",
                            disabled=not s["enabled"]
                        )
                    with o3:
                        s["width"] = st.slider(
                            "Width", 0.5, 8.0, float(s["width"] or ps["line_width"]),
                            key=f"{key_prefix}_over_width_{c}", disabled=not s["enabled"]
                        )
                    with o4:
                        s["dash"] = st.selectbox(
                            "Dash", dash_opts, index=dash_opts.index(s["dash"] or "solid"),
                            key=f"{key_prefix}_over_dash_{c}", disabled=not s["enabled"]
                        )
                    with o5:
                        s["marker"] = st.selectbox(
                            "Marker", marker_opts, index=marker_opts.index(s["marker"] or "circle"),
                            key=f"{key_prefix}_over_marker_{c}", disabled=not s["enabled"]
                        )
                    s["msize"] = st.slider(
                        "Marker size", 0, 18, int(s["msize"] or ps["marker_size"]),
                        key=f"{key_prefix}_over_msize_{c}", disabled=not s["enabled"]
                    )
        st.markdown("---")
        if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset"):
            st.session_state.plot_styles[selected_plot] = {}
            widget_keys = [
                f"{key_prefix}_font_family", f"{key_prefix}_font_size", f"{key_prefix}_title_size",
                f"{key_prefix}_legend_size", f"{key_prefix}_tick_font_size", f"{key_prefix}_axis_title_size",
                f"{key_prefix}_plot_bgcolor", f"{key_prefix}_paper_bgcolor",
                f"{key_prefix}_tick_len", f"{key_prefix}_tick_w", f"{key_prefix}_ticks_outside",
                f"{key_prefix}_x_axis_type", f"{key_prefix}_y_axis_type",
                f"{key_prefix}_x_tick_format", f"{key_prefix}_x_tick_decimals",
                f"{key_prefix}_y_tick_format", f"{key_prefix}_y_tick_decimals",
                f"{key_prefix}_show_axis_lines", f"{key_prefix}_axis_line_width",
                f"{key_prefix}_axis_line_color", f"{key_prefix}_mirror_axes",
                f"{key_prefix}_show_grid", f"{key_prefix}_grid_on_x", f"{key_prefix}_grid_on_y",
                f"{key_prefix}_grid_w", f"{key_prefix}_grid_dash", f"{key_prefix}_grid_color",
                f"{key_prefix}_grid_opacity",
                f"{key_prefix}_show_minor_grid", f"{key_prefix}_minor_grid_w",
                f"{key_prefix}_minor_grid_dash", f"{key_prefix}_minor_grid_color",
                f"{key_prefix}_minor_grid_opacity",
                f"{key_prefix}_line_width", f"{key_prefix}_marker_size", f"{key_prefix}_raw_data_opacity",
                f"{key_prefix}_palette", f"{key_prefix}_template",
                f"{key_prefix}_show_plot_title", f"{key_prefix}_plot_title",
                f"{key_prefix}_isotropic_1_3_color", f"{key_prefix}_isotropic_0_color",
                f"{key_prefix}_stationary_line_color",
                f"{key_prefix}_margin_left", f"{key_prefix}_margin_right",
                f"{key_prefix}_margin_top", f"{key_prefix}_margin_bottom",
                f"{key_prefix}_enable_per_curve",
            ]
            for i in range(10):
                widget_keys.append(f"{key_prefix}_cust_color_{i}")
            for c in curves:
                for suffix in ["over_on", "over_color", "over_width", "over_dash", "over_marker", "over_msize"]:
                    widget_keys.append(f"{key_prefix}_{suffix}_{c}")
            widget_keys.extend([
                f"{key_prefix}_enable_x_limits", f"{key_prefix}_x_min", f"{key_prefix}_x_max",
                f"{key_prefix}_enable_y_limits", f"{key_prefix}_y_min", f"{key_prefix}_y_max",
                f"{key_prefix}_enable_custom_size", f"{key_prefix}_figure_width", f"{key_prefix}_figure_height",
            ])
            for k in widget_keys:
                if k in st.session_state:
                    del st.session_state[k]
            st.toast(f"Reset style for '{selected_plot}'.")
            st.rerun()

    st.session_state.plot_styles[selected_plot] = ps

"""
Isotropy Validation (Real Space) Page — Streamlit

High-standard features:
- Reads real-space isotropy files:
    * eps_real_validation.csv (required)
    * reynolds_stress_validation.csv (optional)
- Computes anisotropy tensor b_ij and Pope/Lumley invariants
- Produces 6 interactive subplots like your simple script:
    (a) Energy fractions vs t/t0 + moving averages + tolerance bands
    (b) Lumley triangle (xi, eta) trajectory
    (c) b11, b22, b33 vs t/t0
    (d) |b12|, |b13|, |b23| + anisotropy index
    (e) energy-fraction deviations from isotropy
    (f) convergence (running std)
- Full user controls (in-memory session state): same system as other pages
- Research-grade export (requires kaleido)

Requires kaleido:
    pip install -U kaleido
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import hex_to_rgb
from pathlib import Path
import sys
import matplotlib


# --- Project imports ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.file_detector import detect_simulation_files
from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
from utils.report_builder import capture_button
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, _get_palette, convert_superscript
)
from utils.export_figs import export_panel
st.set_page_config(page_icon="⚫")


# ==========================================================
# Helpers
# ==========================================================
def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()

# ==========================================================
# Plot styling system (using centralized module)
# ==========================================================
def _normalize_plot_name(plot_name: str) -> str:
    """Normalize plot name to a valid key format (extends centralized version to handle '/')."""
    # Use centralized logic but also handle '/' replacement
    normalized = plot_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    return normalized.replace("/", "_")

def _get_title_dict(ps, title_text):
    """Get title dict with font color for dark theme compatibility."""
    if not title_text:
        return None
    
    # Get font color from plot style (defaults based on template)
    font_color = ps.get("font_color")
    if font_color is None:
        # Auto-detect from template if font_color not set
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
            color=font_color
        )
    )

def apply_plot_style(fig, ps):
    # Clear plot_title if show_plot_title is False to prevent centralized function from setting it
    original_plot_title = ps.get("plot_title", "")
    if not ps.get("show_plot_title", False):
        ps["plot_title"] = ""
    
    fig = apply_plot_style_base(fig, ps)
    
    # Restore original plot_title for later use
    ps["plot_title"] = original_plot_title
    
    if not ps.get("show_plot_title", False):
        fig.update_layout(title=None)
    
    if any(k in ps for k in ["margin_left", "margin_right", "margin_top", "margin_bottom"]):
        fig.update_layout(margin=dict(
            l=ps.get("margin_left", 60),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 40),
            b=ps.get("margin_bottom", 50)
        ))
    
    # Always set title with correct font color if show_plot_title is True
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    
    return fig

def _ensure_curve_defaults(ps, curves, plot_name: str):
    # Use plot-specific key for per-curve styles
    plot_key = _normalize_plot_name(plot_name)
    style_key = f"per_curve_style_{plot_key}"
    ps.setdefault(style_key, {})
    for c in curves:
        ps[style_key].setdefault(c, {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": "solid",
            "marker": "circle",
            "msize": None
        })
    return style_key

def get_plot_style(plot_name: str):
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    # Use centralized default_plot_style, then add page-specific defaults
    default = default_plot_style()
    # Add page-specific defaults (margins and per-curve style)
    default.update({
        "enable_per_curve_style": False,
        "margin_left": 60,
        "margin_right": 20,
        "margin_top": 40,
        "margin_bottom": 50,
        "line_width": 1.6,  # Reduced from 2.2 for better visibility
        "isotropic_1_3_color": "#ff0000",  # Red for light theme
        "isotropic_0_color": "#000000",  # Black for light theme
        "stationary_line_color": "#800080",  # Purple for light theme
    })
    
    # Set default y-axis type to log for plots that use log scale in original script
    # (d) Cross-correlations, (e) Deviations, (f) Convergence
    if plot_name in ["Cross-correlations (D)", "Deviations (E)", "Convergence (F)"]:
        default["y_axis_type"] = "log"
    
    # Set default tick format to "normal" (not "auto") for Deviations plot to avoid SI unit prefixes
    if plot_name == "Deviations (E)":
        default["y_tick_format"] = "normal"
        default["y_tick_decimals"] = 3  # More decimals for small deviation values
    
    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    
    # Apply theme first to get theme defaults
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = default.copy()
    merged = apply_theme_to_plot_style(merged, current_theme)
    
    # Store all theme-determined properties before applying user overrides
    theme_props = {
        "plot_bgcolor": merged["plot_bgcolor"],
        "paper_bgcolor": merged["paper_bgcolor"],
        "font_color": merged.get("font_color"),
        "axis_line_color": merged.get("axis_line_color"),
        "grid_color": merged.get("grid_color"),
        "template": merged.get("template"),
    }
    
    # Then apply user overrides (from plot_style) - this ensures user settings override theme
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    
    # Restore theme properties unless they were explicitly customized to non-default values
    # This ensures app theme changes always apply (unless user picked custom colors)
    for prop_key, theme_value in theme_props.items():
        if prop_key in plot_style:
            stored_value = plot_style[prop_key]
            # Check if stored value is a template default (not a custom color)
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
                # Always use theme template unless user explicitly changed it
                if stored_value in ["plotly_white", "simple_white", "plotly_dark"]:
                    merged[prop_key] = theme_value
        else:
            # Property not in plot_style, use theme default
            merged[prop_key] = theme_value
    
    # Update reference line colors for dark theme if they're still at light theme defaults
    if "Dark" in current_theme:
        if merged.get("isotropic_1_3_color") == "#ff0000":
            merged["isotropic_1_3_color"] = "#f48771"  # Light red/coral - visible on dark background
        if merged.get("isotropic_0_color") == "#000000":
            merged["isotropic_0_color"] = "#d4d4d4"  # Light gray - visible on dark background
        if merged.get("stationary_line_color") == "#800080":
            merged["stationary_line_color"] = "#c586c0"  # Light purple - visible on dark background
    
    return merged

def plot_style_sidebar(data_dir: Path, curves, plot_names: list):
    # Plot selector
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure",
        plot_names,
        key="realiso_plot_selector"
    )
    
    # Get or create plot-specific style
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}
    
    # Start with defaults, merge with plot-specific overrides
    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name(selected_plot)
    style_key = _ensure_curve_defaults(ps, curves, selected_plot)
    
    # Create unique key prefix for all widgets
    key_prefix = f"realiso_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        ps["font_family"] = st.selectbox("Font family", fonts, index=fonts.index(ps.get("font_family", "Arial")),
                                         key=f"{key_prefix}_font_family")
        ps["font_size"] = st.slider("Base font size", 8, 26, int(ps.get("font_size", 14)),
                                     key=f"{key_prefix}_font_size")
        ps["title_size"] = st.slider("Title size", 10, 32, int(ps.get("title_size", 16)),
                                      key=f"{key_prefix}_title_size")
        ps["legend_size"] = st.slider("Legend size", 8, 24, int(ps.get("legend_size", 12)),
                                       key=f"{key_prefix}_legend_size")
        ps["tick_font_size"] = st.slider("Tick label size", 6, 24, int(ps.get("tick_font_size", 12)),
                                          key=f"{key_prefix}_tick_font_size")
        ps["axis_title_size"] = st.slider("Axis title size", 8, 28, int(ps.get("axis_title_size", 14)),
                                           key=f"{key_prefix}_axis_title_size")

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background", ps.get("plot_bgcolor", "#FFFFFF"),
                                             key=f"{key_prefix}_plot_bgcolor")
        ps["paper_bgcolor"] = st.color_picker("Paper background", ps.get("paper_bgcolor", "#FFFFFF"),
                                               key=f"{key_prefix}_paper_bgcolor")

        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)),
                                    key=f"{key_prefix}_tick_len")
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)),
                                  key=f"{key_prefix}_tick_w")
        ps["ticks_outside"] = st.checkbox("Ticks outside", bool(ps.get("ticks_outside", True)),
                                           key=f"{key_prefix}_ticks_outside")

        st.markdown("---")
        render_axis_scale_ui(ps, key_prefix=key_prefix)

        st.markdown("---")
        render_tick_format_ui(ps, key_prefix=key_prefix)

        st.markdown("---")
        render_axis_borders_ui(ps, key_prefix=key_prefix)

        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox("Show major grid", bool(ps.get("show_grid", True)),
                                       key=f"{key_prefix}_show_grid")
        c1, c2 = st.columns(2)
        with c1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)),
                                           key=f"{key_prefix}_grid_on_x")
        with c2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)),
                                           key=f"{key_prefix}_grid_on_y")
        ps["grid_w"] = st.slider("Grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)),
                                  key=f"{key_prefix}_grid_w")
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        ps["grid_dash"] = st.selectbox("Grid type", grid_styles,
                                       index=grid_styles.index(ps.get("grid_dash", "dot")),
                                       key=f"{key_prefix}_grid_dash")
        ps["grid_color"] = st.color_picker("Grid color", ps.get("grid_color", "#B0B0B0"),
                                           key=f"{key_prefix}_grid_color")
        ps["grid_opacity"] = st.slider("Grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)),
                                        key=f"{key_prefix}_grid_opacity")

        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)),
                                             key=f"{key_prefix}_show_minor_grid")
        ps["minor_grid_w"] = st.slider("Minor width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)),
                                        key=f"{key_prefix}_minor_grid_w")
        ps["minor_grid_dash"] = st.selectbox("Minor type", grid_styles,
                                             index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
                                             key=f"{key_prefix}_minor_grid_dash")
        ps["minor_grid_color"] = st.color_picker("Minor color", ps.get("minor_grid_color", "#D0D0D0"),
                                                  key=f"{key_prefix}_minor_grid_color")
        ps["minor_grid_opacity"] = st.slider("Minor opacity", 0.0, 1.0, float(ps.get("minor_grid_opacity", 0.4)),
                                              key=f"{key_prefix}_minor_grid_opacity")

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 1.6)),
                                      key=f"{key_prefix}_line_width")
        ps["marker_size"] = st.slider("Global marker size", 0, 14, int(ps.get("marker_size", 6)),
                                       key=f"{key_prefix}_marker_size")
        ps["raw_data_opacity"] = st.slider("Raw data opacity", 0.0, 1.0, float(ps.get("raw_data_opacity", 0.5)),
                                            key=f"{key_prefix}_raw_data_opacity",
                                            help="Opacity for raw fluctuation lines and markers (0.0 = transparent, 1.0 = fully opaque)")

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox("Palette", palettes,
                                     index=palettes.index(ps.get("palette", "Plotly")),
                                     key=f"{key_prefix}_palette")
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#ff7f0e", "#2ca02c"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c,
                                                          key=f"{key_prefix}_cust_color_{i}"))
            ps["custom_colors"] = new_cols

        st.markdown("---")
        st.markdown("**Theme**")
        # Store template in plot-specific style
        old_template = ps.get("template", "plotly_white")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox("Template", templates,
                                      index=templates.index(old_template),
                                      key=f"{key_prefix}_template")
        # Auto-update backgrounds when template changes
        if ps["template"] != old_template:
            if ps["template"] == "plotly_dark":
                ps["plot_bgcolor"] = "#1e1e1e"
                ps["paper_bgcolor"] = "#1e1e1e"
            else:
                ps["plot_bgcolor"] = "#FFFFFF"
                ps["paper_bgcolor"] = "#FFFFFF"

        st.markdown("---")
        render_plot_title_ui(ps, key_prefix=key_prefix)

        # Reference line colors (shown based on selected plot)
        if selected_plot == "Energy Fractions (A)":
            ps["isotropic_1_3_color"] = st.color_picker(
                "Isotropic (1/3) line color",
                ps.get("isotropic_1_3_color", "#ff0000"),
                key=f"{key_prefix}_isotropic_1_3_color"
            )
            ps["stationary_line_color"] = st.color_picker(
                "Statistical stationarity line color",
                ps.get("stationary_line_color", "#800080"),
                key=f"{key_prefix}_stationary_line_color"
            )
        elif selected_plot == "Diagonal b_ii (C)":
            ps["isotropic_0_color"] = st.color_picker(
                "Isotropic (0) line color",
                ps.get("isotropic_0_color", "#000000"),
                key=f"{key_prefix}_isotropic_0_color"
            )
        elif selected_plot == "Deviations (E)":
            ps["stationary_line_color"] = st.color_picker(
                "Statistical stationarity line color",
                ps.get("stationary_line_color", "#800080"),
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
            ps["margin_left"] = st.number_input("Left margin (px)", min_value=0, max_value=200, 
                                                value=int(ps.get("margin_left", 60)), 
                                                step=5, key=f"{key_prefix}_margin_left")
            ps["margin_top"] = st.number_input("Top margin (px)", min_value=0, max_value=200, 
                                                value=int(ps.get("margin_top", 40)), 
                                                step=5, key=f"{key_prefix}_margin_top")
        with col2:
            ps["margin_right"] = st.number_input("Right margin (px)", min_value=0, max_value=200, 
                                                  value=int(ps.get("margin_right", 20)), 
                                                  step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, 
                                                   value=int(ps.get("margin_bottom", 50)), 
                                                   step=5, key=f"{key_prefix}_margin_bottom")
        st.markdown("---")
        st.markdown("**Per-curve overrides (optional)**")
        ps["enable_per_curve_style"] = st.checkbox("Enable per-curve overrides",
                                                   bool(ps.get("enable_per_curve_style", False)),
                                                   key=f"{key_prefix}_enable_per_curve")
        if ps["enable_per_curve_style"]:
            dash_opts = ["solid", "dot", "dash", "dashdot", "longdash"]
            marker_opts = ["circle", "square", "diamond", "cross", "x",
                           "triangle-up", "triangle-down", "star"]
            with st.container(border=True):
                for c in curves:
                    s = ps[style_key][c]
                    st.markdown(f"`{c}`")
                    o1, o2, o3, o4, o5 = st.columns([1,1,1,1,1])
                    with o1:
                        s["enabled"] = st.checkbox("Override", value=s["enabled"],
                                                   key=f"{key_prefix}_over_on_{c}")
                    with o2:
                        s["color"] = st.color_picker("Color", value=s["color"] or "#000000",
                                                     key=f"{key_prefix}_over_color_{c}",
                                                     disabled=not s["enabled"])
                    with o3:
                        s["width"] = st.slider("Width", 0.5, 8.0,
                                               float(s["width"] or ps["line_width"]),
                                               key=f"{key_prefix}_over_width_{c}",
                                               disabled=not s["enabled"])
                    with o4:
                        s["dash"] = st.selectbox("Dash", dash_opts,
                                                 index=dash_opts.index(s["dash"] or "solid"),
                                                 key=f"{key_prefix}_over_dash_{c}",
                                                 disabled=not s["enabled"])
                    with o5:
                        s["marker"] = st.selectbox("Marker", marker_opts,
                                                   index=marker_opts.index(s["marker"] or "circle"),
                                                   key=f"{key_prefix}_over_marker_{c}",
                                                   disabled=not s["enabled"])
                    s["msize"] = st.slider("Marker size", 0, 18,
                                           int(s["msize"] or ps["marker_size"]),
                                           key=f"{key_prefix}_over_msize_{c}",
                                           disabled=not s["enabled"])

        st.markdown("---")
        reset_pressed = False
        if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset"):
                # 1) Reset the underlying style dict
                st.session_state.plot_styles[selected_plot] = {}
                
                # 2) Clear widget state so widgets re-read from defaults next run
                widget_keys = [
                    # Fonts
                    f"{key_prefix}_font_family",
                    f"{key_prefix}_font_size",
                    f"{key_prefix}_title_size",
                    f"{key_prefix}_legend_size",
                    f"{key_prefix}_tick_font_size",
                    f"{key_prefix}_axis_title_size",
                    # Backgrounds
                    f"{key_prefix}_plot_bgcolor",
                    f"{key_prefix}_paper_bgcolor",
                    # Ticks
                    f"{key_prefix}_tick_len",
                    f"{key_prefix}_tick_w",
                    f"{key_prefix}_ticks_outside",
                    # Axis scale
                    f"{key_prefix}_x_axis_type",
                    f"{key_prefix}_y_axis_type",
                    # Tick format
                    f"{key_prefix}_x_tick_format",
                    f"{key_prefix}_x_tick_decimals",
                    f"{key_prefix}_y_tick_format",
                    f"{key_prefix}_y_tick_decimals",
                    # Axis borders
                    f"{key_prefix}_show_axis_lines",
                    f"{key_prefix}_axis_line_width",
                    f"{key_prefix}_axis_line_color",
                    f"{key_prefix}_mirror_axes",
                    # Major grid
                    f"{key_prefix}_show_grid",
                    f"{key_prefix}_grid_on_x",
                    f"{key_prefix}_grid_on_y",
                    f"{key_prefix}_grid_w",
                    f"{key_prefix}_grid_dash",
                    f"{key_prefix}_grid_color",
                    f"{key_prefix}_grid_opacity",
                    # Minor grid
                    f"{key_prefix}_show_minor_grid",
                    f"{key_prefix}_minor_grid_w",
                    f"{key_prefix}_minor_grid_dash",
                    f"{key_prefix}_minor_grid_color",
                    f"{key_prefix}_minor_grid_opacity",
                    # Curves
                    f"{key_prefix}_line_width",
                    f"{key_prefix}_marker_size",
                    f"{key_prefix}_raw_data_opacity",
                    # Palette
                    f"{key_prefix}_palette",
                    # Theme
                    f"{key_prefix}_template",
                    # Plot title
                    f"{key_prefix}_show_plot_title",
                    f"{key_prefix}_plot_title",
                    # Reference line colors
                    f"{key_prefix}_isotropic_1_3_color",
                    f"{key_prefix}_isotropic_0_color",
                    f"{key_prefix}_stationary_line_color",
                    # Margins
                    f"{key_prefix}_margin_left",
                    f"{key_prefix}_margin_right",
                    f"{key_prefix}_margin_top",
                    f"{key_prefix}_margin_bottom",
                    # Per-curve toggle
                    f"{key_prefix}_enable_per_curve",
                ]
                
                # Custom color inputs
                for i in range(10):
                    widget_keys.append(f"{key_prefix}_cust_color_{i}")
                
                # Per-curve style widgets
                for c in curves:
                    for suffix in [
                        "over_on",
                        "over_color",
                        "over_width",
                        "over_dash",
                        "over_marker",
                        "over_msize",
                    ]:
                        widget_keys.append(f"{key_prefix}_{suffix}_{c}")
                
                # Axis limits widgets (from render_axis_limits_ui)
                widget_keys.extend([
                    f"{key_prefix}_enable_x_limits",
                    f"{key_prefix}_x_min",
                    f"{key_prefix}_x_max",
                    f"{key_prefix}_enable_y_limits",
                    f"{key_prefix}_y_min",
                    f"{key_prefix}_y_max",
                ])
                
                # Figure size widgets (from render_figure_size_ui)
                widget_keys.extend([
                    f"{key_prefix}_enable_custom_size",
                    f"{key_prefix}_figure_width",
                    f"{key_prefix}_figure_height",
                ])
                
                # Delete all widget state keys
                for k in widget_keys:
                    if k in st.session_state:
                        del st.session_state[k]
                
                st.toast(f"Reset style for '{selected_plot}'.")
                reset_pressed = True
                st.rerun()

    # Auto-save plot style changes (applies immediately) - but not if reset was pressed
    if not reset_pressed:
        st.session_state.plot_styles[selected_plot] = ps

def _resolve_curve_style(curve, idx, colors, ps, plot_name: str):
    default_color = colors[idx % len(colors)]
    default_width = ps["line_width"]
    default_dash = "solid"
    default_marker = "circle"
    default_msize = ps["marker_size"]

    if not ps.get("enable_per_curve_style", False):
        return default_color, default_width, default_dash, default_marker, default_msize

    # Use plot-specific style key
    plot_key = _normalize_plot_name(plot_name)
    style_key = f"per_curve_style_{plot_key}"
    s = ps.get(style_key, {}).get(curve, {})
    if not s.get("enabled", False):
        return default_color, default_width, default_dash, default_marker, default_msize

    return (
        s.get("color") or default_color,
        float(s.get("width") or default_width),
        s.get("dash") or default_dash,
        s.get("marker") or default_marker,
        int(s.get("msize") or default_msize),
    )


# ==========================================================
# Physics / isotropy computations
# ==========================================================
def load_turbulence_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    # robust numeric parse
    for col in ["iter", "iter_norm", "TKE_real", "u_rms_real", "eps_real",
                "frac_x", "frac_y", "frac_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fallback mapping if columns are different
    # user's eps_real_validation.csv mapping:
    # iter,iter_norm,eps_real,eps_spectral,TKE_real,u_rms_real,...,frac_x,frac_y,frac_z
    cols = df.columns.tolist()
    if "frac_x" not in cols and len(cols) >= 20:
        df["frac_x"] = pd.to_numeric(df.iloc[:, 17], errors="coerce")
        df["frac_y"] = pd.to_numeric(df.iloc[:, 18], errors="coerce")
        df["frac_z"] = pd.to_numeric(df.iloc[:, 19], errors="coerce")

    data = {
        "iter": df["iter"].to_numpy(),
        "iter_norm": df.get("iter_norm", df["iter"]).to_numpy(),
        "TKE": df.get("TKE_real", df.iloc[:, 4]).to_numpy(),
        "u_rms": df.get("u_rms_real", df.iloc[:, 5]).to_numpy(),
        "eps0": df.get("eps_real", df.iloc[:, 2]).to_numpy(),
        "frac_x": df["frac_x"].to_numpy(),
        "frac_y": df["frac_y"].to_numpy(),
        "frac_z": df["frac_z"].to_numpy(),
    }
    return data

def load_reynolds_stress(stress_path: Path, turb):
    if not stress_path.exists():
        return compute_reynolds_from_fractions(turb)

    df = pd.read_csv(stress_path)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    n = min(len(df), len(turb["iter"]))

    R11, R22, R33 = df.iloc[:n, 1], df.iloc[:n, 2], df.iloc[:n, 3]
    R12, R13, R23 = df.iloc[:n, 4], df.iloc[:n, 5], df.iloc[:n, 6]
    TKE_from_R = 0.5 * (R11 + R22 + R33)

    return {
        "R11": R11.to_numpy(),
        "R22": R22.to_numpy(),
        "R33": R33.to_numpy(),
        "R12": R12.to_numpy(),
        "R13": R13.to_numpy(),
        "R23": R23.to_numpy(),
        "TKE": TKE_from_R.to_numpy(),
    }

def compute_reynolds_from_fractions(turb):
    TKE = turb["TKE"]
    R11 = turb["frac_x"] * 2 * TKE
    R22 = turb["frac_y"] * 2 * TKE
    R33 = turb["frac_z"] * 2 * TKE
    n = len(TKE)
    return dict(R11=R11, R22=R22, R33=R33,
                R12=np.zeros(n), R13=np.zeros(n), R23=np.zeros(n),
                TKE=TKE)

def anisotropy_tensor(R):
    k = R["TKE"]
    k_safe = np.where(k > 1e-10, k, 1e-10)

    b11 = R["R11"]/(2*k_safe) - 1/3
    b22 = R["R22"]/(2*k_safe) - 1/3
    b33 = R["R33"]/(2*k_safe) - 1/3
    b12 = R["R12"]/(2*k_safe)
    b13 = R["R13"]/(2*k_safe)
    b23 = R["R23"]/(2*k_safe)

    return dict(b11=b11, b22=b22, b33=b33, b12=b12, b13=b13, b23=b23)

def invariants(b):
    II_b = -0.5 * (
        b["b11"]**2 + b["b22"]**2 + b["b33"]**2 +
        2*(b["b12"]**2 + b["b13"]**2 + b["b23"]**2)
    )
    III_b = (1/3) * (
        b["b11"]**3 + b["b22"]**3 + b["b33"]**3 +
        3*b["b11"]*(b["b12"]**2 + b["b13"]**2) +
        3*b["b22"]*(b["b12"]**2 + b["b23"]**2) +
        3*b["b33"]*(b["b13"]**2 + b["b23"]**2) +
        6*b["b12"]*b["b13"]*b["b23"]
    )
    anis_index = np.sqrt(-2*II_b)
    eta = np.sqrt(-II_b/3)
    xi = np.cbrt(III_b/2)
    return dict(II_b=II_b, III_b=III_b, anis_index=anis_index, xi=xi, eta=eta)


# ==========================================================
# Page main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("Isotropy Validation — Real Space")

    data_dir = st.session_state.get("data_directory", None)
    if not data_dir:
        st.warning("Please select a data directory from the Overview page.")
        return
    data_dir = Path(data_dir)

    # Default values (using Unicode/HTML instead of LaTeX for Streamlit compatibility)
    default_legends = {
        "Ex": "E<sub>x</sub>/E<sub>tot</sub>",
        "Ey": "E<sub>y</sub>/E<sub>tot</sub>",
        "Ez": "E<sub>z</sub>/E<sub>tot</sub>",
        "b11": "b<sub>11</sub>",
        "b22": "b<sub>22</sub>",
        "b33": "b<sub>33</sub>",
        "b12": "|b<sub>12</sub>|",
        "b13": "|b<sub>13</sub>|",
        "b23": "|b<sub>23</sub>|",
        "anis": "Anisotropy index"
    }
    default_axis_labels = {
        "time": "t/t₀",
        "energy_frac": "Energy fraction",
        "bij": "Anisotropy tensor b<sub>ij</sub>",
        "cross": "Cross-correlations / Anisotropy index",
        "dev": "Absolute deviation",
        "lumley_x": "ξ = (III<sub>b</sub>/2)<sup>1/3</sup>",
        "lumley_y": "η = (-II<sub>b</sub>/3)<sup>1/2</sup>",
    }
    
    # Initialize with defaults, then merge with any loaded data
    if "real_iso_legends" not in st.session_state:
        st.session_state.real_iso_legends = default_legends.copy()
    else:
        # Ensure all default keys exist (merge defaults with existing)
        for key, value in default_legends.items():
            if key not in st.session_state.real_iso_legends:
                st.session_state.real_iso_legends[key] = value
    
    if "axis_labels_real_iso" not in st.session_state:
        st.session_state.axis_labels_real_iso = default_axis_labels.copy()
    else:
        # Ensure all default keys exist
        for key, value in default_axis_labels.items():
            if key not in st.session_state.axis_labels_real_iso:
                st.session_state.axis_labels_real_iso[key] = value
    
    # Initialize plot_styles if not exists
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}

    # Ensure all required keys are present
    for key, value in default_legends.items():
        if key not in st.session_state.real_iso_legends:
            st.session_state.real_iso_legends[key] = value
    for key, value in default_axis_labels.items():
        if key not in st.session_state.axis_labels_real_iso:
            st.session_state.axis_labels_real_iso[key] = value

    # locate required file
    files = detect_simulation_files(str(data_dir))
    eps_file = None
    
    # First, check files detected by file_detector (spectral_turb_stats key)
    for f in files.get("spectral_turb_stats", []):
        if Path(f).name == "eps_real_validation.csv" or Path(f).name.startswith("eps_real_validation"):
            eps_file = Path(f)
            break
    
    # If not found, check for exact filename in directory
    if eps_file is None:
        exact_file = data_dir / "eps_real_validation.csv"
        if exact_file.exists():
            eps_file = exact_file
    
    # If still not found, check for any eps_real_validation*.csv file (generalized for data1, data2, etc.)
    if eps_file is None:
        import glob
        pattern = str(data_dir / "eps_real_validation*.csv")
        matches = glob.glob(pattern)
        if matches:
            eps_file = Path(matches[0])  # Use first match
    
    if eps_file is None or not eps_file.exists():
        st.error(f"eps_real_validation.csv not found in dataset folder: {data_dir}")
        st.info(f"Looking for files matching: eps_real_validation*.csv")
        st.info(f"📂 Current directory: {data_dir}")
        # Show what files are actually in the directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            st.write("Available CSV files in directory:")
            for f in csv_files:
                st.write(f"  - {f.name}")
        return

    # Find Reynolds stress file using same pattern as eps file
    stress_file = None
    
    # Extract tag from eps filename (e.g., "_data1" from "eps_real_validation_data1.csv")
    eps_name = eps_file.name
    if "_data" in eps_name:
        import re
        tag_match = re.search(r'_data\d+', eps_name)
        if tag_match:
            tag = tag_match.group(0)  # e.g., "_data1"
            # Try to find matching stress file with same tag
            stress_with_tag = data_dir / f"reynolds_stress_validation{tag}.csv"
            if stress_with_tag.exists():
                stress_file = stress_with_tag
    
    # If not found with matching tag, check for exact filename
    if stress_file is None:
        exact_stress = data_dir / "reynolds_stress_validation.csv"
        if exact_stress.exists():
            stress_file = exact_stress
    
    # If still not found, check for any reynolds_stress_validation*.csv file (same as eps pattern)
    if stress_file is None:
        import glob
        pattern = str(data_dir / "reynolds_stress_validation*.csv")
        matches = glob.glob(pattern)
        if matches:
            stress_file = Path(matches[0])  # Use first match

    turb = load_turbulence_data(eps_file)
    R = load_reynolds_stress(stress_file, turb)
    b = anisotropy_tensor(R)
    inv = invariants(b)

    t0_raw = turb["iter"][0] if turb["iter"][0] != 0 else 1.0
    # time_norm will be computed after user selects normalization option

    # Sidebar: labels/legends persistence
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Curve names")
        for k in st.session_state.real_iso_legends:
            st.session_state.real_iso_legends[k] = st.text_input(
                k, st.session_state.real_iso_legends[k], key=f"realiso_leg_{k}"
            )

        st.markdown("---")
        st.markdown("### Axis labels")
        st.caption("**Which subplot uses each label:**")
        st.caption("• time → X-axis for plots A, C, D, E, F")
        st.caption("• energy_frac → Y-axis for plot A")
        st.caption("• lumley_x → X-axis for plot B")
        st.caption("• lumley_y → Y-axis for plot B")
        st.caption("• bij → Y-axis for plot C")
        st.caption("• cross → Y-axis for plot D")
        st.caption("• dev → Y-axis for plot E")
        st.markdown("")
        for k in st.session_state.axis_labels_real_iso:
            st.session_state.axis_labels_real_iso[k] = st.text_input(
                k, st.session_state.axis_labels_real_iso[k], key=f"realiso_ax_{k}"
            )

        if st.button("♻️ Reset labels/legends"):
            st.session_state.real_iso_legends = {
                "Ex": "E<sub>x</sub>/E<sub>tot</sub>", 
                "Ey": "E<sub>y</sub>/E<sub>tot</sub>", 
                "Ez": "E<sub>z</sub>/E<sub>tot</sub>",
                "b11": "b<sub>11</sub>", 
                "b22": "b<sub>22</sub>", 
                "b33": "b<sub>33</sub>",
                "b12": "|b<sub>12</sub>|", 
                "b13": "|b<sub>13</sub>|", 
                "b23": "|b<sub>23</sub>|",
                "anis": "Anisotropy index"
            }
            st.session_state.axis_labels_real_iso = {
                "time": "t/t₀", 
                "energy_frac": "Energy fraction",
                "bij": "Anisotropy tensor b<sub>ij</sub>",
                "cross": "Cross-correlations / Anisotropy index",
                "dev": "Absolute deviation",
                "lumley_x": "ξ = (III<sub>b</sub>/2)<sup>1/3</sup>",
                "lumley_y": "η = (-II<sub>b</sub>/3)<sup>1/2</sup>",
                "convergence": "Running standard deviation",
            }
            st.toast("Reset.")
            st.rerun()

    # Sidebar: analysis controls
    st.sidebar.subheader("Analysis Controls")
    
    # Normalize X-axis option (matching other turbulence stats pages)
    normalize_x = st.sidebar.checkbox("Normalize X-axis (t/t₀)", value=True, key="real_iso_norm_x",
                                      help="Use normalized time (t/t₀) instead of raw iteration numbers")
    x_norm = st.sidebar.number_input("X normalization constant", value=float(t0_raw), min_value=1.0, 
                                     step=1000.0, disabled=not normalize_x, key="real_iso_x_norm",
                                     help="Normalization constant for X-axis (default: first iteration value)")
    
    # Compute time axis based on normalization option
    if normalize_x:
        time_norm = turb["iter"] / x_norm
    else:
        time_norm = turb["iter"]
    
    stationary_iter = st.sidebar.number_input("Stationarity iteration", value=50000.0, step=5000.0)
    stationary_t = stationary_iter / (x_norm if normalize_x else t0_raw)

    st.sidebar.markdown("**Tolerance bands**")
    tol_list_a = st.sidebar.multiselect("Subplot A (Energy fractions)", [0.005, 0.01, 0.02],
                                        default=[0.005, 0.01, 0.02], key="tol_a")
    tol_list_c = st.sidebar.multiselect("Subplot C (Diagonal b_ii)", [0.005, 0.01, 0.02],
                                        default=[0.005, 0.01, 0.02], key="tol_c")
    tol_list_d = st.sidebar.multiselect("Subplot D (Cross-correlations)", [0.001, 0.005, 0.01],
                                        default=[0.001, 0.01], key="tol_d")
    tol_list_e = st.sidebar.multiselect("Subplot E (Deviations)", [0.005, 0.01, 0.02],
                                        default=[0.01, 0.02], key="tol_e")

    # Calculate default moving average window (matching original script logic)
    min_len = len(turb["frac_x"])
    default_ma_win = max(10, min_len // 10) if min_len > 20 else 0
    ma_win = st.sidebar.slider("Moving average window (0=off)", 0, 500, default_ma_win, 5)

    # curve list for overrides
    curves = ["Ex","Ey","Ez","b11","b22","b33","b12","b13","b23","anis","devx","devy","devz","maxdev"]
    plot_names = ["Energy Fractions (A)", "Lumley Triangle (B)", "Diagonal b_ii (C)", 
                  "Cross-correlations (D)", "Deviations (E)", "Convergence (F)"]
    plot_style_sidebar(data_dir, curves, plot_names)

    # Layout - 3 tabs with vertically stacked figures
    st.markdown("### Real-space isotropy diagnostics")
    
    # Prepare data that's needed across tabs
    E_x, E_y, E_z = turb["frac_x"], turb["frac_y"], turb["frac_z"]
    
    tab1, tab2, tab3 = st.tabs(["Energy & Lumley", "Anisotropy Tensor", "Deviations & Convergence"])

    # ======================================================
    # Tab 1: Energy Fractions (A) + Lumley Triangle (B)
    # ======================================================
    with tab1:
        # (a) Temporal energy fractions
        plot_name_a = "Energy Fractions (A)"
        ps_a = get_plot_style(plot_name_a)
        
        # Use exact colors from original script (not palette) to avoid dimming
        colors_orig = {
            'primary': '#1f77b4',   # Blue
            'secondary': '#ff7f0e', # Orange
            'tertiary': '#2ca02c'   # Green
        }
        color_list = [colors_orig['primary'], colors_orig['secondary'], colors_orig['tertiary']]
        
        fig_a = go.Figure()

        # Raw data with markers (matching original script: opacity 0.4, markersize 1.5)
        # Also add lines for better visibility
        markers = ["circle", "square", "triangle-up"]
        for i, ((curve, arr), marker) in enumerate(zip([("Ex",E_x),("Ey",E_y),("Ez",E_z)], markers)):
            c = color_list[i]  # Use original script colors directly
            # Raw data: clearly visible fluctuations behind the MA curves
            rgb = hex_to_rgb(c)
            # Use configurable opacity from plot style
            raw_opacity = ps_a.get("raw_data_opacity", 0.5)
            line_color_rgba = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
            # Use plot style line width - reasonable width (0.8x) for visibility
            raw_line_width = ps_a.get("line_width", 1.6) * 0.8
            # Use plot style marker size scaled appropriately
            raw_marker_size = max(2, ps_a.get("marker_size", 6) * 0.4)
            fig_a.add_trace(go.Scatter(
                x=time_norm, y=arr, mode="lines+markers",
                line=dict(color=line_color_rgba, width=raw_line_width),
                marker=dict(symbol=marker, size=raw_marker_size, color=c, opacity=raw_opacity, line=dict(width=0)),
                name=f"{st.session_state.real_iso_legends[curve]} (raw)",
                showlegend=True,
            ))

        # Moving average (optional)
        # alpha=0.7-0.9 for moving averages, so use full color (opacity=1.0)
        if ma_win and ma_win > 1 and len(E_x) > ma_win:
            def _ma(x):
                k = np.ones(ma_win)/ma_win
                return np.convolve(x, k, mode="valid")
            t_ma = time_norm[ma_win//2: ma_win//2 + len(_ma(E_x))]

            for i, (curve, arr) in enumerate([("Ex",E_x),("Ey",E_y),("Ez",E_z)]):
                c = color_list[i]  # Use my other python script colors method directly
                # Use plot style line width for moving average lines
                ma_line_width = ps_a.get("line_width", 1.6) * 1.1  # Slightly thicker than default
                # Full color for moving average lines - explicitly set opacity=1.0
                fig_a.add_trace(go.Scatter(
                    x=t_ma, y=_ma(arr), mode="lines",
                    name=f"{st.session_state.real_iso_legends[curve]} (MA-{ma_win})",
                    line=dict(color=c, width=ma_line_width),
                    marker=dict(opacity=1.0),  # Ensure full opacity
                    opacity=1.0,  # Explicitly set trace opacity to 1.0
                ))

        # Isotropic reference line (no annotation - label in legend only)
        iso_color = ps_a.get("isotropic_1_3_color", "#ff0000")
        fig_a.add_hline(y=1/3, line_dash="dash", line_color=iso_color, line_width=1.5, 
                       opacity=0.8, annotation_text="", showlegend=False)
        # Add as trace for legend
        fig_a.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=iso_color, width=1.5, dash="dash"),
            name="Isotropic (1/3)",
            showlegend=True,
        ))
        
        # Tolerance bands (matching original: ±0.005, ±0.01, ±0.02)
        # Add as shapes first, then add legend entries
        tol_colors = ["lightcoral", "lightpink", "mistyrose"]
        tol_values_a = [0.005, 0.01, 0.02]
        for tol, color in zip(tol_values_a, tol_colors):
            if tol in tol_list_a:
                # Add tolerance band as shape (layer="below" so it's behind curves)
                fig_a.add_hrect(y0=1/3-tol, y1=1/3+tol, fillcolor=color, opacity=0.3, 
                               line_width=0, layer="below")
                # Add invisible trace for legend entry
                fig_a.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, opacity=0.3),
                    name=f"±{tol:.1%} tolerance",
                    showlegend=True,
                ))

        # Statistical stationarity line (no annotation - label in legend only)
        stat_color = ps_a.get("stationary_line_color", "#800080")
        fig_a.add_vline(x=stationary_t, line_dash="dash", line_color=stat_color, line_width=1.5, 
                       opacity=0.8, annotation_text="", showlegend=False)
        # Add as trace for legend
        fig_a.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=stat_color, width=1.5, dash="dash"),
            name="Statistical stationarity",
            showlegend=True,
        ))

        layout_kwargs_a = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["energy_frac"],
            height=420,
        )
        layout_kwargs_a = apply_axis_limits(layout_kwargs_a, ps_a)
        layout_kwargs_a = apply_figure_size(layout_kwargs_a, ps_a)
        fig_a.update_layout(**layout_kwargs_a)
        fig_a = apply_plot_style(fig_a, ps_a)
        
        # Re-apply colors after plot style to prevent dimming
        # Update moving average traces to ensure full color and use plot style line width
        ma_line_width = ps_a.get("line_width", 2.2) * 1.1  # Slightly thicker than default
        raw_opacity = ps_a.get("raw_data_opacity", 0.5)  # Get opacity from plot style
        raw_marker_size = max(2, ps_a.get("marker_size", 6) * 0.4)  # Scaled marker size for raw data
        for trace in fig_a.data:
            if trace.name and "(MA-" in trace.name:
                # Restore original colors and full opacity
                if "Ex" in trace.name or "E<sub>x</sub>" in trace.name:
                    trace.line.color = colors_orig['primary']
                elif "Ey" in trace.name or "E<sub>y</sub>" in trace.name:
                    trace.line.color = colors_orig['secondary']
                elif "Ez" in trace.name or "E<sub>z</sub>" in trace.name:
                    trace.line.color = colors_orig['tertiary']
                trace.line.width = ma_line_width  # Use plot style line width
                trace.opacity = 1.0
            elif trace.name and "(raw)" in trace.name:
                # Update opacity and marker size based on plot style settings
                if "Ex" in trace.name or "E<sub>x</sub>" in trace.name:
                    rgb = hex_to_rgb(colors_orig['primary'])
                    trace.line.color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
                    trace.marker.color = colors_orig['primary']
                elif "Ey" in trace.name or "E<sub>y</sub>" in trace.name:
                    rgb = hex_to_rgb(colors_orig['secondary'])
                    trace.line.color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
                    trace.marker.color = colors_orig['secondary']
                elif "Ez" in trace.name or "E<sub>z</sub>" in trace.name:
                    rgb = hex_to_rgb(colors_orig['tertiary'])
                    trace.line.color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {raw_opacity})"
                    trace.marker.color = colors_orig['tertiary']
                if hasattr(trace, 'marker') and trace.marker:
                    trace.marker.size = raw_marker_size  # Use plot style marker size
                    trace.marker.opacity = raw_opacity  # Use plot style opacity
                trace.opacity = 1.0  # Trace-level opacity = 1.0, marker/line opacity controlled separately
        
        st.plotly_chart(fig_a, width='stretch')
        capture_button(fig_a, title="Real-Space Isotropy Analysis (Part A)", source_page="Real Isotropy")

        export_panel(fig_a, data_dir, "real_iso_energy_fractions")

        # (b) Lumley triangle
        plot_name_b = "Lumley Triangle (B)"
        ps_b = get_plot_style(plot_name_b)
        
        fig_b = go.Figure()
        xi, eta = inv["xi"], inv["eta"]

        # Realizability boundaries (matching original code exactly)
        xi_vals = np.linspace(-1/6, 1/3, 300)
        eta_two_comp = np.sqrt(1/27 + 2*xi_vals**3)
        # Axisymmetric boundaries
        eta_axi_exp = -xi_vals[xi_vals <= 0]  # Expansion (ξ ≤ 0): η = -ξ
        eta_axi_con = xi_vals[xi_vals >= 0]    # Contraction (ξ ≥ 0): η = ξ
        
        # Theme-aware colors for boundary lines
        current_theme = st.session_state.get("theme", "Light Scientific")
        is_dark = "Dark" in current_theme
        boundary_color = "#d4d4d4" if is_dark else "black"
        
        # Plot boundaries (matching original: 3 boundary lines)
        fig_b.add_trace(go.Scatter(
            x=xi_vals[xi_vals <= 0], y=eta_axi_exp, mode="lines",
            line=dict(color=boundary_color, width=1.5),
            name="Axisymmetric expansion",
            showlegend=True
        ))
        fig_b.add_trace(go.Scatter(
            x=xi_vals[xi_vals >= 0], y=eta_axi_con, mode="lines",
            line=dict(color=boundary_color, width=1.5),
            name="Axisymmetric contraction",
            showlegend=True
        ))
        fig_b.add_trace(go.Scatter(
            x=xi_vals, y=eta_two_comp, mode="lines",
            line=dict(color="red", width=1.5),
            name="Two-component limit",
            showlegend=True
        ))
        
        # Fill realizability region (matching original)
        eta_lower = np.where(xi_vals < 0, -xi_vals, xi_vals)
        # Add lower boundary as invisible trace for fill
        fig_b.add_trace(go.Scatter(
            x=xi_vals, y=eta_lower, mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))
        # Add filled area (fills between lower boundary and two-component limit)
        # Note: Original code (I mean my auxiliary script which uses matplotlib) uses fill_between without label, so not in legend
        # Theme-aware fill color for dark theme
        if is_dark:
            # Use transparent dark gray for dark theme
            fill_color = "rgba(62, 62, 66, 0.3)"  # #3e3e42 with 30% opacity
        else:
            # Use light gray for light theme
            fill_color = "rgba(211, 211, 211, 0.3)"  # lightgray with 30% opacity
        
        fig_b.add_trace(go.Scatter(
            x=xi_vals, y=eta_two_comp, mode="lines",
            fill="tonexty", fillcolor=fill_color,
            line=dict(width=0),
            showlegend=False,  # Not in legend (matching original fill_between behavior)
            hoverinfo="skip"
        ))
        
        # Plot DNS trajectory with time-coloring (matching original: wire-like appearance)
        # Original plots line segments between consecutive points, each colored by time
        # This creates a "wire" effect where the trajectory follows different paths
        n = len(xi)
        viridis = matplotlib.colormaps.get_cmap('viridis')
        # Plot line segments between consecutive points (wire-like appearance)
        # Each segment is colored according to time using viridis colormap (alpha=0.8 matching original)
        for i in range(1, n):
            # Color each segment by time (viridis colormap)
            color_val = i / n
            rgba = viridis(color_val)
            color_rgb = f"rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, 0.8)"
            fig_b.add_trace(go.Scatter(
                x=xi[i-1:i+1], y=eta[i-1:i+1], mode="lines",
                line=dict(color=color_rgb, width=1.5),
                showlegend=False,
                hoverinfo="skip"
            ))
        
        # Add scatter points for visibility with time-coloring (matching original)
        # Original: s=8, edgecolor='k', linewidth=0.6, alpha=0.9
        # Note: Matplotlib s=8 appears smaller than Plotly size=8, so using smaller size
        fig_b.add_trace(go.Scatter(
            x=xi, y=eta, mode="markers",
            marker=dict(
                size=3,  # Reduced to match matplotlib s=8 visual appearance
                color=np.linspace(0, 1, len(xi)),
                colorscale="Viridis",
                line=dict(width=0.5, color="black"),  # Also reduced linewidth slightly
                opacity=0.9
            ),
            name="DNS trajectory",
            showlegend=True
        ))
        
        # Mark start and end points (matching original)
        fig_b.add_trace(go.Scatter(
            x=[xi[0]], y=[eta[0]], mode="markers",
            marker=dict(size=12, color="red", symbol="circle", 
                       line=dict(width=2, color="black")),
            name="Start",
            showlegend=True
        ))
        fig_b.add_trace(go.Scatter(
            x=[xi[-1]], y=[eta[-1]], mode="markers",
            marker=dict(size=12, color="green", symbol="circle",
                       line=dict(width=2, color="black")),
            name="End",
            showlegend=True
        ))
        
        # Mark special points (matching original: Table 11.1)
        fig_b.add_trace(go.Scatter(
            x=[0], y=[0], mode="markers",
            marker=dict(size=12, color="yellow", symbol="star",
                       line=dict(width=1.5, color="black")),
            name="Isotropic",
            showlegend=True
        ))
        fig_b.add_trace(go.Scatter(
            x=[-1/6], y=[1/6], mode="markers",
            marker=dict(size=10, color="magenta", symbol="circle",
                       line=dict(width=1.5, color="black")),
            name="2-component axisym",
            showlegend=True
        ))
        fig_b.add_trace(go.Scatter(
            x=[1/3], y=[1/3], mode="markers",
            marker=dict(size=10, color="blue", symbol="circle",
                       line=dict(width=1.5, color="black")),
            name="1-component",
            showlegend=True
        ))

        layout_kwargs_b = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["lumley_x"],
            yaxis_title=st.session_state.axis_labels_real_iso["lumley_y"],
            height=420,
            showlegend=True,  # Show all 9 legends (matching original)
        )
        layout_kwargs_b = apply_axis_limits(layout_kwargs_b, ps_b)
        layout_kwargs_b = apply_figure_size(layout_kwargs_b, ps_b)
        fig_b.update_layout(**layout_kwargs_b)
        fig_b = apply_plot_style(fig_b, ps_b)
        st.plotly_chart(fig_b, width='stretch')
        capture_button(fig_b, title="Real-Space Isotropy Analysis (Part B)", source_page="Real Isotropy")
        export_panel(fig_b, data_dir, "real_iso_lumley_triangle")

    # ======================================================
    # Tab 2: Diagonal b_ii (C) + Cross-correlations (D)
    # ======================================================
    with tab2:
        # (c) Diagonal b_ii
        plot_name_c = "Diagonal b_ii (C)"
        ps_c = get_plot_style(plot_name_c)
        colors_c = _get_palette(ps_c)
        
        fig_c = go.Figure()
        for i, curve in enumerate(["b11","b22","b33"]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_c, ps_c, plot_name_c)
            fig_c.add_trace(go.Scatter(
                x=time_norm, y=b[curve], mode="lines",
                name=st.session_state.real_iso_legends[curve],
                line=dict(color=c, width=lw, dash=dash),
            ))
        # Isotropic reference line (no annotation - label in legend only)
        iso_0_color = ps_c.get("isotropic_0_color", "#000000")
        fig_c.add_hline(y=0, line_dash="dash", line_color=iso_0_color, line_width=1.5,
                       annotation_text="", showlegend=False)
        # Add as trace for legend
        fig_c.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=iso_0_color, width=1.5, dash="dash"),
            name="Isotropic value (0)",
            showlegend=True,
        ))
        
        # Tolerance bands (using sidebar tol_list_c)
        tol_colors_c = ["lightcoral", "lightpink", "mistyrose"]
        tol_values_c = [0.005, 0.01, 0.02]
        for tol, color in zip(tol_values_c, tol_colors_c):
            if tol in tol_list_c:
                # Add tolerance band as shape (layer="below" so it's behind curves)
                fig_c.add_hrect(y0=-tol, y1=tol, fillcolor=color, opacity=0.3, 
                               line_width=0, layer="below")
                # Add invisible trace for legend entry
                fig_c.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, opacity=0.3),
                    name=f"±{tol:.1%} tolerance",
                    showlegend=True,
                ))
        layout_kwargs_c = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["bij"],
            height=360,
        )
        layout_kwargs_c = apply_axis_limits(layout_kwargs_c, ps_c)
        layout_kwargs_c = apply_figure_size(layout_kwargs_c, ps_c)
        fig_c.update_layout(**layout_kwargs_c)
        fig_c = apply_plot_style(fig_c, ps_c)
        st.plotly_chart(fig_c, width='stretch')
        export_panel(fig_c, data_dir, "real_iso_bii_diag")

        # (d) Cross-correlations
        plot_name_d = "Cross-correlations (D)"
        ps_d = get_plot_style(plot_name_d)
        colors_d = _get_palette(ps_d)
        
        fig_d = go.Figure()
        for i, curve in enumerate(["b12","b13","b23"]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_d, ps_d, plot_name_d)
            fig_d.add_trace(go.Scatter(
                x=time_norm, y=np.abs(b[curve]), mode="lines",
                name=st.session_state.real_iso_legends[curve],
                line=dict(color=c, width=lw, dash=dash),
            ))
        c, lw, dash, mk, ms = _resolve_curve_style("anis", 3, colors_d, ps_d, plot_name_d)
        fig_d.add_trace(go.Scatter(
            x=time_norm, y=inv["anis_index"], mode="lines",
            name=st.session_state.real_iso_legends["anis"],
            line=dict(color="black", width=2.2),
        ))
        # Tolerance lines (using sidebar tol_list_d)
        tol_colors_d = ["lightcoral", "lightpink", "mistyrose"]
        tol_values_d = [0.001, 0.005, 0.01]
        for tol, color in zip(tol_values_d, tol_colors_d):
            if tol in tol_list_d:
                # Add tolerance line as shape
                fig_d.add_hline(y=tol, line_dash="dot", line_color=color, line_width=1.5,
                               annotation_text="", showlegend=False)
                # Add invisible trace for legend entry
                fig_d.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    name=f"{tol:.1%} tolerance",
                    showlegend=True,
                ))
        layout_kwargs_d = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["cross"],
            height=360,
        )
        layout_kwargs_d = apply_axis_limits(layout_kwargs_d, ps_d)
        layout_kwargs_d = apply_figure_size(layout_kwargs_d, ps_d)
        fig_d.update_layout(**layout_kwargs_d)
        fig_d = apply_plot_style(fig_d, ps_d)
        st.plotly_chart(fig_d, width='stretch')
        export_panel(fig_d, data_dir, "real_iso_cross_corr")

    # ======================================================
    # Tab 3: Deviations (E) + Convergence (F)
    # ======================================================
    with tab3:
        # (e) Deviations
        plot_name_e = "Deviations (E)"
        ps_e = get_plot_style(plot_name_e)
        colors_e = _get_palette(ps_e)
        
        fig_e = go.Figure()
        devx = np.abs(E_x - 1/3)
        devy = np.abs(E_y - 1/3)
        devz = np.abs(E_z - 1/3)
        maxdev = np.maximum(np.maximum(devx, devy), devz)

        for i,(curve,arr) in enumerate([("devx",devx),("devy",devy),("devz",devz)]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_e, ps_e, plot_name_e)
            fig_e.add_trace(go.Scatter(
                x=time_norm, y=arr, mode="lines",
                name=curve,
                line=dict(color=c, width=lw, dash=dash)
            ))

        c, lw, dash, mk, ms = _resolve_curve_style("maxdev", 3, colors_e, ps_e, plot_name_e)
        fig_e.add_trace(go.Scatter(
            x=time_norm, y=maxdev, mode="lines",
            name="Max deviation",
            line=dict(color="black", width=1.5)
        ))

        # Tolerance lines (using sidebar tol_list_e)
        tol_colors_e = ["lightcoral", "lightpink", "mistyrose"]
        tol_values_e = [0.005, 0.01, 0.02]
        for tol, color in zip(tol_values_e, tol_colors_e):
            if tol in tol_list_e:
                # Add tolerance line as shape
                fig_e.add_hline(y=tol, line_dash="dot", line_color=color, line_width=1.5,
                               annotation_text="", showlegend=False)
                # Add invisible trace for legend entry
                fig_e.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode="lines",
                    line=dict(color=color, width=1.5, dash="dot"),
                    name=f"{tol:.1%} tolerance",
                    showlegend=True,
                ))
        
        # Statistical stationarity line (no annotation - label in legend only)
        stat_color_e = ps_e.get("stationary_line_color", "#800080")
        fig_e.add_vline(x=stationary_t, line_dash="dash", line_color=stat_color_e, line_width=1.5,
                       annotation_text="", showlegend=False)
        # Add as trace for legend
        fig_e.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="lines",
            line=dict(color=stat_color_e, width=1.5, dash="dash"),
            name="Statistical stationarity",
            showlegend=True,
        ))

        layout_kwargs_e = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["dev"],
            height=360,
        )
        layout_kwargs_e = apply_axis_limits(layout_kwargs_e, ps_e)
        layout_kwargs_e = apply_figure_size(layout_kwargs_e, ps_e)
        fig_e.update_layout(**layout_kwargs_e)
        fig_e = apply_plot_style(fig_e, ps_e)
        st.plotly_chart(fig_e, width='stretch')
        export_panel(fig_e, data_dir, "real_iso_deviation")

        # (f) Convergence - matching original code exactly
        plot_name_f = "Convergence (F)"
        ps_f = get_plot_style(plot_name_f)
        
        fig_f = go.Figure()
        min_len = len(E_x)
        if min_len > 20:
            # Original code: conv_windows = [max(10, min_len // 10), max(20, min_len // 5)]
            conv_windows = [max(10, min_len // 10), max(20, min_len // 5)]
            # Use matplotlib default color cycle: first curve blue, second orange
            colors_conv = ['#1f77b4', '#ff7f0e']  # Blue, Orange (matplotlib default cycle)
            
            for idx, window in enumerate(conv_windows):
                if window < min_len:
                    # Calculate running standard deviation (matching original exactly)
                    running_stds = []
                    for i in range(window, min_len + 1):
                        std_x = np.std(E_x[i-window:i])
                        std_y = np.std(E_y[i-window:i])
                        std_z = np.std(E_z[i-window:i])
                        avg_std = (std_x + std_y + std_z) / 3
                        running_stds.append(avg_std)
                    
                    conv_time = time_norm[window-1:window-1+len(running_stds)]
                    # Original: ax4.semilogy(conv_time, running_stds, '-', linewidth=2, label=f'Running σ (window={window})')
                    # Matplotlib uses default color cycle, so each curve gets different color
                    fig_f.add_trace(go.Scatter(
                        x=conv_time, y=running_stds, mode="lines",
                        name=f"Running std (window={window})",
                        line=dict(color=colors_conv[idx % len(colors_conv)], width=1.5)
                    ))

        layout_kwargs_f = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso.get("convergence", "Running standard deviation"),
            height=360,
        )
        layout_kwargs_f = apply_axis_limits(layout_kwargs_f, ps_f)
        layout_kwargs_f = apply_figure_size(layout_kwargs_f, ps_f)
        fig_f.update_layout(**layout_kwargs_f)
        fig_f = apply_plot_style(fig_f, ps_f)
        st.plotly_chart(fig_f, width='stretch')
        export_panel(fig_f, data_dir, "real_iso_convergence")

    # ======================================================
    # Summary table
    # ======================================================
    st.markdown("### Final isotropy summary")
    df_sum = pd.DataFrame([{
        "Final Ex": float(E_x[-1]),
        "Final Ey": float(E_y[-1]),
        "Final Ez": float(E_z[-1]),
        "Final anisotropy index": float(inv["anis_index"][-1]),
        "Mean anisotropy index": float(np.mean(inv["anis_index"])),
    }])
    st.dataframe(df_sum, width='stretch')

    st.download_button(
        "Download summary CSV",
        df_sum.to_csv(index=False).encode("utf-8"),
        file_name="real_isotropy_summary.csv",
        mime="text/csv"
    )

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**Reynolds stress tensor:**")
        st.latex(r"R_{ij} = \langle u'_i u'_j \rangle")
        st.markdown(r"""
        where $u'_i = u_i - \langle u_i \rangle$ are velocity fluctuations and $\langle \cdot \rangle$ denotes ensemble or spatial average.
        """)
        
        st.markdown("**Turbulent kinetic energy:**")
        st.latex(r"k = \frac{1}{2}\langle u'_i u'_i \rangle = \frac{1}{2}(R_{11} + R_{22} + R_{33})")
        
        st.markdown("**Energy fractions:**")
        st.latex(r"\frac{E_x}{E_{\text{tot}}} = \frac{R_{11}}{2k}, \quad \frac{E_y}{E_{\text{tot}}} = \frac{R_{22}}{2k}, \quad \frac{E_z}{E_{\text{tot}}} = \frac{R_{33}}{2k}")
        st.markdown("Isotropy implies each approaches $1/3$.")
        
        st.markdown("**Reynolds stress anisotropy tensor:**")
        st.latex(r"b_{ij} = \frac{R_{ij}}{2k} - \frac{1}{3}\delta_{ij}")
        st.markdown("**Component form:**")
        st.latex(r"""
        \begin{aligned}
        b_{ii} &= \frac{R_{ii}}{2k} - \frac{1}{3}, \quad i = 1,2,3 \\
        b_{ij} &= \frac{R_{ij}}{2k}, \quad i \neq j
        \end{aligned}
        """)
        
        st.markdown("**Invariants:**")
        st.latex(r"""
        \text{II}_b = -\frac{1}{2}\mathrm{tr}(b^2), \qquad \text{III}_b = \frac{1}{3}\mathrm{tr}(b^3)
        """)
        
        st.markdown("**Lumley coordinates:**")
        st.latex(r"\eta = \left(-\frac{\text{II}_b}{3}\right)^{1/2}, \quad \xi = \left(\frac{\text{III}_b}{2}\right)^{1/3}")
        
        st.markdown("**Anisotropy index:**")
        st.latex(r"A = \sqrt{-2 \text{II}_b}")
        
        st.divider()
        st.markdown("**Reference:** [Pope (2001)](/Citation#pope2001) — Turbulent flows")


if __name__ == "__main__":
    main()

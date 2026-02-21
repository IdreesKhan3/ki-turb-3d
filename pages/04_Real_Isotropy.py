"""
Isotropy Validation (Real Space) Page — Streamlit

High-standard features:
- Reads real-space isotropy files (LBM/NS):
    * eps_real_validation*.csv or turbulence_validation*.csv (required)
    * reynolds_stress_validation*.csv (optional)
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


# --- Project imports ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.file_detector import detect_simulation_files
from core_physics import (
    load_turbulence_data,
    load_reynolds_stress,
    compute_reynolds_from_fractions,
    anisotropy_tensor,
    invariants,
)
from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
from content.real_isotropy_theory_content import get_real_isotropy_theory_markdown
from utils.report_builder import capture_button
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, _get_palette, convert_superscript
)
from utils.export_figs import export_panel
from visualizations.real_isotropy_vis import create_energy_fractions_figure, create_lumley_triangle_figure, create_diagonal_bii_figure, create_cross_correlations_figure, create_deviations_figure, create_convergence_figure

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
    """Delegate to utils.plot_style.resolve_curve_style (shared with vis and agents)."""
    from utils.plot_style import resolve_curve_style
    plot_key = _normalize_plot_name(plot_name)
    return resolve_curve_style(curve, idx, colors, ps, plot_key)


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
        "convergence": "Running standard deviation",
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
    
    # First, check files detected by file_detector (spectral_turb_stats: eps_real_validation or turbulence_validation)
    for f in files.get("spectral_turb_stats", []):
        name = Path(f).name
        if name.startswith("eps_real_validation") or name.startswith("turbulence_validation"):
            eps_file = Path(f)
            break

    # If not found, check for exact filenames in directory
    if eps_file is None:
        for candidate in ("eps_real_validation.csv", "turbulence_validation.csv"):
            exact_file = data_dir / candidate
            if exact_file.exists():
                eps_file = exact_file
                break

    # If still not found, check for glob patterns (LBM/NS)
    if eps_file is None:
        import glob
        for pattern in ("eps_real_validation*.csv", "turbulence_validation*.csv"):
            matches = glob.glob(str(data_dir / pattern))
            if matches:
                eps_file = Path(matches[0])
                break

    if eps_file is None or not eps_file.exists():
        st.error("Validation CSV not found in dataset folder (eps_real_validation*.csv or turbulence_validation*.csv)")
        st.info(f"Looking for: eps_real_validation*.csv, turbulence_validation*.csv")
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
        st.caption("• convergence → Y-axis for plot F")
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
        # (a) Temporal energy fractions — shared vis (agent-controlled ma_win)
        plot_name_a = "Energy Fractions (A)"
        ps_a = get_plot_style(plot_name_a)
        # Use palette from ps (matches spectra; sidebar palette selector applies)
        cols_a = _get_palette(ps_a)
        colors_orig = {'primary': cols_a[0 % len(cols_a)], 'secondary': cols_a[1 % len(cols_a)], 'tertiary': cols_a[2 % len(cols_a)]}
        legend_names_a = {
            "frac_x": st.session_state.real_iso_legends["Ex"],
            "frac_y": st.session_state.real_iso_legends["Ey"],
            "frac_z": st.session_state.real_iso_legends["Ez"],
        }
        axis_labels_a = {"x": st.session_state.axis_labels_real_iso["time"], "y": st.session_state.axis_labels_real_iso["energy_frac"]}
        fig_a = create_energy_fractions_figure(
            time_norm, E_x, E_y, E_z, ps_a,
            axis_labels=axis_labels_a,
            legend_names=legend_names_a,
            apply_style=False,
            ma_win=ma_win if ma_win and ma_win > 1 else None,
            add_raw_suffix=True,
            tol_list=tol_list_a,
            stationary_t=stationary_t,
        )

        layout_kwargs_a = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["energy_frac"],
            height=420,
        )
        layout_kwargs_a = apply_axis_limits(layout_kwargs_a, ps_a)
        layout_kwargs_a = apply_figure_size(layout_kwargs_a, ps_a)
        fig_a.update_layout(**layout_kwargs_a)
        fig_a = apply_plot_style(fig_a, ps_a)
        
        # Re-apply colors after plot style to prevent dimming (vis sets raw opacity; restore MA)
        # Skip when per-curve overrides are enabled — vis already applied them
        if not ps_a.get("enable_per_curve_style", False):
            ma_line_width = ps_a.get("line_width", 2.2) * 1.1
            raw_opacity = ps_a.get("raw_data_opacity", 0.5)
            raw_marker_size = max(2, ps_a.get("marker_size", 6) * 0.4)
            for trace in fig_a.data:
                if trace.name and "(MA-" in trace.name:
                    if "Ex" in trace.name or "E<sub>x</sub>" in trace.name:
                        trace.line.color = colors_orig['primary']
                    elif "Ey" in trace.name or "E<sub>y</sub>" in trace.name:
                        trace.line.color = colors_orig['secondary']
                    elif "Ez" in trace.name or "E<sub>z</sub>" in trace.name:
                        trace.line.color = colors_orig['tertiary']
                    trace.line.width = ma_line_width
                    trace.opacity = 1.0
                elif trace.name and "(raw)" in trace.name:
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
                        trace.marker.size = raw_marker_size
                        trace.marker.opacity = raw_opacity
                    trace.opacity = 1.0

        st.plotly_chart(fig_a, width='stretch')
        capture_button(fig_a, title="Real-Space Isotropy Analysis (Part A)", source_page="Real Isotropy")

        export_panel(fig_a, data_dir, "real_iso_energy_fractions")

        # (b) Lumley triangle
        plot_name_b = "Lumley Triangle (B)"
        ps_b = get_plot_style(plot_name_b)
        xi, eta = inv["xi"], inv["eta"]
        axis_labels_b = {"x": st.session_state.axis_labels_real_iso["lumley_x"], "y": st.session_state.axis_labels_real_iso["lumley_y"]}
        fig_b = create_lumley_triangle_figure(xi, eta, ps_b, axis_labels=axis_labels_b, apply_style=True)
        layout_kwargs_b = dict(height=420, showlegend=True)
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
        # (c) Diagonal b_ii — shared vis (agent-controlled tol_list)
        plot_name_c = "Diagonal b_ii (C)"
        ps_c = get_plot_style(plot_name_c)
        legend_names_c = {
            "b11": st.session_state.real_iso_legends["b11"],
            "b22": st.session_state.real_iso_legends["b22"],
            "b33": st.session_state.real_iso_legends["b33"],
        }
        axis_labels_c = {"x": st.session_state.axis_labels_real_iso["time"], "y": st.session_state.axis_labels_real_iso["bij"]}
        fig_c = create_diagonal_bii_figure(
            time_norm, b["b11"], b["b22"], b["b33"], ps_c,
            axis_labels=axis_labels_c,
            legend_names=legend_names_c,
            apply_style=False,
            tol_list=tol_list_c,
        )
        layout_kwargs_c = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["bij"],
            height=360,
        )
        layout_kwargs_c = apply_axis_limits(layout_kwargs_c, ps_c)
        layout_kwargs_c = apply_figure_size(layout_kwargs_c, ps_c)
        fig_c.update_layout(**layout_kwargs_c)
        fig_c = apply_plot_style(fig_c, ps_c)
        # Apply per-curve overrides (consistency with subplots D, E)
        colors_c = _get_palette(ps_c)
        for i, curve in enumerate(["b11", "b22", "b33"]):
            if i < len(fig_c.data):
                c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_c, ps_c, plot_name_c)
                fig_c.data[i].line.color = c
                fig_c.data[i].line.width = lw
                fig_c.data[i].line.dash = dash
        st.plotly_chart(fig_c, width='stretch')
        export_panel(fig_c, data_dir, "real_iso_bii_diag")

        # (d) Cross-correlations — shared vis (agent-controlled tol_list)
        plot_name_d = "Cross-correlations (D)"
        ps_d = get_plot_style(plot_name_d)
        legend_names_d = {
            "b12": st.session_state.real_iso_legends["b12"],
            "b13": st.session_state.real_iso_legends["b13"],
            "b23": st.session_state.real_iso_legends["b23"],
            "anis": st.session_state.real_iso_legends["anis"],
        }
        axis_labels_d = {"x": st.session_state.axis_labels_real_iso["time"], "y": st.session_state.axis_labels_real_iso["cross"]}
        fig_d = create_cross_correlations_figure(
            time_norm,
            b["b12"], b["b13"], b["b23"],
            inv["anis_index"],
            ps_d,
            axis_labels=axis_labels_d,
            legend_names=legend_names_d,
            tol_list=tol_list_d,
            apply_style=False,
        )
        layout_kwargs_d = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["cross"],
            height=360,
        )
        layout_kwargs_d = apply_axis_limits(layout_kwargs_d, ps_d)
        layout_kwargs_d = apply_figure_size(layout_kwargs_d, ps_d)
        fig_d.update_layout(**layout_kwargs_d)
        fig_d = apply_plot_style(fig_d, ps_d)
        # Apply per-curve overrides (consistency with subplots C, E)
        colors_d = _get_palette(ps_d)
        for i, curve in enumerate(["b12", "b13", "b23", "anis"]):
            if i < len(fig_d.data):
                c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_d, ps_d, plot_name_d)
                fig_d.data[i].line.color = c
                fig_d.data[i].line.width = lw
                fig_d.data[i].line.dash = dash
        st.plotly_chart(fig_d, width='stretch')
        export_panel(fig_d, data_dir, "real_iso_cross_corr")

    # ======================================================
    # Tab 3: Deviations (E) + Convergence (F)
    # ======================================================
    with tab3:
        # (e) Deviations — shared vis (agent-controlled tol_list, stationary_t)
        plot_name_e = "Deviations (E)"
        ps_e = get_plot_style(plot_name_e)
        devx = np.abs(E_x - 1 / 3)
        devy = np.abs(E_y - 1 / 3)
        devz = np.abs(E_z - 1 / 3)
        maxdev = np.maximum(np.maximum(devx, devy), devz)
        legend_names_e = {"devx": "devx", "devy": "devy", "devz": "devz", "maxdev": "Max deviation"}
        axis_labels_e = {"x": st.session_state.axis_labels_real_iso["time"], "y": st.session_state.axis_labels_real_iso["dev"]}
        fig_e = create_deviations_figure(
            time_norm, devx, devy, devz, maxdev, ps_e,
            axis_labels=axis_labels_e,
            legend_names=legend_names_e,
            tol_list=tol_list_e,
            stationary_t=stationary_t,
            apply_style=False,
        )
        layout_kwargs_e = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["dev"],
            height=360,
        )
        layout_kwargs_e = apply_axis_limits(layout_kwargs_e, ps_e)
        layout_kwargs_e = apply_figure_size(layout_kwargs_e, ps_e)
        fig_e.update_layout(**layout_kwargs_e)
        fig_e = apply_plot_style(fig_e, ps_e)
        # Apply per-curve overrides (consistency with subplots C, D)
        colors_e = _get_palette(ps_e)
        for i, curve in enumerate(["devx", "devy", "devz", "maxdev"]):
            if i < len(fig_e.data):
                c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_e, ps_e, plot_name_e)
                fig_e.data[i].line.color = c
                fig_e.data[i].line.width = lw
                fig_e.data[i].line.dash = dash
        st.plotly_chart(fig_e, width='stretch')
        export_panel(fig_e, data_dir, "real_iso_deviation")

        # (f) Convergence — shared vis (agent-controlled conv_windows)
        plot_name_f = "Convergence (F)"
        ps_f = get_plot_style(plot_name_f)
        min_len = len(E_x)
        conv_windows = [max(10, min_len // 10), max(20, min_len // 5)] if min_len > 20 else None
        axis_labels_f = {"x": st.session_state.axis_labels_real_iso["time"], "y": st.session_state.axis_labels_real_iso.get("convergence", "Running standard deviation")}
        fig_f = create_convergence_figure(
            time_norm, E_x, E_y, E_z, ps_f,
            axis_labels=axis_labels_f,
            conv_windows=conv_windows,
            apply_style=False,
        )
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
        st.markdown(get_real_isotropy_theory_markdown())


if __name__ == "__main__":
    main()

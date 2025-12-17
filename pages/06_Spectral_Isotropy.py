"""
Isotropy Validation (Spectral) Page — Streamlit

Features:
- Auto-detects isotropy coefficient files: isotropy_coeff_*.dat
- Supports multiple simulations for comparison (groups files by simulation/directory)
- Time-averages derivative-based spectral isotropy ratio IC(k) over user-selected snapshot window
- Computes derivative-based IC from Fortran output (column 7 / index 6)
- Computes and displays E11, E22, E33 component spectra
- Optional plots:
    (1) Time-averaged IC(k) with error bars/bands
    (2) Time-averaged E11, E22, E33 component spectra
    (3) Optional per-snapshot IC(k) lines for convergence visualization
- Summary table with statistics (mean, std, min, max) for each simulation
- Full user controls (in-memory session state): legend names, axis labels, plot style
- Research-grade export (PNG/PDF/SVG/JPG/WEBP/HTML)

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
import re
import glob
import sys


# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.file_detector import detect_simulation_files, natural_sort_key, group_files_by_simulation
from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
from utils.report_builder import capture_button
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, _get_palette, _normalize_plot_name, resolve_line_style,
    render_per_sim_style_ui
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
        text=title_text,
        font=dict(
            family=ps.get("font_family", "Arial"),
            size=ps.get("title_size", 16),
            color=font_color
        )
    )

def apply_plot_style(fig, ps):
    # Temporarily clear plot_title if show_plot_title is False to prevent centralized function from setting it
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
    default = default_plot_style()
    default.update({
        "enable_per_curve_style": False,
        "margin_left": 60,
        "margin_right": 20,
        "margin_top": 40,
        "margin_bottom": 50,
        "line_width": 2.2,
    })
    
    # Set default axis scale types and limits based on plot name
    if plot_name == "IC(k) Time-Avg":
        default["x_axis_type"] = "log"
        default["y_axis_type"] = "linear"
        default["enable_y_limits"] = True
        default["y_min"] = 0.8
        default["y_max"] = 1.3
    elif plot_name == "Component Spectra":
        default["x_axis_type"] = "log"
        default["y_axis_type"] = "log"
    
    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    
    # Apply theme first to get theme defaults
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = default.copy()
    merged = apply_theme_to_plot_style(merged, current_theme)
    
    # Then apply user overrides (from plot_style) - this ensures user settings override theme
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    
    return merged

def plot_style_sidebar(data_dir: Path, curves, plot_names: list, sim_groups=None):
    # Plot selector
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure",
        plot_names,
        key="speciso_plot_selector"
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
    key_prefix = f"speciso_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        saved_font = ps.get("font_family", "Arial")
        font_idx = fonts.index(saved_font) if saved_font in fonts else 0
        ps["font_family"] = st.selectbox("Font family", fonts, index=font_idx,
                                         key=f"{key_prefix}_font_family")
        ps["font_size"] = st.slider("Base/global font size", 8, 26, int(ps.get("font_size", 14)),
                                     key=f"{key_prefix}_font_size")
        ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)),
                                      key=f"{key_prefix}_title_size")
        ps["legend_size"] = st.slider("Legend font size", 8, 24, int(ps.get("legend_size", 12)),
                                       key=f"{key_prefix}_legend_size")
        ps["tick_font_size"] = st.slider("Tick label font size", 6, 24, int(ps.get("tick_font_size", 12)),
                                          key=f"{key_prefix}_tick_font_size")
        ps["axis_title_size"] = st.slider("Axis title font size", 8, 28, int(ps.get("axis_title_size", 14)),
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
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.2)),
                                      key=f"{key_prefix}_line_width")
        ps["marker_size"] = st.slider("Global marker size", 0, 14, int(ps.get("marker_size", 6)),
                                       key=f"{key_prefix}_marker_size")

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
        st.caption("**Curve meanings:** IC = time-averaged IC(k) main curves | IC_snap = per-snapshot IC(k) lines | E11/E22/E33 = component energy spectra")
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
                        saved_dash = s.get("dash") or "solid"
                        dash_idx = dash_opts.index(saved_dash) if saved_dash in dash_opts else 0
                        s["dash"] = st.selectbox("Dash", dash_opts,
                                                 index=dash_idx,
                                                 key=f"{key_prefix}_over_dash_{c}",
                                                 disabled=not s["enabled"])
                    with o5:
                        saved_marker = s.get("marker") or "circle"
                        marker_idx = marker_opts.index(saved_marker) if saved_marker in marker_opts else 0
                        s["marker"] = st.selectbox("Marker", marker_opts,
                                                   index=marker_idx,
                                                   key=f"{key_prefix}_over_marker_{c}",
                                                   disabled=not s["enabled"])
                    s["msize"] = st.slider("Marker size", 0, 18,
                                           int(s.get("msize") or ps.get("marker_size", 6)),
                                           key=f"{key_prefix}_over_msize_{c}",
                                           disabled=not s["enabled"])

        # Per-simulation styling (for multi-simulation comparison)
        if sim_groups and len(sim_groups) > 1:
            st.markdown("---")
            st.markdown("**Per-simulation styling**")
            if selected_plot == "IC(k) Time-Avg":
                render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_ic", 
                                        key_prefix=f"{key_prefix}_ic", include_marker=True, show_enable_checkbox=False)
            elif selected_plot == "Component Spectra":
                render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_eii", 
                                        key_prefix=f"{key_prefix}_eii", include_marker=True, show_enable_checkbox=False)

        st.markdown("---")
        reset_pressed = False
        if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset"):
                st.session_state.plot_styles[selected_plot] = {}
                
                # Clear widget state so widgets re-read from defaults on next run
                # This list is page-specific because each page has different key prefixes
                # and may have page-specific widgets (e.g., raw_data_opacity, show_plot_title)
                widget_keys = [
                    f"{key_prefix}_font_family",
                    f"{key_prefix}_font_size",
                    f"{key_prefix}_title_size",
                    f"{key_prefix}_legend_size",
                    f"{key_prefix}_tick_font_size",
                    f"{key_prefix}_axis_title_size",
                    f"{key_prefix}_plot_bgcolor",
                    f"{key_prefix}_paper_bgcolor",
                    f"{key_prefix}_tick_len",
                    f"{key_prefix}_tick_w",
                    f"{key_prefix}_ticks_outside",
                    f"{key_prefix}_x_axis_type",
                    f"{key_prefix}_y_axis_type",
                    f"{key_prefix}_x_tick_format",
                    f"{key_prefix}_x_tick_decimals",
                    f"{key_prefix}_y_tick_format",
                    f"{key_prefix}_y_tick_decimals",
                    f"{key_prefix}_show_axis_lines",
                    f"{key_prefix}_axis_line_width",
                    f"{key_prefix}_axis_line_color",
                    f"{key_prefix}_mirror_axes",
                    f"{key_prefix}_show_grid",
                    f"{key_prefix}_grid_on_x",
                    f"{key_prefix}_grid_on_y",
                    f"{key_prefix}_grid_w",
                    f"{key_prefix}_grid_dash",
                    f"{key_prefix}_grid_color",
                    f"{key_prefix}_grid_opacity",
                    f"{key_prefix}_show_minor_grid",
                    f"{key_prefix}_minor_grid_w",
                    f"{key_prefix}_minor_grid_dash",
                    f"{key_prefix}_minor_grid_color",
                    f"{key_prefix}_minor_grid_opacity",
                    f"{key_prefix}_line_width",
                    f"{key_prefix}_marker_size",
                    f"{key_prefix}_palette",
                    f"{key_prefix}_template",
                    # Plot Title
                    f"{key_prefix}_show_plot_title",
                    f"{key_prefix}_plot_title",
                    f"{key_prefix}_margin_left",
                    f"{key_prefix}_margin_right",
                    f"{key_prefix}_margin_top",
                    f"{key_prefix}_margin_bottom",
                    f"{key_prefix}_enable_per_curve",
                ]
                
                for i in range(10):
                    widget_keys.append(f"{key_prefix}_cust_color_{i}")
                
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
                
                widget_keys.extend([
                    f"{key_prefix}_enable_x_limits",
                    f"{key_prefix}_x_min",
                    f"{key_prefix}_x_max",
                    f"{key_prefix}_enable_y_limits",
                    f"{key_prefix}_y_min",
                    f"{key_prefix}_y_max",
                    f"{key_prefix}_enable_custom_size",
                    f"{key_prefix}_figure_width",
                    f"{key_prefix}_figure_height",
                ])
                
                for k in widget_keys:
                    if k in st.session_state:
                        del st.session_state[k]
                
                st.toast(f"Reset style for '{selected_plot}'.", icon="♻️")
                reset_pressed = True
                st.rerun()

    # Auto-save plot style changes (applies immediately) - but not if reset was pressed
    if not reset_pressed:
        st.session_state.plot_styles[selected_plot] = ps

def _resolve_curve_style(curve, idx, colors, ps, plot_name: str):
    default_color = colors[idx % len(colors)]
    default_width = ps.get("line_width", 2.2)
    default_dash = "solid"
    default_marker = "circle"
    default_msize = ps.get("marker_size", 6)

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
# Readers / time averaging
# ==========================================================
def _extract_iter(fname: str):
    stem = Path(fname).stem
    nums = re.findall(r"(\d+)", stem)
    return int(nums[-1]) if nums else None

@st.cache_data(show_spinner=False)
def _read_isotropy_coeff_file(fname: str):
    data = np.loadtxt(fname, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def _avg_isotropy_coeff(files):
    """
    Average IC(k), E11,E22,E33 across snapshots.
    Returns common k grid and mean/std arrays.
    """
    all_k, all_ic, all_e11, all_e22, all_e33 = [], [], [], [], []

    for f in files:
        d = _read_isotropy_coeff_file(str(f))
        if d.size == 0:
            continue

        k = d[:, 0]
        E11 = d[:, 1] if d.shape[1] > 1 else None
        E22 = d[:, 2] if d.shape[1] > 2 else None
        E33 = d[:, 3] if d.shape[1] > 3 else None

        # Use the derivative-based Spectral Isotropy Ratio from Fortran (column 6, 0-indexed)
        # Expected columns: k, E11, E22, E33, dE11/dk, IC_standard, IC_deriv
        if d.shape[1] >= 7:
            IC = d[:, 6]   # IC_deriv from Fortran (column 6, 0-indexed)
        else:
            # Fallback to spectral isotropy ratio if derivative-based not available
            IC = np.divide(E11, E22, out=np.zeros_like(E11), where=E22 != 0)

        # Filter valid data (matching auxiliary script criteria)
        valid = (k > 0.5) & np.isfinite(IC) & (E11 > 1e-15)
        if np.any(valid):
            all_k.append(k[valid])
            all_ic.append(IC[valid])
            if E11 is not None: all_e11.append(E11[valid])
            if E22 is not None: all_e22.append(E22[valid])
            if E33 is not None: all_e33.append(E33[valid])

    if not all_ic:
        return None

    unique_k = np.unique(np.concatenate(all_k))
    ic_mean, ic_std = np.zeros_like(unique_k), np.zeros_like(unique_k)
    e11_mean = np.zeros_like(unique_k)
    e22_mean = np.zeros_like(unique_k)
    e33_mean = np.zeros_like(unique_k)
    counts = np.zeros_like(unique_k)

    for i, k0 in enumerate(unique_k):
        ic_vals, e11_vals, e22_vals, e33_vals = [], [], [], []
        for k, ic, e11, e22, e33 in zip(all_k, all_ic,
                                        all_e11 or [None]*len(all_ic),
                                        all_e22 or [None]*len(all_ic),
                                        all_e33 or [None]*len(all_ic)):
            idx = np.argmin(np.abs(k - k0))
            if np.abs(k[idx] - k0) < 0.1:
                ic_vals.append(ic[idx])
                if e11 is not None: e11_vals.append(e11[idx])
                if e22 is not None: e22_vals.append(e22[idx])
                if e33 is not None: e33_vals.append(e33[idx])

        if ic_vals:
            ic_mean[i] = np.mean(ic_vals)
            ic_std[i] = np.std(ic_vals)
            counts[i] = len(ic_vals)
            if e11_vals: e11_mean[i] = np.mean(e11_vals)
            if e22_vals: e22_mean[i] = np.mean(e22_vals)
            if e33_vals: e33_mean[i] = np.mean(e33_vals)

    min_samples = max(1, len(all_ic)//2)
    mask = counts >= min_samples

    return {
        "k": unique_k[mask],
        "IC_mean": ic_mean[mask],
        "IC_std": ic_std[mask],
        "E11_mean": e11_mean[mask] if all_e11 else None,
        "E22_mean": e22_mean[mask] if all_e22 else None,
        "E33_mean": e33_mean[mask] if all_e33 else None,
    }


# ==========================================================
# Page main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    st.title("📈 Isotropy Validation — Spectral")

    # Get data directories from session state (support multiple directories)
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        # Fallback to single directory for backward compatibility
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return
    
    # Use first directory for metadata storage
    data_dir = Path(data_dirs[0]).resolve()

    # Default values (using Unicode/HTML instead of LaTeX for Streamlit compatibility)
    default_legends = {
        "IC": "IC(k) (time-avg)",
        "IC_snap": "IC(k) snapshots",
        "E11": "E<sub>11</sub>(k)",
        "E22": "E<sub>22</sub>(k)",
        "E33": "E<sub>33</sub>(k)",
    }
    default_axis_labels = {
        "k": "k",
        "ic": "IC(k)",
        "ek": "E<sub>ii</sub>(k)",
    }
    
    # Initialize with defaults, then merge with any loaded data
    if "spec_iso_legends" not in st.session_state:
        st.session_state.spec_iso_legends = default_legends.copy()
    else:
        # Ensure all default keys exist (merge defaults with existing)
        for key, value in default_legends.items():
            if key not in st.session_state.spec_iso_legends:
                st.session_state.spec_iso_legends[key] = value
    
    if "axis_labels_spec_iso" not in st.session_state:
        st.session_state.axis_labels_spec_iso = default_axis_labels.copy()
    else:
        # Ensure all default keys exist
        for key, value in default_axis_labels.items():
            if key not in st.session_state.axis_labels_spec_iso:
                st.session_state.axis_labels_spec_iso[key] = value
    # Initialize plot_styles if not exists
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}

    # Ensure all required keys are present
    for key, value in default_legends.items():
        if key not in st.session_state.spec_iso_legends:
            st.session_state.spec_iso_legends[key] = value
    for key, value in default_axis_labels.items():
        if key not in st.session_state.axis_labels_spec_iso:
            st.session_state.axis_labels_spec_iso[key] = value

    # Initialize legend names for simulations
    st.session_state.setdefault("spec_iso_sim_legend_names", {})

    # Find isotropy files from ALL directories and group by simulation
    ic_groups = {}
    for data_dir_path in data_dirs:
        # Resolve path to ensure it works regardless of how it was stored
        try:
            dir_path = Path(data_dir_path).resolve()
            if dir_path.exists() and dir_path.is_dir():
                # Process each directory independently
                files = detect_simulation_files(str(dir_path))
                # file_detector uses "isotropy" key for isotropy_coeff_*.dat files
                dir_ic_files = files.get("isotropy", [])
                if not dir_ic_files:
                    # Fallback to direct glob search
                    dir_ic_files = glob.glob(str(dir_path / "isotropy_coeff_*.dat"))
                
                if dir_ic_files:
                    # Group files by simulation pattern
                    dir_name = dir_path.name
                    dir_ic_str = [str(f) for f in dir_ic_files]
                    
                    # Try to group by pattern first
                    grouped = group_files_by_simulation(
                        dir_ic_str, r"(isotropy_coeff[_\w]*\d+)_\d+\.dat"
                    )
                    if not grouped:
                        # Fallback: group by directory name if pattern doesn't match
                        grouped = group_files_by_simulation(
                            dir_ic_str, r"isotropy_coeff_(\d+)_\d+\.dat"
                        )
                    
                    if grouped:
                        # Add directory prefix to distinguish simulations from different directories
                        for key, file_list in grouped.items():
                            new_key = f"{dir_name}_{key}" if len(data_dirs) > 1 else key
                            if new_key not in ic_groups:
                                ic_groups[new_key] = []
                            ic_groups[new_key].extend(file_list)
                    else:
                        # No pattern match: use directory name as group key
                        group_key = dir_name if len(data_dirs) > 1 else "default"
                        if group_key not in ic_groups:
                            ic_groups[group_key] = []
                        ic_groups[group_key].extend(dir_ic_str)
        except Exception:
            continue  # Skip invalid directories
    
    # Sort files within each group
    for key in ic_groups:
        ic_groups[key] = sorted(ic_groups[key], key=natural_sort_key)

    if not ic_groups:
        st.info("No isotropy_coeff_*.dat files found in any of the selected directories.")
        return

    # Sidebar legends + axis labels persistence
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Simulation legend names")
        for sim_prefix in sorted(ic_groups.keys()):
            st.session_state.spec_iso_sim_legend_names.setdefault(sim_prefix, _default_labelify(sim_prefix))
            st.session_state.spec_iso_sim_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.spec_iso_sim_legend_names[sim_prefix],
                key=f"speciso_sim_leg_{sim_prefix}"
            )
        st.markdown("---")
        st.markdown("### Curve names")
        for k in st.session_state.spec_iso_legends:
            st.session_state.spec_iso_legends[k] = st.text_input(
                k, st.session_state.spec_iso_legends[k], key=f"speciso_leg_{k}"
            )
        st.markdown("---")
        st.markdown("### Axis labels")
        for k in st.session_state.axis_labels_spec_iso:
            st.session_state.axis_labels_spec_iso[k] = st.text_input(
                k, st.session_state.axis_labels_spec_iso[k], key=f"speciso_ax_{k}"
            )
        if st.button("♻️ Reset labels/legends"):
            st.session_state.spec_iso_sim_legend_names = {k: _default_labelify(k) for k in ic_groups.keys()}
            st.session_state.spec_iso_legends = {
                "IC": "IC(k) (time-avg)",
                "IC_snap": "IC(k) snapshots",
                "E11": "E<sub>11</sub>(k)",
                "E22": "E<sub>22</sub>(k)",
                "E33": "E<sub>33</sub>(k)",
            }
            st.session_state.axis_labels_spec_iso = {
                "k": "k",
                "ic": "IC(k)",
                "ek": "E<sub>ii</sub>(k)",
            }
            st.toast("Reset.", icon="♻️")
            st.rerun()

    # Sidebar time window (use minimum length across all groups)
    min_len = min(len(files) for files in ic_groups.values()) if ic_groups else 1
    start_idx = st.sidebar.slider("Start file index", 1, min_len, 1)
    end_idx = st.sidebar.slider("End file index", start_idx, min_len, min_len)

    st.sidebar.subheader("Options")
    show_snapshot_lines = st.sidebar.checkbox("Show per-snapshot IC(k)", value=False)
    error_display = st.sidebar.radio(
        "Error display",
        ["Shaded band", "Error bars", "Both", "None"],
        index=3,
        help="Choose how to display ±1σ uncertainty"
    )
    show_std_band = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]
    show_component_spectra = st.sidebar.checkbox("Show E11/E22/E33 plot", value=True)

    # Curves used for per-curve overrides
    curves = ["IC","IC_snap","E11","E22","E33"]
    plot_names = ["IC(k) Time-Avg", "Component Spectra"]
    plot_style_sidebar(data_dir, curves, plot_names, sim_groups=ic_groups)

    tabs = st.tabs(["IC(k) Time-Avg", "Component Spectra", "Summary"])

    # ======================================================
    # Tab 1: IC(k)
    # ======================================================
    with tabs[0]:
        st.subheader("Time-averaged Spectral Isotropy Ratio")

        # Get plot-specific style
        plot_name_ic = "IC(k) Time-Avg"
        ps_ic = get_plot_style(plot_name_ic)
        colors_ic = _get_palette(ps_ic)

        fig_ic = go.Figure()

        # optional snapshot lines
        if show_snapshot_lines:
            for sim_prefix, files in sorted(ic_groups.items()):
                selected_files = tuple(files[start_idx-1:end_idx])
                for i, f in enumerate(selected_files):
                    d = _read_isotropy_coeff_file(str(f))
                    if d.size == 0:
                        continue
                    k0 = d[:,0]
                    if d.shape[1] >= 7:
                        IC0 = d[:,6]
                    else:
                        # Compute spectral isotropy ratio = E11/E22
                        IC0 = np.divide(d[:,1], d[:,2], out=np.zeros_like(d[:,1]), where=d[:,2]!=0)

                    # Use per-curve style for IC_snap if enabled
                    c_snap, lw_snap, dash_snap, mk_snap, ms_snap = _resolve_curve_style(
                        "IC_snap", 0, colors_ic, ps_ic, plot_name_ic
                    )
                    fig_ic.add_trace(go.Scatter(
                        x=k0, y=IC0, mode="lines",
                        name=st.session_state.spec_iso_legends["IC_snap"],
                        line=dict(color=c_snap, width=lw_snap, dash=dash_snap),
                        showlegend=(sim_prefix == sorted(ic_groups.keys())[0] and i==0)
                    ))

        # Plot each simulation group as a separate curve
        plotted_any = False
        for idx, (sim_prefix, files) in enumerate(sorted(ic_groups.items())):
            selected_files = tuple(files[start_idx-1:end_idx])
            if not selected_files:
                continue

            avg = _avg_isotropy_coeff(selected_files)
            if avg is None:
                continue

            k = avg["k"]
            IC_mean = avg["IC_mean"]
            IC_std = avg["IC_std"]

            # Check if per-curve style is enabled for IC
            c_ic, lw_ic, dash_ic, mk_ic, ms_ic = _resolve_curve_style(
                "IC", idx, colors_ic, ps_ic, plot_name_ic
            )
            
            # Get per-simulation style (for when per-curve is not enabled)
            color_sim, lw_sim, dash_sim, marker_sim, msize_sim, override_on_sim = resolve_line_style(
                sim_prefix, idx, colors_ic, ps_ic,
                style_key="per_sim_style_ic",
                include_marker=True,
                default_marker="circle"
            )
            
            # Use per-curve style if enabled, otherwise use per-simulation style
            if ps_ic.get("enable_per_curve_style", False):
                color, lw, dash = c_ic, lw_ic, dash_ic
                marker, msize = mk_ic, ms_ic
                override_on = (mk_ic != "circle" or ms_ic > 0)
            else:
                color, lw, dash = color_sim, lw_sim, dash_sim
                marker, msize = marker_sim, msize_sim
                override_on = override_on_sim
            
            legend_name = st.session_state.spec_iso_sim_legend_names.get(
                sim_prefix, _default_labelify(sim_prefix)
            )
            plotted_any = True

            mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
            trace_kwargs = dict(
                x=k, y=IC_mean, mode=mode,
                name=legend_name,
                line=dict(color=color, width=lw, dash=dash),
            )
            if override_on and marker and msize > 0:
                trace_kwargs["marker"] = dict(symbol=marker, size=msize)
            if show_error_bars and IC_std is not None:
                trace_kwargs["error_y"] = dict(
                    type="data",
                    array=IC_std,
                    visible=True,
                    thickness=1,
                    color=color
                )
            fig_ic.add_trace(go.Scatter(**trace_kwargs))

            if show_std_band and IC_std is not None:
                rgb = hex_to_rgb(color)
                fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)"
                fig_ic.add_trace(go.Scatter(
                    x=np.concatenate([k, k[::-1]]),
                    y=np.concatenate([IC_mean-IC_std, (IC_mean+IC_std)[::-1]]),
                    fill="toself", fillcolor=fill_rgba,
                    line=dict(width=0), showlegend=False, hoverinfo="skip"
                ))

        if not plotted_any:
            st.error("No valid data in selected isotropy files.")
            return

        fig_ic.add_hline(y=1.0, line_dash="dash", line_color="red", line_width=1.2)

        layout_kwargs_ic = dict(
            xaxis_title=st.session_state.axis_labels_spec_iso["k"],
            yaxis_title=st.session_state.axis_labels_spec_iso["ic"],
            height=500,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs_ic = apply_axis_limits(layout_kwargs_ic, ps_ic)
        layout_kwargs_ic = apply_figure_size(layout_kwargs_ic, ps_ic)
        fig_ic.update_layout(**layout_kwargs_ic)
        fig_ic = apply_plot_style(fig_ic, ps_ic)
        st.plotly_chart(fig_ic, width='stretch')
        capture_button(fig_ic, title="Spectral Isotropy (IC)", source_page="Spectral Isotropy")
        export_panel(fig_ic, data_dir, "spectral_isotropy_IC")

    # ======================================================
    # Tab 2: Component spectra
    # ======================================================
    with tabs[1]:
        st.subheader("Component Spectra (time-avg)")

        if not show_component_spectra:
            st.info("Component spectra not available (disabled).")
        else:
            # Get plot-specific style
            plot_name_eii = "Component Spectra"
            ps_eii = get_plot_style(plot_name_eii)
            colors_eii = _get_palette(ps_eii)
            
            fig_eii = go.Figure()
            plotted_any_eii = False
            
            # Plot each simulation group
            for idx, (sim_prefix, files) in enumerate(sorted(ic_groups.items())):
                selected_files = tuple(files[start_idx-1:end_idx])
                if not selected_files:
                    continue

                avg = _avg_isotropy_coeff(selected_files)
                if avg is None or avg["E11_mean"] is None:
                    continue

                k = avg["k"]
                legend_name = st.session_state.spec_iso_sim_legend_names.get(
                    sim_prefix, _default_labelify(sim_prefix)
                )
                
                # Get base color for this simulation
                color_base, lw_base, dash_base, marker_base, msize_base, override_on_base = resolve_line_style(
                    sim_prefix, idx, colors_eii, ps_eii,
                    style_key="per_sim_style_eii",
                    include_marker=True,
                    default_marker="circle"
                )
                
                # Plot E11, E22, E33 for this simulation
                for i, curve in enumerate(["E11","E22","E33"]):
                    arr = avg[f"{curve}_mean"]
                    # Use per-curve style if enabled, otherwise use per-simulation style
                    c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_eii, ps_eii, plot_name_eii)
                    # Only override with simulation color if per-curve style is NOT enabled
                    if not ps_eii.get("enable_per_curve_style", False) and override_on_base:
                        c = color_base
                        lw = lw_base
                        dash = dash_base
                    
                    fig_eii.add_trace(go.Scatter(
                        x=k, y=arr, mode="lines",
                        name=f"{legend_name} - {st.session_state.spec_iso_legends[curve]}",
                        line=dict(color=c, width=lw, dash=dash),
                    ))
                    plotted_any_eii = True
            
            if not plotted_any_eii:
                st.info("Component spectra not available (missing columns in data).")
                return

            layout_kwargs_eii = dict(
                xaxis_title=st.session_state.axis_labels_spec_iso["k"],
                yaxis_title=st.session_state.axis_labels_spec_iso["ek"],
                width=700,
                height=600,
            )
            layout_kwargs_eii = apply_axis_limits(layout_kwargs_eii, ps_eii)
            # Don't apply figure size override to keep fixed dimensions
            fig_eii.update_layout(**layout_kwargs_eii)
            fig_eii = apply_plot_style(fig_eii, ps_eii)
            st.plotly_chart(fig_eii, width='content')
            capture_button(fig_eii, title="Spectral Isotropy (E_ii)", source_page="Spectral Isotropy")
            export_panel(fig_eii, data_dir, "spectral_isotropy_Eii")

    # ======================================================
    # Tab 3: summary
    # ======================================================
    with tabs[2]:
        st.subheader("Summary")
        summary_rows = []
        for sim_prefix, files in sorted(ic_groups.items()):
            selected_files = tuple(files[start_idx-1:end_idx])
            if not selected_files:
                continue
            
            avg = _avg_isotropy_coeff(selected_files)
            if avg is None:
                continue
            
            IC_mean = avg["IC_mean"]
            IC_std = avg["IC_std"]
            legend_name = st.session_state.spec_iso_sim_legend_names.get(
                sim_prefix, _default_labelify(sim_prefix)
            )
            
            summary_rows.append({
                "Simulation": legend_name,
                "Snapshots used": len(selected_files),
                "Mean IC": float(np.nanmean(IC_mean)),
                "Std(IC)": float(np.nanmean(IC_std)),
                "Min IC": float(np.nanmin(IC_mean)),
                "Max IC": float(np.nanmax(IC_mean)),
            })
        
        if summary_rows:
            df = pd.DataFrame(summary_rows)
            st.dataframe(df, width='stretch')
            st.download_button(
                "Download summary CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="spectral_isotropy_summary.csv",
                mime="text/csv"
            )
        else:
            st.info("No data available for summary.")

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**One-dimensional energy spectra:**")
        st.latex(r"""
        E_{11}(k) = |\hat{u}(k)|^2, \quad E_{22}(k) = |\hat{v}(k)|^2, \quad E_{33}(k) = |\hat{w}(k)|^2
        """)
        st.markdown(r"""
        where $\hat{u}(k)$, $\hat{v}(k)$, and $\hat{w}(k)$ are the Fourier transforms of velocity components $u$, $v$, and $w$ in the $x$, $y$, and $z$ directions, respectively. These are plotted in the **Component Spectra** tab.
        """)
        
        st.markdown("**Derivative-based Spectral Isotropy Ratio:**")
        st.latex(r"\text{IC}_{\text{deriv}}(k) = \frac{2E_{11}(k)}{2E_{22}(k) - k \frac{dE_{11}}{dk}}")
        st.markdown(r"""
        The derivative-based formula includes the spectral derivative term, making it less sensitive to numerical noise when $E_{22}(k)$ is small. The ratio $\text{IC}_{\text{deriv}}(k)$ is plotted as a function of wavenumber $k$ in the **IC(k) Time-Avg** tab, and summary statistics (mean, std, min, max) are shown in the **Summary** tab.
        """)
        
        st.markdown("**For isotropic turbulence:**")
        st.latex(r"E_{11}(k) = E_{22}(k) = E_{33}(k) \quad \Rightarrow \quad \text{IC}_{\text{deriv}}(k) \approx 1")
        
        st.divider()
        st.markdown("**References:** [Batchelor (1953)](/Citation#batchelor1953) — The theory of homogeneous turbulence; [Singh & Komrakova (2024)](/Citation#singh2024) — Comparison of forcing schemes to sustain homogeneous isotropic turbulence")


if __name__ == "__main__":
    main()

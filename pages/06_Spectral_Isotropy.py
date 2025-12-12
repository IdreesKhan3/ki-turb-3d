"""
Isotropy Validation (Spectral) Page — Streamlit

Features:
- Auto-detects isotropy coefficient files:
    isotropy_coeff_*.dat
- Groups into single spectral isotropy dataset (can extend to multi-sim later)
- Time-averages IC(k) over user-selected snapshot window
- Uses derivative-based IC from Fortran if present (col 7 / index 6)
  else falls back to IC = E22/E11
- Optional plots:
    (1) Time-averaged IC(k)
    (2) Time-averaged E11,E22,E33
    (3) Snapshot IC(k) lines + avg overlay (convergence)
- FULL persistent UI controls (legend names, axis labels, plot style)
- Research-grade export (PNG/PDF/SVG/EPS/JPG/WEBP/TIFF/HTML)

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
import json
import re
import glob
import sys


# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.file_detector import detect_simulation_files, natural_sort_key
from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
from utils.report_builder import capture_button
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, _get_palette, _normalize_plot_name
)
from utils.export_figs import export_panel
st.set_page_config(page_icon="⚫")


# ==========================================================
# JSON persistence (dataset-local)
# ==========================================================
def _legend_json_path(data_dir: Path) -> Path:
    return data_dir / "legend_names.json"

def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()

def _load_ui_metadata(data_dir: Path):
    path = _legend_json_path(data_dir)
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        # Merge with defaults to ensure all required keys exist
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
        # Start with defaults, then update with saved values
        loaded_legends = default_legends.copy()
        loaded_legends.update(meta.get("spec_iso_legends", {}))
        st.session_state.spec_iso_legends = loaded_legends
        
        loaded_axis_labels = default_axis_labels.copy()
        loaded_axis_labels.update(meta.get("axis_labels_spec_iso", {}))
        st.session_state.axis_labels_spec_iso = loaded_axis_labels
        
        # Load plot-specific styles
        st.session_state.plot_styles = meta.get("plot_styles", {})
        # Backward compatibility: if plot_styles doesn't exist, use old plot_style
        if not st.session_state.plot_styles and "plot_style" in meta:
            st.session_state.plot_styles = {}
    except Exception:
        st.toast("legend_names.json exists but could not be read. Using defaults.", icon="⚠️")

def _save_ui_metadata(data_dir: Path):
    path = _legend_json_path(data_dir)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    old.update({
        "spec_iso_legends": st.session_state.get("spec_iso_legends", {}),
        "axis_labels_spec_iso": st.session_state.get("axis_labels_spec_iso", {}),
        "plot_styles": st.session_state.get("plot_styles", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json: {e}")


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
    # Deep merge: start with defaults, then update with plot-specific overrides
    merged = default.copy()
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    
    # Apply theme to ensure font_color is set correctly
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = apply_theme_to_plot_style(merged, current_theme)
    
    return merged

def plot_style_sidebar(data_dir: Path, curves, plot_names: list):
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
        b1, b2 = st.columns(2)
        reset_pressed = False
        with b1:
            if st.button("💾 Save Plot Style", key=f"{key_prefix}_save"):
                st.session_state.plot_styles[selected_plot] = ps
                _save_ui_metadata(data_dir)
                st.success(f"Saved style for '{selected_plot}'.")
        with b2:
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
                
                _save_ui_metadata(data_dir)
                st.toast(f"Reset style for '{selected_plot}'.", icon="♻️")
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

    if st.session_state.get("_last_speciso_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        # After loading, ensure all required keys are present
        for key, value in default_legends.items():
            if key not in st.session_state.spec_iso_legends:
                st.session_state.spec_iso_legends[key] = value
        for key, value in default_axis_labels.items():
            if key not in st.session_state.axis_labels_spec_iso:
                st.session_state.axis_labels_spec_iso[key] = value
        if "plot_styles" not in st.session_state:
            st.session_state.plot_styles = {}
        st.session_state["_last_speciso_dir"] = str(data_dir)

    # Find isotropy files from ALL directories independently
    all_ic_files = []
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
                all_ic_files.extend([str(f) for f in dir_ic_files])
        except Exception:
            continue  # Skip invalid directories
    
    ic_files = sorted(all_ic_files, key=natural_sort_key)

    if not ic_files:
        st.info("No isotropy_coeff_*.dat files found in any of the selected directories.")
        return

    # Sidebar legends + axis labels persistence
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
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
        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save labels/legends"):
                _save_ui_metadata(data_dir)
                st.success("Saved to legend_names.json")
        with c2:
            if st.button("♻️ Reset labels/legends"):
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
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")

    # Sidebar time window
    st.sidebar.subheader("Time Window")
    min_len = len(ic_files)
    start_idx = st.sidebar.slider("Start file index", 1, min_len, 1)
    end_idx = st.sidebar.slider("End file index", start_idx, min_len, min_len)
    selected_files = ic_files[start_idx-1:end_idx]

    st.sidebar.subheader("Options")
    show_snapshot_lines = st.sidebar.checkbox("Show per-snapshot IC(k)", value=False)
    show_std_band = st.sidebar.checkbox("Show ±1σ band", value=True)
    show_component_spectra = st.sidebar.checkbox("Show E11/E22/E33 plot", value=True)

    # Curves used for per-curve overrides
    curves = ["IC","IC_snap","E11","E22","E33"]
    plot_names = ["IC(k) Time-Avg", "Component Spectra"]
    plot_style_sidebar(data_dir, curves, plot_names)

    # Compute averaging
    avg = _avg_isotropy_coeff(selected_files)
    if avg is None:
        st.error("No valid data in selected isotropy files.")
        return

    k = avg["k"]
    IC_mean = avg["IC_mean"]
    IC_std = avg["IC_std"]

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

                fig_ic.add_trace(go.Scatter(
                    x=k0, y=IC0, mode="lines",
                    name=st.session_state.spec_iso_legends["IC_snap"],
                    line=dict(color="rgba(0,0,0,0.15)", width=1),
                    showlegend=(i==0)
                ))

        c, lw, dash, mk, ms = _resolve_curve_style("IC", 0, colors_ic, ps_ic, plot_name_ic)
        fig_ic.add_trace(go.Scatter(
            x=k, y=IC_mean, mode="lines",
            name=st.session_state.spec_iso_legends["IC"],
            line=dict(color=c, width=lw, dash=dash),
        ))

        if show_std_band:
            rgb = hex_to_rgb(c)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.18)"
            fig_ic.add_trace(go.Scatter(
                x=np.concatenate([k, k[::-1]]),
                y=np.concatenate([IC_mean-IC_std, (IC_mean+IC_std)[::-1]]),
                fill="toself", fillcolor=fill_rgba,
                line=dict(width=0), showlegend=False, hoverinfo="skip"
            ))

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

        if not show_component_spectra or avg["E11_mean"] is None:
            st.info("Component spectra not available (missing columns or disabled).")
        else:
            # Get plot-specific style
            plot_name_eii = "Component Spectra"
            ps_eii = get_plot_style(plot_name_eii)
            colors_eii = _get_palette(ps_eii)
            
            fig_eii = go.Figure()
            for i, curve in enumerate(["E11","E22","E33"]):
                arr = avg[f"{curve}_mean"]
                c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_eii, ps_eii, plot_name_eii)
                fig_eii.add_trace(go.Scatter(
                    x=k, y=arr, mode="lines",
                    name=st.session_state.spec_iso_legends[curve],
                    line=dict(color=c, width=lw, dash=dash),
                ))

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
        df = pd.DataFrame([{
            "Snapshots used": len(selected_files),
            "Mean IC": float(np.nanmean(IC_mean)),
            "Std(IC)": float(np.nanmean(IC_std)),
            "Min IC": float(np.nanmin(IC_mean)),
            "Max IC": float(np.nanmax(IC_mean)),
        }])
        st.dataframe(df, width='stretch')
        st.download_button(
            "Download summary CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="spectral_isotropy_summary.csv",
            mime="text/csv"
        )

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**Spectral Isotropy Ratio:**")
        st.latex(r"IC(k) = \frac{E_{11}(k)}{E_{22}(k)}")
        
        st.markdown("**Derivative-based Spectral Isotropy Ratio (more robust):**")
        st.latex(r"IC_{deriv}(k) = \frac{2E_{11}(k)}{2E_{22}(k) - k \frac{dE_{11}}{dk}}")
        st.markdown("The derivative-based formula includes the spectral derivative term, making it less sensitive to numerical noise when $E_{22}(k)$ is small.")
        
        st.markdown("**For isotropic turbulence:**")
        st.latex(r"E_{11}(k) = E_{22}(k) = E_{33}(k) \quad \Rightarrow \quad IC(k) \approx 1")
        
        st.markdown("**Note:** The code automatically uses the derivative-based ratio from Fortran (column 7) when available, otherwise falls back to spectral isotropy ratio.")


if __name__ == "__main__":
    main()

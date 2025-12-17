"""
Flatness Factors Page (Streamlit) — High Standard + Full Styling

Features:
- Time-averaged flatness with ±1σ band at log-spaced r positions
- Optional Gaussian reference line (F=3)
- Time window selection
- Robust to missing/empty files
- Cached I/O + averaging for speed
- Full user controls (in-memory session state):
    * legends, axis labels
    * fonts, sizes, tick style, grids (major/minor), background colors, theme
    * palette / custom colors
    * per-simulation line style overrides (color/width/dash/marker)
- Research-grade export:
    * Multi-format PNG/PDF/SVG/EPS/JPG/WEBP/TIFF + HTML
    * Scale (DPI-like), optional width/height override

Requires kaleido for static exports:
    pip install -U kaleido
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import hex_to_rgb
from pathlib import Path
import sys
import re

# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.text_reader import read_flatness_file
from utils.file_detector import detect_simulation_files, group_files_by_simulation, natural_sort_key
from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
from utils.report_builder import capture_button
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, _get_palette, _normalize_plot_name,
    resolve_line_style, render_per_sim_style_ui, ensure_per_sim_defaults
)
from utils.export_figs import export_panel
st.set_page_config(page_icon="⚫")


# ==========================================================
# Cached readers
# ==========================================================
@st.cache_data(show_spinner=False)
def _read_flatness_cached(fname: str):
    r, F = read_flatness_file(fname)
    return np.asarray(r, float), np.asarray(F, float)


# ==========================================================
# Helpers
# ==========================================================
@st.cache_data(show_spinner=False)
def _compute_time_avg_flatness(files: tuple, num_errorbars: int = 20):
    """
    Time-average flatness data and select log-spaced r values for error bars.
    Returns (r_plot, F_mean, F_std) on selected r indices.
    """
    all_r = []
    all_flatness = []

    for f in files:
        r, F = _read_flatness_cached(str(f))
        if r is None or F is None or len(r) == 0 or len(F) == 0:
            continue
        all_r.append(r)
        all_flatness.append(F)

    if not all_r:
        return None, None, None

    r_full = all_r[0]
    flatness_array = np.array(all_flatness)

    # Guard shape mismatches
    if flatness_array.ndim != 2 or flatness_array.shape[1] != r_full.shape[0]:
        return None, None, None

    flatness_mean = np.mean(flatness_array, axis=0)
    flatness_std = np.std(flatness_array, axis=0)

    r_pos = r_full[r_full > 0]
    if r_pos.size < 2:
        return None, None, None

    log_r_vals = np.logspace(np.log10(r_pos[0]), np.log10(r_pos[-1]), num=num_errorbars)
    log_indices = sorted(set([int(np.argmin(np.abs(r_full - val))) for val in log_r_vals]))

    r_plot = r_full[log_indices]
    F_mean = flatness_mean[log_indices]
    F_std = flatness_std[log_indices]
    return r_plot, F_mean, F_std


def _format_legend_name(prefix: str) -> str:
    name = prefix.replace("flatness_", "").replace("data", "").strip("_")
    name = name.replace("_", " ").title()
    return name if name else prefix


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
            l=ps.get("margin_left", 50),
            r=ps.get("margin_right", 30),
            t=ps.get("margin_top", 30),
            b=ps.get("margin_bottom", 50)
        ))
    
    # Always set title with correct font color if show_plot_title is True
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    
    return fig

def get_plot_style(plot_name: str):
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
    
    # Update reference line color for dark theme if it's still at light theme default
    if "Dark" in current_theme:
        if merged.get("reference_color") == "#000000":
            merged["reference_color"] = "#dcdcaa"  # Light yellow - visible on dark background
    
    return merged


def plot_style_sidebar(data_dir: Path, sim_groups, plot_names: list):
    # Plot selector
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure",
        plot_names,
        key="flatness_plot_selector"
    )
    
    # Get or create plot-specific style
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}
    
    # Start with defaults, merge with plot-specific overrides
    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name(selected_plot)
    
    # Ensure per-sim defaults
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_flatness", include_marker=True)
    
    # Create unique key prefix for all widgets
    key_prefix = f"flatness_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        font_idx = fonts.index(ps.get("font_family", "Arial")) if ps.get("font_family", "Arial") in fonts else 0
        ps["font_family"] = st.selectbox(
            "Font family", fonts,
            index=font_idx,
            key=f"{key_prefix}_font_family"
        )
        ps["font_size"] = st.slider("Base/global font size", 8, 26, int(ps.get("font_size", 14)),
                                     key=f"{key_prefix}_font_size")
        ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)),
                                      key=f"{key_prefix}_title_size")
        ps["legend_size"] = st.slider("Legend font size", 8, 24, int(ps.get("legend_size", 12)),
                                       key=f"{key_prefix}_legend_size")
        ps["show_legend"] = st.checkbox(
            "Show legend", 
            bool(ps.get("show_legend", True)),
            help="Display legend on the plot",
            key=f"{key_prefix}_show_legend"
        )
        ps["tick_font_size"] = st.slider("Tick label font size", 6, 24, int(ps.get("tick_font_size", 12)),
                                          key=f"{key_prefix}_tick_font_size")
        ps["axis_title_size"] = st.slider("Axis title font size", 8, 28, int(ps.get("axis_title_size", 14)),
                                           key=f"{key_prefix}_axis_title_size")

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background (inside axes)", ps.get("plot_bgcolor", "#FFFFFF"),
                                             key=f"{key_prefix}_plot_bgcolor")
        ps["paper_bgcolor"] = st.color_picker("Paper background (outside axes)", ps.get("paper_bgcolor", "#FFFFFF"),
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
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)),
                                           key=f"{key_prefix}_grid_on_x")
        with gcol2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)),
                                           key=f"{key_prefix}_grid_on_y")
        ps["grid_w"] = st.slider("Major grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)),
                                  key=f"{key_prefix}_grid_w")
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        grid_dash_idx = grid_styles.index(ps.get("grid_dash", "dot")) if ps.get("grid_dash", "dot") in grid_styles else 1
        ps["grid_dash"] = st.selectbox("Major grid type", grid_styles,
                                       index=grid_dash_idx,
                                       key=f"{key_prefix}_grid_dash")
        ps["grid_color"] = st.color_picker("Major grid color", ps.get("grid_color", "#B0B0B0"),
                                           key=f"{key_prefix}_grid_color")
        ps["grid_opacity"] = st.slider("Major grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)),
                                        key=f"{key_prefix}_grid_opacity")

        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)),
                                             key=f"{key_prefix}_show_minor_grid")
        ps["minor_grid_w"] = st.slider("Minor grid width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)),
                                        key=f"{key_prefix}_minor_grid_w")
        minor_grid_dash_idx = grid_styles.index(ps.get("minor_grid_dash", "dot")) if ps.get("minor_grid_dash", "dot") in grid_styles else 1
        ps["minor_grid_dash"] = st.selectbox("Minor grid type", grid_styles,
                                             index=minor_grid_dash_idx,
                                             key=f"{key_prefix}_minor_grid_dash")
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"),
                                                  key=f"{key_prefix}_minor_grid_color")
        ps["minor_grid_opacity"] = st.slider("Minor grid opacity", 0.0, 1.0,
                                             float(ps.get("minor_grid_opacity", 0.45)),
                                             key=f"{key_prefix}_minor_grid_opacity")

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.4)),
                                      key=f"{key_prefix}_line_width")
        ps["marker_size"] = st.slider("Global marker size", 0, 18, int(ps.get("marker_size", 7)),
                                       key=f"{key_prefix}_marker_size")
        ps["std_alpha"] = st.slider("Std band opacity", 0.05, 0.6, float(ps.get("std_alpha", 0.18)),
                                    key=f"{key_prefix}_std_alpha")

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        palette_idx = palettes.index(ps.get("palette", "Plotly")) if ps.get("palette", "Plotly") in palettes else 0
        ps["palette"] = st.selectbox("Palette", palettes,
                                     index=palette_idx,
                                     key=f"{key_prefix}_palette")
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
        ps["reference_color"] = st.color_picker("Reference line color", ps.get("reference_color", "#000000"),
                                                key=f"{key_prefix}_reference_color")
        ps["reference_width"] = st.slider("Reference line width", 0.5, 4.0,
                                          float(ps.get("reference_width", 1.5)),
                                          key=f"{key_prefix}_reference_width")
        ps["reference_dash"] = st.selectbox("Reference line dash", grid_styles,
                                            index=grid_styles.index(ps.get("reference_dash", "dot")) if ps.get("reference_dash", "dot") in grid_styles else 1,
                                            key=f"{key_prefix}_reference_dash")

        st.markdown("---")
        st.markdown("**Theme**")
        old_template = ps.get("template", "plotly_white")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox("Template", templates,
                                      index=templates.index(old_template) if old_template in templates else 0,
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
                                                value=int(ps.get("margin_left", 50)), 
                                                step=5, key=f"{key_prefix}_margin_left")
            ps["margin_top"] = st.number_input("Top margin (px)", min_value=0, max_value=200, 
                                                value=int(ps.get("margin_top", 30)), 
                                                step=5, key=f"{key_prefix}_margin_top")
        with col2:
            ps["margin_right"] = st.number_input("Right margin (px)", min_value=0, max_value=200, 
                                                  value=int(ps.get("margin_right", 30)), 
                                                  step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, 
                                                   value=int(ps.get("margin_bottom", 50)), 
                                                   step=5, key=f"{key_prefix}_margin_bottom")
        st.markdown("---")
        render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_flatness", 
                                key_prefix=f"{key_prefix}_sim", include_marker=True, show_enable_checkbox=True)

        st.markdown("---")
        reset_pressed = False
        if st.button("♻️ Reset Plot Style", key=f"{key_prefix}_reset"):
                st.session_state.plot_styles[selected_plot] = {}
                
                # Clear widget state so widgets re-read from defaults on next run
                widget_keys = [
                    # Fonts
                    f"{key_prefix}_font_family",
                    f"{key_prefix}_font_size",
                    f"{key_prefix}_title_size",
                    f"{key_prefix}_legend_size",
                    f"{key_prefix}_show_legend",
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
                    f"{key_prefix}_std_alpha",
                    # Colors
                    f"{key_prefix}_palette",
                    f"{key_prefix}_reference_color",
                    f"{key_prefix}_reference_width",
                    f"{key_prefix}_reference_dash",
                    # Theme
                    f"{key_prefix}_template",
                    # Plot Title
                    f"{key_prefix}_show_plot_title",
                    f"{key_prefix}_plot_title",
                    # Axis limits
                    f"{key_prefix}_enable_x_limits",
                    f"{key_prefix}_x_min",
                    f"{key_prefix}_x_max",
                    f"{key_prefix}_enable_y_limits",
                    f"{key_prefix}_y_min",
                    f"{key_prefix}_y_max",
                    # Figure size
                    f"{key_prefix}_enable_custom_size",
                    f"{key_prefix}_figure_width",
                    f"{key_prefix}_figure_height",
                    # Margins
                    f"{key_prefix}_margin_left",
                    f"{key_prefix}_margin_right",
                    f"{key_prefix}_margin_top",
                    f"{key_prefix}_margin_bottom",
                    # Per-sim global toggle
                    f"{key_prefix}_enable_per_sim",
                ]
                
                # Custom color inputs
                for i in range(10):
                    widget_keys.append(f"{key_prefix}_cust_color_{i}")
                
                # Per-simulation style widgets
                if sim_groups:
                    for sim_prefix in sim_groups.keys():
                        for suffix in [
                            "over_on",
                            "over_color",
                            "over_width",
                            "over_dash",
                            "over_marker",
                            "over_msize",
                        ]:
                            widget_keys.append(f"{key_prefix}_sim_{suffix}_{sim_prefix}")
                
                for k in widget_keys:
                    if k in st.session_state:
                        del st.session_state[k]
                
                st.toast(f"Reset style for '{selected_plot}'.", icon="♻️")
                reset_pressed = True
                st.rerun()

    # Auto-save plot style changes (applies immediately) - but not if reset was pressed
    if not reset_pressed:
        st.session_state.plot_styles[selected_plot] = ps


def _color_to_rgb_tuple(color):
    """Convert color to RGB tuple, handling both hex and RGB string formats."""
    if color.startswith("rgb("):
        # Parse RGB string like "rgb(27, 158, 119)"
        match = re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", color)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    # Try hex format
    try:
        return hex_to_rgb(color)
    except (ValueError, TypeError):
        # Fallback to default if conversion fails
        return (0, 0, 0)


# ==========================================================
# Main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("📉 Flatness Factors")

    # Get data directories from session state (support multiple directories)
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        # Fallback to single directory for backward compatibility
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the Overview page.")
        return

    # Use first directory for metadata storage
    data_dir = Path(data_dirs[0])

    # Defaults (session)
    st.session_state.setdefault("flatness_legend_names", {})
    st.session_state.setdefault("axis_labels_flatness", {
        "x": "Separation distance $r$ (lattice units)",
        "y": "Longitudinal flatness $F_L(r)$",
    })
    st.session_state.setdefault("plot_styles", {})


    # Detect flatness files from all directories
    all_flatness_files = []
    for data_dir_path in data_dirs:
        # Resolve path to ensure it works regardless of how it was stored
        try:
            data_dir_obj = Path(data_dir_path).resolve()
            if data_dir_obj.exists() and data_dir_obj.is_dir():
                # Process each directory independently
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_flatness = files_dict.get("flatness", [])
                all_flatness_files.extend(dir_flatness)
        except Exception:
            continue  # Skip invalid directories
    
    if not all_flatness_files:
        st.info("No flatness files found. Expected format: `flatness_data*_*.txt`")
        return

    # Group by simulation prefix, with directory name when multiple directories
    if len(data_dirs) > 1:
        # Multiple directories: group by directory name + simulation pattern
        sim_groups = {}
        for data_dir_path in data_dirs:
            data_dir_obj = Path(data_dir_path).resolve()
            dir_name = data_dir_obj.name  # e.g., "768", "512", "128"
            
            # Get files from this directory - use string comparison for robustness
            data_dir_str = str(data_dir_obj)
            dir_flatness = [f for f in all_flatness_files if str(Path(f).resolve().parent) == data_dir_str]
            
            # If no files found, re-check this directory directly
            if not dir_flatness:
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_flatness = [str(f) for f in files_dict.get("flatness", [])]
            
            if dir_flatness:
                # Try pattern with data: flatness_data1_t*.txt
                dir_sim_groups = group_files_by_simulation(
                    sorted([str(f) for f in dir_flatness], key=natural_sort_key),
                    r"(flatness_data\d+)_t\d+\.txt"
                )
                # If that fails, try pattern without underscore: flatness_data1_t*.txt (alternative format)
                if not dir_sim_groups:
                    dir_sim_groups = group_files_by_simulation(
                        sorted([str(f) for f in dir_flatness], key=natural_sort_key),
                        r"(flatness_data\d+)_\d+\.txt"
                    )
                # If that fails, try pattern with just number: flatness1_t*.txt
                if not dir_sim_groups:
                    dir_sim_groups = group_files_by_simulation(
                        sorted([str(f) for f in dir_flatness], key=natural_sort_key),
                        r"(flatness\d+)_t\d+\.txt"
                    )
                
                if dir_sim_groups:
                    # Files matched pattern - use pattern-based grouping
                    for key, files in dir_sim_groups.items():
                        new_key = f"{dir_name}_{key}" if key else dir_name
                        sim_groups[new_key] = files
                else:
                    # Files didn't match pattern - treat entire directory as one simulation
                    sim_groups[dir_name] = sorted([str(f) for f in dir_flatness], key=natural_sort_key)
    else:
        # Single directory - group by simulation prefix
        sim_groups = group_files_by_simulation(
            sorted([str(f) for f in all_flatness_files], key=natural_sort_key),
            r"(flatness_data\d+)_t\d+\.txt"
        ) if all_flatness_files else {}
        # If that fails, try pattern without underscore: flatness_data1_t*.txt
        if not sim_groups and all_flatness_files:
            sim_groups = group_files_by_simulation(
                sorted([str(f) for f in all_flatness_files], key=natural_sort_key),
                r"(flatness_data\d+)_\d+\.txt"
            )
        # If that fails, try pattern with just number: flatness1_t*.txt
        if not sim_groups and all_flatness_files:
            sim_groups = group_files_by_simulation(
                sorted([str(f) for f in all_flatness_files], key=natural_sort_key),
                r"(flatness\d+)_t\d+\.txt"
            )
        
        # If grouping failed in single directory, treat all files as one simulation
        if not sim_groups and all_flatness_files:
            sim_groups["flatness"] = sorted([str(f) for f in all_flatness_files], key=natural_sort_key)
    
    if not sim_groups:
        st.warning("Could not group flatness files by simulation type.")
        return

    # Sidebar: time window + physics options
    st.sidebar.subheader("Time Window")
    max_files = min(len(v) for v in sim_groups.values())
    start_idx = st.sidebar.slider("Start file index", 1, max_files, 1)
    end_idx = st.sidebar.slider("End file index", start_idx, max_files, max_files)

    st.sidebar.subheader("Averaging / Error bars")
    num_errorbars = st.sidebar.slider("Number of error bar points", 10, 80, 20)
    error_display = st.sidebar.radio(
        "Error display",
        ["Shaded band", "Error bars", "Both", "None"],
        index=0,
        help="Choose how to display ±1σ uncertainty"
    )
    show_std = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]

    st.sidebar.subheader("Plot Options")
    show_reference = st.sidebar.checkbox("Show Gaussian reference (F=3)", value=True)

    # Sidebar: legends + axis labels (persistent)
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        for sim_prefix in sorted(sim_groups.keys()):
            st.session_state.flatness_legend_names.setdefault(
                sim_prefix, _format_legend_name(sim_prefix)
            )
            st.session_state.flatness_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.flatness_legend_names[sim_prefix],
                key=f"legend_flat_{sim_prefix}"
            )

        st.markdown("---")
        st.markdown("### Axis labels")
        st.session_state.axis_labels_flatness["x"] = st.text_input(
            "X-axis label",
            value=st.session_state.axis_labels_flatness.get("x"),
            key="axis_flat_x"
        )
        st.session_state.axis_labels_flatness["y"] = st.text_input(
            "Y-axis label",
            value=st.session_state.axis_labels_flatness.get("y"),
            key="axis_flat_y"
        )

        if st.button("♻️ Reset labels/legends"):
            st.session_state.flatness_legend_names = {
                k: _format_legend_name(k) for k in sim_groups.keys()
            }
            st.session_state.axis_labels_flatness = {
                "x": "Separation distance $r$ (lattice units)",
                "y": "Longitudinal flatness $F_L(r)$",
            }
            st.toast("Reset.", icon="♻️")
            st.rerun()

    # Sidebar: full plot style (persistent)
    plot_names = ["Flatness Factors"]
    plot_style_sidebar(data_dir, sim_groups, plot_names)
    
    # Get plot-specific style
    plot_name = "Flatness Factors"
    ps = get_plot_style(plot_name)
    colors = _get_palette(ps)

    # =========================
    # Main plot
    # =========================
    st.header("Time-Averaged Flatness Factors")

    fig = go.Figure()
    plotted_any = False

    for idx, (sim_prefix, files) in enumerate(sorted(sim_groups.items())):
        selected_files = tuple(files[start_idx-1:end_idx])
        if not selected_files:
            continue

        r_plot, F_mean, F_std = _compute_time_avg_flatness(selected_files, num_errorbars)
        if r_plot is None:
            continue

        color, lw, dash, marker, msize, override_on = resolve_line_style(
            sim_prefix, idx, colors, ps,
            style_key="per_sim_style_flatness",
            include_marker=True,
            default_marker="square"
        )

        legend_name = st.session_state.flatness_legend_names.get(
            sim_prefix, _format_legend_name(sim_prefix)
        )
        plotted_any = True

        mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
        trace_kwargs = dict(
            x=r_plot,
            y=F_mean,
            mode=mode,
            name=legend_name,
            line=dict(color=color, width=lw, dash=dash),
            hovertemplate="r=%{x:.3g}<br>F(r)=%{y:.3g}<extra></extra>"
        )
        if override_on and marker and msize > 0:
            trace_kwargs["marker"] = dict(size=msize, symbol=marker, line=dict(width=1, color=color))
        if show_error_bars and F_std is not None:
            trace_kwargs["error_y"] = dict(
                type="data",
                array=F_std,
                visible=True,
                thickness=1,
                color=color
            )
        fig.add_trace(go.Scatter(**trace_kwargs))

        if show_std and F_std is not None:
            rgb = _color_to_rgb_tuple(color)
            fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps['std_alpha']})"
            fig.add_trace(go.Scatter(
                x=np.concatenate([r_plot, r_plot[::-1]]),
                y=np.concatenate([F_mean - F_std, (F_mean + F_std)[::-1]]),
                fill="toself",
                fillcolor=fill_rgba,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip"
            ))

    if plotted_any and show_reference:
        # add_hline uses axis coords; fine after we set axes
        fig.add_hline(
            y=3,
            line_dash=ps.get("reference_dash", "dot"),
            line_color=ps.get("reference_color", "#000000"),
            line_width=ps.get("reference_width", 1.5),
            annotation_text="Gaussian (F=3)",
            annotation_position="right"
        )

    if plotted_any:
        layout_kwargs = dict(
            xaxis_title=st.session_state.axis_labels_flatness["x"],
            yaxis_title=st.session_state.axis_labels_flatness["y"],
            legend_title="Simulation",
            height=500,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs = apply_axis_limits(layout_kwargs, ps)
        layout_kwargs = apply_figure_size(layout_kwargs, ps)
        fig.update_layout(**layout_kwargs)
        fig = apply_plot_style(fig, ps)

        st.plotly_chart(fig, width='stretch')
        capture_button(fig, title="Flatness Factors", source_page="Flatness")

        st.subheader("Export Figure")
        export_panel(fig, data_dir, base_name="flatness_factors")

    else:
        st.info("No valid flatness data could be plotted from selected range.")

    # Theory section
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**Longitudinal flatness factor:**")
        st.latex(r"""
        F_L(r) = \frac{\langle [\delta u_L(r)]^4 \rangle}{\langle [\delta u_L(r)]^2 \rangle^2}
        """)
        st.markdown(r"""
        where $\delta u_L(r) = u_L(\mathbf{x} + r\mathbf{e}_L) - u_L(\mathbf{x})$ is the longitudinal velocity increment.
        """)
        
        st.markdown("**Interpretation:**")
        st.markdown(r"""
        - $F_L(r) = 3$: Gaussian increments (no intermittency)
        - $F_L(r) > 3$: Intermittent, fat-tailed PDFs
        - $F_L(r) < 3$: Sub-Gaussian
        """)
        
        st.divider()
        st.markdown("**Reference:** [Pope (2001)](/Citation#pope2001) — Turbulent flows")


if __name__ == "__main__":
    main()

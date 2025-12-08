"""
Flatness Factors Page (Streamlit) — High Standard + Persistent UI Metadata + Full Styling

Features:
- Time-averaged flatness with ±1σ band at log-spaced r positions
- Optional Gaussian reference line (F=3)
- Time window selection
- Robust to missing/empty files
- Cached I/O + averaging for speed
- Persistent dataset-local JSON: legend_names.json stores:
    * flatness legends
    * flatness axis labels
    * plot_style (fonts/ticks/grids/backgrounds/palette/highlight/per-sim)
- Full user controls:
    * legends, axis labels (persistent)
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
import json

# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.text_reader import read_flatness_file
from utils.file_detector import detect_simulation_files, group_files_by_simulation, natural_sort_key
from utils.theme_config import inject_theme_css
from utils.report_builder import capture_button
from utils.plot_style import resolve_line_style, render_per_sim_style_ui, render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size
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
    """Load flatness legends + axis labels + plot_styles from legend_names.json if present."""
    path = _legend_json_path(data_dir)
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))

        # Keep other pages' settings intact; just read our keys if present
        st.session_state.flatness_legend_names = meta.get("flatness_legends", {})
        st.session_state.axis_labels_flatness = meta.get("axis_labels_flatness", {})
        
        # Load plot_styles (per-plot system)
        st.session_state.plot_styles = meta.get("plot_styles", {})
        # Backward compatibility: if plot_styles doesn't exist, migrate from old plot_style
        if not st.session_state.plot_styles and "plot_style" in meta:
            # Migrate old single plot_style to plot
            old_style = meta.get("plot_style", {})
            st.session_state.plot_styles = {
                "Flatness Factors": old_style.copy(),
            }
    except Exception:
        st.toast("legend_names.json exists but could not be read. Using defaults.", icon="⚠️")

def _save_ui_metadata(data_dir: Path):
    """Merge-save UI state to legend_names.json."""
    path = _legend_json_path(data_dir)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    old.update({
        "flatness_legends": st.session_state.get("flatness_legend_names", {}),
        "axis_labels_flatness": st.session_state.get("axis_labels_flatness", {}),
        "plot_styles": st.session_state.get("plot_styles", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json (read-only folder?): {e}")


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
# Plot styling system (shared with spectra page)
# ==========================================================
def _default_plot_style():
    return {
        # Fonts
        "font_family": "Arial",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,

        # Ticks
        "tick_len": 6,
        "tick_w": 1.2,
        "ticks_outside": True,

        # Backgrounds
        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",

        # Grid (MAJOR)
        "show_grid": True,
        "grid_on_x": True,
        "grid_on_y": True,
        "grid_w": 0.6,
        "grid_dash": "dot",
        "grid_color": "#B0B0B0",
        "grid_opacity": 0.6,

        # Grid (MINOR)
        "show_minor_grid": False,
        "minor_grid_w": 0.4,
        "minor_grid_dash": "dot",
        "minor_grid_color": "#D0D0D0",
        "minor_grid_opacity": 0.45,

        # Curves
        "line_width": 2.4,
        "marker_size": 7,
        "std_alpha": 0.18,

        # Colors & theme
        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",

        # Reference line style
        "reference_color": "#000000",
        "reference_dash": "dot",
        "reference_width": 1.5,

        # Per-simulation overrides
        "enable_per_sim_style": False,
        "per_sim_style_flatness": {},  # {sim_prefix: {enabled,color,width,dash,marker,msize}}
        
        # Axis limits
        "enable_x_limits": False,
        "x_min": None,
        "x_max": None,
        "enable_y_limits": False,
        "y_min": None,
        "y_max": None,
        
        # Figure size
        "enable_custom_size": False,
        "figure_width": 800,
        "figure_height": 500,
        
        # Frame/Margin size
        "margin_left": 50,
        "margin_right": 30,
        "margin_top": 30,
        "margin_bottom": 50,
    }

def _get_palette(ps):
    if ps["palette"] == "Custom":
        cols = ps.get("custom_colors", [])
        return cols if cols else pc.qualitative.Plotly

    mapping = {
        "Plotly": pc.qualitative.Plotly,
        "D3": pc.qualitative.D3,
        "G10": pc.qualitative.G10,
        "T10": pc.qualitative.T10,
        "Dark2": pc.qualitative.Dark2,
        "Set1": pc.qualitative.Set1,
        "Set2": pc.qualitative.Set2,
        "Pastel1": pc.qualitative.Pastel1,
        "Bold": pc.qualitative.Bold,
        "Prism": pc.qualitative.Prism,
    }
    return mapping.get(ps["palette"], pc.qualitative.Plotly)

def _axis_title_font(ps):
    return dict(family=ps["font_family"], size=ps["axis_title_size"])

def _tick_font(ps):
    return dict(family=ps["font_family"], size=ps["tick_font_size"])

def apply_plot_style(fig, ps):
    fig.update_layout(
        template=ps["template"],
        font=dict(family=ps["font_family"], size=ps["font_size"]),
        legend=dict(font=dict(size=ps["legend_size"])),
        hovermode="x unified",
        plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
        paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"),
        margin=dict(
            l=ps.get("margin_left", 50),
            r=ps.get("margin_right", 30),
            t=ps.get("margin_top", 30),
            b=ps.get("margin_bottom", 50)
        ),
    )

    tick_dir = "outside" if ps["ticks_outside"] else "inside"

    show_x_grid = ps["show_grid"] and ps.get("grid_on_x", True)
    show_y_grid = ps["show_grid"] and ps.get("grid_on_y", True)
    grid_rgba = f"rgba{hex_to_rgb(ps['grid_color']) + (ps['grid_opacity'],)}"

    show_minor = ps.get("show_minor_grid", False)
    minor_rgba = f"rgba{hex_to_rgb(ps['minor_grid_color']) + (ps['minor_grid_opacity'],)}"

    fig.update_xaxes(
        ticks=tick_dir,
        ticklen=ps["tick_len"],
        tickwidth=ps["tick_w"],
        tickfont=_tick_font(ps),
        title_font=_axis_title_font(ps),
        showgrid=show_x_grid,
        gridwidth=ps["grid_w"],
        griddash=ps["grid_dash"],
        gridcolor=grid_rgba,
        minor=dict(
            showgrid=show_minor,
            gridwidth=ps["minor_grid_w"],
            griddash=ps["minor_grid_dash"],
            gridcolor=minor_rgba,
        )
    )
    fig.update_yaxes(
        ticks=tick_dir,
        ticklen=ps["tick_len"],
        tickwidth=ps["tick_w"],
        tickfont=_tick_font(ps),
        title_font=_axis_title_font(ps),
        showgrid=show_y_grid,
        gridwidth=ps["grid_w"],
        griddash=ps["grid_dash"],
        gridcolor=grid_rgba,
        minor=dict(
            showgrid=show_minor,
            gridwidth=ps["minor_grid_w"],
            griddash=ps["minor_grid_dash"],
            gridcolor=minor_rgba,
        )
    )
    return fig

def _normalize_plot_name(plot_name: str) -> str:
    """Normalize plot name to a valid key format."""
    return plot_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

def get_plot_style(plot_name: str):
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    default = _default_plot_style()
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
    return merged

def _ensure_per_sim_defaults(ps, sim_groups):
    ps.setdefault("per_sim_style_flatness", {})
    for k in sim_groups.keys():
        ps["per_sim_style_flatness"].setdefault(k, {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": None,
            "marker": "square",
            "msize": None,
        })


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
    _ensure_per_sim_defaults(ps, sim_groups)
    
    # Create unique key prefix for all widgets
    key_prefix = f"flatness_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        ps["font_family"] = st.selectbox(
            "Font family", fonts,
            index=fonts.index(ps.get("font_family", "Arial")),
            key=f"{key_prefix}_font_family"
        )
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
        ps["grid_dash"] = st.selectbox("Major grid type", grid_styles,
                                       index=grid_styles.index(ps.get("grid_dash", "dot")),
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
        ps["minor_grid_dash"] = st.selectbox("Minor grid type", grid_styles,
                                             index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
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
        ps["marker_size"] = st.slider("Global marker size", 0, 14, int(ps.get("marker_size", 7)),
                                       key=f"{key_prefix}_marker_size")
        ps["std_alpha"] = st.slider("Std band opacity", 0.05, 0.6, float(ps.get("std_alpha", 0.18)),
                                    key=f"{key_prefix}_std_alpha")

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox("Palette", palettes,
                                     index=palettes.index(ps.get("palette", "Plotly")),
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
                                            index=grid_styles.index(ps.get("reference_dash", "dot")),
                                            key=f"{key_prefix}_reference_dash")

        st.markdown("---")
        st.markdown("**Theme**")
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
                                key_prefix=f"{key_prefix}_sim", include_marker=True)

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
                _save_ui_metadata(data_dir)
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

def _resolve_line_style(sim_prefix, idx, colors, ps):
    default_color = colors[idx % len(colors)]
    default_width = ps["line_width"]
    default_dash = "solid"
    default_marker = "square"
    default_msize = ps["marker_size"]

    if not ps.get("enable_per_sim_style", False):
        return default_color, default_width, default_dash, default_marker, default_msize

    s = ps.get("per_sim_style_flatness", {}).get(sim_prefix, {})
    if not s.get("enabled", False):
        return default_color, default_width, default_dash, default_marker, default_msize

    color = s.get("color") or default_color
    width = float(s.get("width") or default_width)
    dash = s.get("dash") or default_dash
    marker = s.get("marker") or default_marker
    msize = int(s.get("msize") or default_msize)
    return color, width, dash, marker, msize


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

    # Load JSON once per dataset change
    if st.session_state.get("_last_flatness_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        if "plot_styles" not in st.session_state:
            st.session_state.plot_styles = {}
        st.session_state.setdefault("flatness_legend_names", {})
        st.session_state.setdefault("axis_labels_flatness", {
            "x": "Separation distance $r$ (lattice units)",
            "y": "Longitudinal flatness $F_L(r)$",
        })
        st.session_state["_last_flatness_dir"] = str(data_dir)

    # Detect flatness files from all directories
    all_flatness_files = []
    for data_dir_path in data_dirs:
        data_dir_obj = Path(data_dir_path)
        if data_dir_obj.exists():
            files_dict = detect_simulation_files(str(data_dir_obj))
            dir_flatness = files_dict.get("flatness", [])
            all_flatness_files.extend(dir_flatness)
    
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

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save labels/legends"):
                _save_ui_metadata(data_dir)
                st.success("Saved to legend_names.json")
        with b2:
            if st.button("♻️ Reset labels/legends"):
                st.session_state.flatness_legend_names = {
                    k: _format_legend_name(k) for k in sim_groups.keys()
                }
                st.session_state.axis_labels_flatness = {
                    "x": "Separation distance $r$ (lattice units)",
                    "y": "Longitudinal flatness $F_L(r)$",
                }
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")
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
            xaxis_type="log",
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
        st.markdown("**Longitudinal Flatness Factor**")
        st.markdown("The longitudinal flatness factor $F_L(r)$ measures intermittency of velocity increments:")
        st.latex(r"F_L(r) = \frac{\langle [\delta u_L(r)]^4 \rangle}{\langle [\delta u_L(r)]^2 \rangle^2}")
        st.markdown("where $\\delta u_L(r) = u_L(\\mathbf{x}+r\\mathbf{e}_L) - u_L(\\mathbf{x})$.")
        
        st.markdown("**Interpretation**")
        st.markdown("- $F_L(r)=3$: Gaussian increments (no intermittency)")
        st.markdown("- $F_L(r)>3$: intermittent, fat-tailed PDFs")
        st.markdown("- $F_L(r)<3$: sub-Gaussian")


if __name__ == "__main__":
    main()

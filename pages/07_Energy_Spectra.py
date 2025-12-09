"""
Energy Spectra Page (Streamlit) — High Standard + Persistent UI Metadata + Full Styling

NEW in this version:
- Research-grade export panel:
    * User can export to: PNG, PDF, SVG, EPS, JPG/JPEG, WEBP, TIFF*
    * Chooses format(s) via multiselect
    * Controls export scale (like DPI), width/height override
    * One-click export for current figure(s)
  (*TIFF depends on kaleido + pillow; if TIFF fails, app explains.)

Everything remains persistent per dataset in legend_names.json:
- legends, axis labels
- plot style (fonts/ticks/grids/backgrounds/highlight/per-sim styles)
"""

import streamlit as st
import numpy as np
import glob # module
import re   # module
import json # module
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import hex_to_rgb

# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.spectrum_reader import read_spectrum_file
from data_readers.norm_spectrum_reader import read_norm_spectrum_file
from utils.file_detector import natural_sort_key, group_files_by_simulation
from utils.theme_config import inject_theme_css
from utils.report_builder import capture_button
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    _get_palette, _normalize_plot_name,
    resolve_line_style, render_per_sim_style_ui
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
    """Load legends + axis labels + plot_styles from legend_names.json if present."""
    path = _legend_json_path(data_dir)
    if not path.exists():
        return

    try:
        meta = json.loads(path.read_text(encoding="utf-8"))

        st.session_state.spectrum_legend_names = meta.get("raw_legends", {})
        st.session_state.norm_legend_names = meta.get("norm_legends", {})

        st.session_state.axis_labels_raw = meta.get("axis_labels_raw", {})
        st.session_state.axis_labels_norm = meta.get("axis_labels_norm", {})

        # Load plot_styles (new per-plot system)
        st.session_state.plot_styles = meta.get("plot_styles", {})
        # Backward compatibility: if plot_styles doesn't exist, migrate from old plot_style
        if not st.session_state.plot_styles and "plot_style" in meta:
            # Migrate old single plot_style to all plots
            old_style = meta.get("plot_style", {})
            st.session_state.plot_styles = {
                "Raw Energy Spectrum": old_style.copy(),
                "Normalized Spectrum": old_style.copy(),
                "Time Evolution": old_style.copy(),
            }

    except Exception:
        st.toast("legend_names.json exists but could not be read. Using defaults.",
                 icon="⚠️")

def _save_ui_metadata(data_dir: Path):
    """Save UI state to legend_names.json."""
    meta = {
        "raw_legends": st.session_state.get("spectrum_legend_names", {}),
        "norm_legends": st.session_state.get("norm_legend_names", {}),
        "axis_labels_raw": st.session_state.get("axis_labels_raw", {}),
        "axis_labels_norm": st.session_state.get("axis_labels_norm", {}),
        "plot_styles": st.session_state.get("plot_styles", {}),
    }
    try:
        _legend_json_path(data_dir).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
    except Exception as e:
        st.error(f"Could not save legend_names.json (read-only folder?): {e}")


# ==========================================================
# Cached readers
# ==========================================================
@st.cache_data(show_spinner=False)
def _read_spectrum_cached(fname: str):
    k, E = read_spectrum_file(fname)
    return np.asarray(k, float), np.asarray(E, float)

@st.cache_data(show_spinner=False)
def _read_norm_cached(fname: str):
    keta, Enorm, Epope = read_norm_spectrum_file(fname)
    return (np.asarray(keta, float),
            np.asarray(Enorm, float),
            np.asarray(Epope, float))


# ==========================================================
# Helpers
# ==========================================================
def _extract_iter(fname: str):
    """Extract last number from filename stem."""
    stem = Path(fname).stem
    nums = re.findall(r"(\d+)", stem)
    return int(nums[-1]) if nums else None

@st.cache_data(show_spinner=False)
def _compute_time_avg(files: tuple):
    """Return k, E_avg, E_std using notebook-identical formulas."""
    energy_accum = None
    energy_sq_accum = None
    count = 0
    k_vals = None

    for f in files:
        k, E = _read_spectrum_cached(str(f))
        if k_vals is None:
            k_vals = k
            energy_accum = np.zeros_like(E)
            energy_sq_accum = np.zeros_like(E)

        if E.shape != energy_accum.shape:
            continue

        energy_accum += E
        energy_sq_accum += E**2
        count += 1

    if count == 0:
        return None, None, None

    E_avg = energy_accum / count
    E_var = (energy_sq_accum / count) - E_avg**2
    E_std = np.sqrt(np.maximum(E_var, 0.0))
    return k_vals, E_avg, E_std

@st.cache_data(show_spinner=False)
def _compute_time_avg_norm(files: tuple):
    """Average normalized spectra + Pope model over selected files."""
    keta_vals = None
    En_accum, En_sq_accum, Ep_accum = None, None, None
    count = 0

    for f in files:
        keta, Enorm, Epope = _read_norm_cached(str(f))
        if keta_vals is None:
            keta_vals = keta
            En_accum = np.zeros_like(Enorm)
            En_sq_accum = np.zeros_like(Enorm)
            Ep_accum = np.zeros_like(Epope)

        if Enorm.shape != En_accum.shape:
            continue

        En_accum += Enorm
        En_sq_accum += Enorm**2
        Ep_accum += Epope
        count += 1

    if count == 0:
        return None, None, None, None

    En_avg = En_accum / count
    En_var = (En_sq_accum / count) - En_avg**2
    En_std = np.sqrt(np.maximum(En_var, 0.0))
    Ep_avg = Ep_accum / count
    return keta_vals, En_avg, En_std, Ep_avg


def _add_kolmogorov_line(fig, k_vals, E_avg, kmin, kmax, ps,
                         kolm_scale_factor=1.0,
                         label="Kolmogorov k<sup>-5/3</sup>"):
    """Add scaled -5/3 reference line on [kmin,kmax]."""
    mask = (k_vals >= kmin) & (k_vals <= kmax)
    k_fit = k_vals[mask]
    if k_fit.size < 3:
        return fig

    ref = k_fit ** (-5.0 / 3.0)
    mid_idx = np.argmin(np.abs(k_fit - np.median(k_fit)))
    scale = E_avg[mask][mid_idx] / ref[mid_idx]
    ref *= scale
    ref *= kolm_scale_factor  # Apply the manual scale factor

    fig.add_trace(go.Scatter(
        x=k_fit, y=ref,
        mode="lines",
        name=label,
        line=dict(color=ps["kolmogorov_color"],
                  width=ps["line_width"],
                  dash="dot"),
        hovertemplate="k=%{x:.3g}<br>ref=%{y:.3g}<extra></extra>"
    ))
    return fig


# ==========================================================
# Plot styling system (using centralized module)
# ==========================================================
def apply_plot_style(fig, ps):
    fig = apply_plot_style_base(fig, ps)
    
    if any(k in ps for k in ["margin_left", "margin_right", "margin_top", "margin_bottom"]):
        fig.update_layout(margin=dict(
            l=ps.get("margin_left", 40),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 30),
            b=ps.get("margin_bottom", 40)
        ))
    
    return fig

def get_plot_style(plot_name: str):
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    default = default_plot_style()
    default.update({
        "line_width": 2.4,
        "margin_left": 40,
        "margin_right": 20,
        "margin_top": 30,
        "margin_bottom": 40,
        "std_alpha": 0.18,
        "pope_color": "#000000",
        "kolmogorov_color": "#666666",
        "highlight_color": "#E41A1C",
        "per_sim_style_raw": {},
        "per_sim_style_norm": {},
        "x_axis_type": "log",
        "y_axis_type": "log",
    })
    
    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    merged = default.copy()
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

def plot_style_sidebar(data_dir: Path, sim_groups, norm_groups, plot_names: list):
    # Plot selector
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure",
        plot_names,
        key="energy_plot_selector"
    )
    
    # Get or create plot-specific style
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}
    
    # Start with defaults, merge with plot-specific overrides
    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name(selected_plot)
    
    # Create unique key prefix for all widgets
    key_prefix = f"energy_{plot_key}"

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
        ps["plot_bgcolor"] = st.color_picker(
            "Plot background (inside axes)", ps.get("plot_bgcolor", "#FFFFFF"),
            key=f"{key_prefix}_plot_bgcolor"
        )
        ps["paper_bgcolor"] = st.color_picker(
            "Paper background (outside axes)", ps.get("paper_bgcolor", "#FFFFFF"),
            key=f"{key_prefix}_paper_bgcolor"
        )

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
        ps["grid_dash"] = st.selectbox(
            "Major grid type", grid_styles,
            index=grid_styles.index(ps.get("grid_dash", "dot")),
            key=f"{key_prefix}_grid_dash"
        )
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
        ps["minor_grid_dash"] = st.selectbox(
            "Minor grid type", grid_styles,
            index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
            key=f"{key_prefix}_minor_grid_dash"
        )
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"),
                                                  key=f"{key_prefix}_minor_grid_color")
        ps["minor_grid_opacity"] = st.slider(
            "Minor grid opacity", 0.0, 1.0, float(ps.get("minor_grid_opacity", 0.45)),
            key=f"{key_prefix}_minor_grid_opacity"
        )

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 1.0, 7.0, float(ps.get("line_width", 2.4)),
                                      key=f"{key_prefix}_line_width")
        ps["marker_size"] = st.slider("Marker size", 0, 14, int(ps.get("marker_size", 6)),
                                       key=f"{key_prefix}_marker_size")
        ps["std_alpha"] = st.slider("Std band opacity", 0.05, 0.6, float(ps.get("std_alpha", 0.18)),
                                    key=f"{key_prefix}_std_alpha")

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox("Palette", palettes, index=palettes.index(ps.get("palette", "Plotly")),
                                     key=f"{key_prefix}_palette")

        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"{key_prefix}_cust_color_{i}"))
            ps["custom_colors"] = new_cols

        ps["pope_color"] = st.color_picker("Pope model color", ps.get("pope_color", "#000000"),
                                           key=f"{key_prefix}_pope_color")
        ps["kolmogorov_color"] = st.color_picker("Kolmogorov line color", ps.get("kolmogorov_color", "#666666"),
                                                  key=f"{key_prefix}_kolmogorov_color")

        st.markdown("---")
        st.markdown("**Highlight curve**")
        ps["highlight_color"] = st.color_picker(
            "Highlight color (time evolution)", ps.get("highlight_color", "#E41A1C"),
            key=f"{key_prefix}_highlight_color"
        )

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
                                                value=int(ps.get("margin_left", 40)), 
                                                step=5, key=f"{key_prefix}_margin_left")
            ps["margin_top"] = st.number_input("Top margin (px)", min_value=0, max_value=200, 
                                                value=int(ps.get("margin_top", 30)), 
                                                step=5, key=f"{key_prefix}_margin_top")
        with col2:
            ps["margin_right"] = st.number_input("Right margin (px)", min_value=0, max_value=200, 
                                                  value=int(ps.get("margin_right", 20)), 
                                                  step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, 
                                                   value=int(ps.get("margin_bottom", 40)), 
                                                   step=5, key=f"{key_prefix}_margin_bottom")
        st.markdown("---")
        
        # Single checkbox for both plots
        if sim_groups or norm_groups:
            st.markdown("**Per-simulation line styles (optional)**")
            ps["enable_per_sim_style"] = st.checkbox(
                "Enable per-simulation overrides",
                bool(ps.get("enable_per_sim_style", False)),
                key=f"{key_prefix}_enable_per_sim"
            )
        
        if sim_groups:
            st.markdown("**Raw spectra per-simulation styles**")
            render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_raw", 
                                    key_prefix=f"{key_prefix}_raw", include_marker=True, show_enable_checkbox=False)

        if norm_groups:
            st.markdown("**Normalized spectra per-simulation styles**")
            render_per_sim_style_ui(ps, norm_groups, style_key="per_sim_style_norm", 
                                    key_prefix=f"{key_prefix}_norm", include_marker=True, show_enable_checkbox=False)

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
                    f"{key_prefix}_std_alpha",
                    # Colors
                    f"{key_prefix}_palette",
                    f"{key_prefix}_pope_color",
                    f"{key_prefix}_kolmogorov_color",
                    f"{key_prefix}_highlight_color",
                    # Theme
                    f"{key_prefix}_template",
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
                            widget_keys.append(f"{key_prefix}_raw_{suffix}_{sim_prefix}")
                
                if norm_groups:
                    for sim_prefix in norm_groups.keys():
                        for suffix in [
                            "over_on",
                            "over_color",
                            "over_width",
                            "over_dash",
                            "over_marker",
                            "over_msize",
                        ]:
                            widget_keys.append(f"{key_prefix}_norm_{suffix}_{sim_prefix}")
                
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




# ==========================================================
# Main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("📈 Energy Spectra")

    st.session_state.setdefault("spectrum_legend_names", {})
    st.session_state.setdefault("norm_legend_names", {})
    st.session_state.setdefault("axis_labels_raw", {"x": "Wavenumber k", "y": "Energy spectrum E(k)"})
    st.session_state.setdefault("axis_labels_norm", {
        "x": "Normalized wavenumber kη",
        "y": "Normalized spectrum E<sub>norm</sub>(kη)",
    })
    st.session_state.setdefault("plot_styles", {})
    st.session_state.setdefault("file_selection_mode", "directory")
    st.session_state.setdefault("custom_spectrum_files", [])
    st.session_state.setdefault("custom_norm_files", [])

    # File selection mode
    st.sidebar.header("📁 File Selection")
    file_mode = st.sidebar.radio(
        "Selection Mode",
        ["Directory (Auto-detect)", "Custom Files (Any location/name)"],
        index=0 if st.session_state.file_selection_mode == "directory" else 1,
        key="file_mode_radio"
    )
    st.session_state.file_selection_mode = "directory" if file_mode == "Directory (Auto-detect)" else "custom"

    spectrum_files = []
    norm_files = []

    if st.session_state.file_selection_mode == "directory":
        # Check for multiple directories mode
        data_dirs = st.session_state.get("data_directories", [])
        if not data_dirs and st.session_state.get("data_directory"):
            # Fallback to single directory for backward compatibility
            data_dirs = [st.session_state.data_directory]
        
        if not data_dirs:
            st.warning("Please select a data directory from the Overview page.")
            return

        # Load files from all directories
        spectrum_files = []
        norm_files = []
        
        for data_dir_path in data_dirs:
            data_dir = Path(data_dir_path)
            if data_dir.exists():
                dir_spectrum = sorted(glob.glob(str(data_dir / "spectrum*.dat")), key=natural_sort_key)
                dir_norm = sorted(glob.glob(str(data_dir / "norm*.dat")), key=natural_sort_key)
                spectrum_files.extend(dir_spectrum)
                norm_files.extend(dir_norm)

        if not spectrum_files and not norm_files:
            st.error("No spectrum*.dat or norm*.dat files found in the selected directories.")
            st.info("💡 Switch to 'Custom Files' mode to select files from any location, or select multiple directories in the main app.")
            return
        
        # Use first directory for metadata storage
        data_dir = Path(data_dirs[0])
    else:
        # Custom file selection mode
        st.sidebar.markdown("---")
        st.sidebar.subheader("Custom File Selection")
        
        st.sidebar.markdown("**Raw Spectrum Files** (k, E(k)):")
        raw_file_input = st.sidebar.text_area(
            "Enter file paths (one per line):",
            value="\n".join(st.session_state.custom_spectrum_files),
            height=100,
            help="Enter full paths to spectrum files, one per line. Files can be from any directory."
        )
        
        st.sidebar.markdown("**Normalized Spectrum Files** (kη, E_norm, E_pope):")
        norm_file_input = st.sidebar.text_area(
            "Enter file paths (one per line):",
            value="\n".join(st.session_state.custom_norm_files),
            height=100,
            help="Enter full paths to normalized spectrum files, one per line."
        )
        
        if st.sidebar.button("Load Custom Files", type="primary"):
            raw_paths = [p.strip() for p in raw_file_input.strip().split("\n") if p.strip()]
            norm_paths = [p.strip() for p in norm_file_input.strip().split("\n") if p.strip()]
            
            # Validate files exist
            valid_raw = []
            valid_norm = []
            
            for p in raw_paths:
                path = Path(p)
                if path.exists() and path.is_file():
                    valid_raw.append(str(path.absolute()))
                else:
                    st.sidebar.warning(f"File not found: {p}")
            
            for p in norm_paths:
                path = Path(p)
                if path.exists() and path.is_file():
                    valid_norm.append(str(path.absolute()))
                else:
                    st.sidebar.warning(f"File not found: {p}")
            
            st.session_state.custom_spectrum_files = valid_raw
            st.session_state.custom_norm_files = valid_norm
            
            if valid_raw or valid_norm:
                st.sidebar.success(f"Loaded {len(valid_raw)} raw + {len(valid_norm)} normalized files")
            else:
                st.sidebar.error("No valid files found. Check paths.")
        
        spectrum_files = [Path(f) for f in st.session_state.custom_spectrum_files if Path(f).exists()]
        norm_files = [Path(f) for f in st.session_state.custom_norm_files if Path(f).exists()]
        
        if not spectrum_files and not norm_files:
            st.info("👈 Use the sidebar to enter file paths, then click 'Load Custom Files'.")
            return
        
        # For custom mode, use first file's directory as data_dir for metadata storage
        if spectrum_files:
            data_dir = Path(spectrum_files[0]).parent
        elif norm_files:
            data_dir = Path(norm_files[0]).parent
        else:
            data_dir = Path.cwd()

    # Group files by simulation (try pattern matching, fallback to filename-based grouping)
    if st.session_state.file_selection_mode == "directory":
        # When multiple directories are loaded, include directory name in grouping
        if len(data_dirs) > 1:
            # Group by directory name + simulation pattern
            sim_groups = {}
            norm_groups = {}
            
            for data_dir_path in data_dirs:
                data_dir = Path(data_dir_path)
                dir_name = data_dir.name  # e.g., "768", "512", "128"
                
                # Get files from this directory
                dir_spectrum = [f for f in spectrum_files if Path(f).parent == data_dir]
                dir_norm = [f for f in norm_files if Path(f).parent == data_dir]
                
                # Group files from this directory
                dir_sim_groups = group_files_by_simulation(
                    dir_spectrum, r"(spectrum[_\w]*\d+)_\d+\.dat"
                ) if dir_spectrum else {}
                dir_norm_groups = group_files_by_simulation(
                    dir_norm, r"(norm[_\w]*\d+)_\d+\.dat"
                ) if dir_norm else {}
                
                # Fallback patterns
                if not dir_sim_groups and dir_spectrum:
                    dir_sim_groups = group_files_by_simulation(dir_spectrum, r"(spectrum\d+)_\d+\.dat")
                if not dir_norm_groups and dir_norm:
                    dir_norm_groups = group_files_by_simulation(dir_norm, r"(norm\d+)_\d+\.dat")
                
                # Add directory prefix to group keys to distinguish simulations from different directories
                for key, files in dir_sim_groups.items():
                    new_key = f"{dir_name}_{key}" if key else dir_name
                    sim_groups[new_key] = files
                
                for key, files in dir_norm_groups.items():
                    new_key = f"{dir_name}_{key}" if key else dir_name
                    norm_groups[new_key] = files
        else:
            # Single directory - original behavior
            sim_groups = group_files_by_simulation(
                spectrum_files, r"(spectrum[_\w]*\d+)_\d+\.dat"
            ) if spectrum_files else {}
            norm_groups = group_files_by_simulation(
                norm_files, r"(norm[_\w]*\d+)_\d+\.dat"
            ) if norm_files else {}

            if not sim_groups and spectrum_files:
                sim_groups = group_files_by_simulation(spectrum_files, r"(spectrum\d+)_\d+\.dat")
            if not norm_groups and norm_files:
                norm_groups = group_files_by_simulation(norm_files, r"(norm\d+)_\d+\.dat")
    else:
        # Custom mode: group by filename stem (without extension) or directory
        # This allows any filename pattern
        sim_groups = {}
        unique_dirs = set(Path(str(f)).parent for f in spectrum_files)
        for f in spectrum_files:
            fpath = Path(f)
            # Use filename stem as group key, or directory name if files are from different dirs
            if len(unique_dirs) > 1:
                # Files from multiple directories: use directory name as group
                group_key = fpath.parent.name
            else:
                # Files from same directory: use filename stem (without extension)
                group_key = fpath.stem.rsplit("_", 1)[0] if "_" in fpath.stem else fpath.stem
            
            if group_key not in sim_groups:
                sim_groups[group_key] = []
            sim_groups[group_key].append(str(fpath))
        
        # Sort files within each group
        for key in sim_groups:
            sim_groups[key] = sorted(sim_groups[key], key=natural_sort_key)
        
        norm_groups = {}
        for f in norm_files:
            fpath = Path(f)
            unique_dirs = set(Path(str(f)).parent for f in norm_files)
            if len(unique_dirs) > 1:
                group_key = fpath.parent.name
            else:
                stem = fpath.stem
                if "_" in stem:
                    parts = stem.rsplit("_", 1)
                    if parts[1].isdigit():
                        group_key = parts[0]
                    else:
                        group_key = stem
                else:
                    group_key = stem
            
            if group_key not in norm_groups:
                norm_groups[group_key] = []
            norm_groups[group_key].append(str(fpath))
        
        for key in norm_groups:
            norm_groups[key] = sorted(norm_groups[key], key=natural_sort_key)

    if st.session_state.get("_last_legend_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        if "plot_styles" not in st.session_state:
            st.session_state.plot_styles = {}
        st.session_state["_last_legend_dir"] = str(data_dir)

    st.sidebar.header("Options")
    view_mode = st.sidebar.radio("View Mode", ["Time-Averaged", "Time Evolution"], index=0)

    # Legends + axis labels
    if sim_groups or norm_groups:
        with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
            st.markdown("### Legend names")
            if sim_groups:
                st.markdown("**Raw spectra:**")
                for sim_prefix in sorted(sim_groups.keys()):
                    st.session_state.spectrum_legend_names.setdefault(sim_prefix, _default_labelify(sim_prefix))
                    st.session_state.spectrum_legend_names[sim_prefix] = st.text_input(
                        f"Name for `{sim_prefix}`",
                        value=st.session_state.spectrum_legend_names[sim_prefix],
                        key=f"legend_raw_{sim_prefix}"
                    )

            if norm_groups:
                st.markdown("**Normalized spectra:**")
                for norm_prefix in sorted(norm_groups.keys()):
                    st.session_state.norm_legend_names.setdefault(norm_prefix, _default_labelify(norm_prefix))
                    st.session_state.norm_legend_names[norm_prefix] = st.text_input(
                        f"Name for `{norm_prefix}`",
                        value=st.session_state.norm_legend_names[norm_prefix],
                        key=f"legend_norm_{norm_prefix}"
                    )

            st.markdown("---")
            st.markdown("### Axis labels")
            st.caption("Raw spectrum labels")
            st.session_state.axis_labels_raw["x"] = st.text_input(
                "Raw x-axis label",
                value=st.session_state.axis_labels_raw.get("x", "Wavenumber k"),
                key="axis_raw_x"
            )
            st.session_state.axis_labels_raw["y"] = st.text_input(
                "Raw y-axis label",
                value=st.session_state.axis_labels_raw.get("y", "Energy spectrum E(k)"),
                key="axis_raw_y"
            )

            st.caption("Normalized spectrum labels")
            st.session_state.axis_labels_norm["x"] = st.text_input(
                "Norm x-axis label",
                value=st.session_state.axis_labels_norm.get("x", "Normalized wavenumber kη"),
                key="axis_norm_x"
            )
            st.session_state.axis_labels_norm["y"] = st.text_input(
                "Norm y-axis label",
                value=st.session_state.axis_labels_norm.get("y", "Normalized spectrum E<sub>norm</sub>(kη)"),
                key="axis_norm_y"
            )

            st.markdown("---")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("💾 Save labels/legends"):
                    _save_ui_metadata(data_dir)
                    st.success("Saved to legend_names.json")
            with b2:
                if st.button("♻️ Reset labels/legends"):
                    st.session_state.spectrum_legend_names = {k: _default_labelify(k) for k in sim_groups.keys()}
                    st.session_state.norm_legend_names = {k: _default_labelify(k) for k in norm_groups.keys()}
                    st.session_state.axis_labels_raw = {"x": "Wavenumber k", "y": "Energy spectrum E(k)"}
                    st.session_state.axis_labels_norm = {
                        "x": "Normalized wavenumber kη",
                        "y": "Normalized spectrum E<sub>norm</sub>(kη)",
                    }
                    _save_ui_metadata(data_dir)
                    st.toast("Reset + saved.", icon="♻️")

    # Full plot style sidebar (includes backgrounds, ticks, per-sim, highlight)
    plot_names = ["Raw Energy Spectrum", "Normalized Spectrum", "Time Evolution"]
    plot_style_sidebar(data_dir, sim_groups, norm_groups, plot_names)

    # ======================================================
    # TIME-AVERAGED
    # ======================================================
    if view_mode == "Time-Averaged":
        st.header("Time-Averaged Energy Spectra")

        if not sim_groups:
            st.warning("No spectrum*.dat groups found.")
            return

        # Get user-facing names and their corresponding prefixes for Kolmogorov scaling selection
        sim_options = sorted(sim_groups.keys())
        legend_names = st.session_state.spectrum_legend_names
        
        # Create a mapping for display (Name: Prefix)
        display_to_prefix = {
            legend_names.get(prefix, _default_labelify(prefix)): prefix
            for prefix in sim_options
        }
        
        display_names = list(display_to_prefix.keys())
        
        # Use a selectbox to choose which simulation to use for the scaling
        kolm_scaling_name = st.sidebar.selectbox(
            "Scale Kolmogorov Line to Spectra:",
            ["None"] + display_names,
            index=1 if display_names else 0,
            help="Select the simulation curve to scale the k^(-5/3) line against."
        )
        
        if kolm_scaling_name != "None":
            kolm_scaling_prefix = display_to_prefix[kolm_scaling_name]
        else:
            kolm_scaling_prefix = None

        total_files = min(len(g) for g in sim_groups.values())
        start_idx = st.sidebar.slider("Start file index", 1, total_files, min(20, total_files))
        end_idx = st.sidebar.slider("End file index", start_idx, total_files, total_files)

        show_kolm = st.sidebar.checkbox("Show Kolmogorov -5/3 line", value=True)
        error_display = st.sidebar.radio(
            "Error display",
            ["Shaded band", "Error bars", "Both", "None"],
            index=0,
            help="Choose how to display ±1σ uncertainty"
        )
        show_std = error_display in ["Shaded band", "Both"]
        show_error_bars = error_display in ["Error bars", "Both"]
        show_normalized = st.sidebar.checkbox("Show normalized (collapsed) panel with Pope", value=True)

        # Pope model selection for normalized spectrum (related to normalized panel)
        if show_normalized and norm_groups:
            # Get user-facing names and their corresponding prefixes (using normalized spectrum groups)
            norm_options = sorted(norm_groups.keys())
            norm_legend_names = st.session_state.norm_legend_names
            
            # Create a mapping for display (Name: Prefix)
            norm_display_to_prefix = {
                norm_legend_names.get(prefix, _default_labelify(prefix)): prefix
                for prefix in norm_options
            }
            
            norm_display_names = list(norm_display_to_prefix.keys())
            
            # Use a selectbox to choose which simulation's Pope model to display
            pope_scaling_name = st.sidebar.selectbox(
                "Plot Pope Model from Spectra:",
                ["None (Default: Use all plotted simulations)"] + norm_display_names,
                index=0,
                help="If set, only the Pope model corresponding to this selected simulation will be plotted. Otherwise, every plotted simulation's Pope model will be shown."
            )
            
            if pope_scaling_name != "None (Default: Use all plotted simulations)":
                pope_scaling_prefix = norm_display_to_prefix[pope_scaling_name]
            else:
                pope_scaling_prefix = None
        else:
            pope_scaling_prefix = None

        kmin = st.sidebar.number_input("Inertial range k_min", min_value=1.0, value=3.0)
        kmax = st.sidebar.number_input("Inertial range k_max", min_value=kmin + 1e-6, value=20.0)
        
        kolm_scale_factor = st.sidebar.slider(
            "Kolmogorov Line Scale Factor",
            0.1, 5.0,
            1.0,
            step=0.05,
            help="Manually scale the k^(-5/3) line up (> 1.0) or down (< 1.0) for better visual fit."
        )

        # Compute average data for the selected scaling simulation (for Kolmogorov line)
        k_scale, E_avg_scale = None, None
        if kolm_scaling_prefix and show_kolm:
            selected_files_scale = tuple(sim_groups[kolm_scaling_prefix][start_idx-1:end_idx])
            if selected_files_scale:
                k_scale, E_avg_scale, _ = _compute_time_avg(selected_files_scale)

        # Get plot-specific style
        plot_name_raw = "Raw Energy Spectrum"
        ps_raw = get_plot_style(plot_name_raw)
        colors_raw = _get_palette(ps_raw)

        fig_raw = go.Figure()
        plotted_any = False

        for idx, (sim_prefix, files) in enumerate(sorted(sim_groups.items())):
            selected_files = tuple(files[start_idx-1:end_idx])
            if not selected_files:
                continue

            k_vals, E_avg, E_std = _compute_time_avg(selected_files)
            if k_vals is None:
                continue

            color, lw, dash, marker, msize, override_on = resolve_line_style(
                sim_prefix, idx, colors_raw, ps_raw, 
                style_key="per_sim_style_raw",
                include_marker=True,
                default_marker="circle"
            )
            legend_name = st.session_state.spectrum_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
            plotted_any = True

            mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
            trace_kwargs = dict(
                x=k_vals, y=E_avg,
                mode=mode,
                name=legend_name,
                line=dict(color=color, width=lw, dash=dash),
            )
            if override_on and marker and msize > 0:
                trace_kwargs["marker"] = dict(symbol=marker, size=msize)
            if show_error_bars and E_std is not None:
                trace_kwargs["error_y"] = dict(
                    type="data",
                    array=E_std,
                    visible=True,
                    thickness=1,
                    color=color
                )
            fig_raw.add_trace(go.Scatter(**trace_kwargs))

            if show_std:
                rgb = hex_to_rgb(color)
                fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps_raw['std_alpha']})"
                fig_raw.add_trace(go.Scatter(
                    x=np.concatenate([k_vals, k_vals[::-1]]),
                    y=np.concatenate([E_avg - E_std, (E_avg + E_std)[::-1]]),
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip"
                ))

        # Add Kolmogorov line using the selected scaling data (after all traces are added)
        if show_kolm and k_scale is not None and E_avg_scale is not None:
            fig_raw = _add_kolmogorov_line(fig_raw, k_scale, E_avg_scale, kmin, kmax, ps_raw,
                                           kolm_scale_factor=kolm_scale_factor)

        if not plotted_any:
            st.info("No valid spectra could be plotted from selected range.")
            return

        layout_kwargs_raw = dict(
            xaxis_title=st.session_state.axis_labels_raw["x"],
            yaxis_title=st.session_state.axis_labels_raw["y"],
            xaxis_type="log",
            yaxis_type="log",
            legend_title="Simulation",
            height=400,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs_raw = apply_axis_limits(layout_kwargs_raw, ps_raw)
        layout_kwargs_raw = apply_figure_size(layout_kwargs_raw, ps_raw)
        fig_raw.update_layout(**layout_kwargs_raw)
        fig_raw = apply_plot_style(fig_raw, ps_raw)

        fig_norm = None
        if show_normalized and norm_groups:
            # Get plot-specific style
            plot_name_norm = "Normalized Spectrum"
            ps_norm = get_plot_style(plot_name_norm)
            colors_norm = _get_palette(ps_norm)
            
            fig_norm = go.Figure()

            for idx, (norm_prefix, files) in enumerate(sorted(norm_groups.items())):
                selected_files = tuple(files[start_idx-1:end_idx])
                if not selected_files:
                    continue

                keta, En_avg, En_std, Ep_avg = _compute_time_avg_norm(selected_files)
                if keta is None:
                    continue

                color, lw, dash, marker, msize, override_on = resolve_line_style(
                    norm_prefix, idx, colors_norm, ps_norm, 
                    style_key="per_sim_style_norm",
                    include_marker=True,
                    default_marker="circle"
                )
                legend_name = st.session_state.norm_legend_names.get(norm_prefix, _default_labelify(norm_prefix))

                mode = "lines+markers" if (override_on and marker and msize > 0) else "lines"
                trace_kwargs = dict(
                    x=keta, y=En_avg,
                    mode=mode,
                    name=legend_name,
                    line=dict(color=color, width=lw, dash=dash),
                )
                if override_on and marker and msize > 0:
                    trace_kwargs["marker"] = dict(symbol=marker, size=msize)
                if show_error_bars and En_std is not None:
                    trace_kwargs["error_y"] = dict(
                        type="data",
                        array=En_std,
                        visible=True,
                        thickness=1,
                        color=color
                    )
                fig_norm.add_trace(go.Scatter(**trace_kwargs))

                if show_std:
                    rgb = hex_to_rgb(color)
                    fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps_norm['std_alpha']})"
                    fig_norm.add_trace(go.Scatter(
                        x=np.concatenate([keta, keta[::-1]]),
                        y=np.concatenate([En_avg - En_std, (En_avg + En_std)[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

                # Plot the Pope model trace only if:
                # 1. No specific prefix was selected (pope_scaling_prefix is None) OR
                # 2. The current norm_prefix matches the selected scaling prefix
                if pope_scaling_prefix is None or norm_prefix == pope_scaling_prefix:
                    # If we only plot a single selected Pope model, improve its legend name
                    if pope_scaling_prefix is not None:
                        pope_legend_name = f"{legend_name} Pope Model"
                    else:
                        pope_legend_name = f"{legend_name} Pope"
                    
                    fig_norm.add_trace(go.Scatter(
                        x=keta, y=Ep_avg,
                        mode="lines",
                        name=pope_legend_name,
                        line=dict(color=ps_norm["pope_color"], width=ps_norm["line_width"], dash="dash"),
                    ))

            layout_kwargs = dict(
                xaxis_title=st.session_state.axis_labels_norm["x"],
                yaxis_title=st.session_state.axis_labels_norm["y"],
                xaxis_type="log",
                yaxis_type="log",
                legend_title="Simulation",
                width=550,  # Narrower than raw spectrum
                height=600,  # Taller than raw spectrum
            )
            layout_kwargs = apply_axis_limits(layout_kwargs, ps_norm)
            layout_kwargs = apply_figure_size(layout_kwargs, ps_norm)
            fig_norm.update_layout(**layout_kwargs)
            fig_norm = apply_plot_style(fig_norm, ps_norm)

        if fig_norm is not None:
            st.markdown("### Raw Energy Spectrum")
            st.plotly_chart(fig_raw, width='stretch')
            capture_button(fig_raw, title="Energy Spectra (Raw)", source_page="Energy Spectra")
            st.markdown("### Normalized (Collapsed) Spectrum")
            st.plotly_chart(fig_norm, width='stretch')
            capture_button(fig_norm, title="Energy Spectra (Normalized)", source_page="Energy Spectra")
        else:
            st.plotly_chart(fig_raw, width='stretch')
            capture_button(fig_raw, title="Energy Spectra", source_page="Energy Spectra")

        st.subheader("Export")
        export_panel(fig_raw, data_dir, "energy_spectra_raw")
        if fig_norm is not None:
            export_panel(fig_norm, data_dir, "energy_spectra_normalized")

        # Averaged CSV export
        first_sim = list(sorted(sim_groups.keys()))[0]
        files0 = tuple(sim_groups[first_sim][start_idx-1:end_idx])
        k0, E0, S0 = _compute_time_avg(files0)
        if k0 is not None:
            import pandas as pd
            df_out = pd.DataFrame({"k": k0, "E_avg": E0, "E_std": S0})
            st.download_button(
                "Download averaged CSV",
                df_out.to_csv(index=False).encode("utf-8"),
                file_name="energy_spectrum_avg.csv",
                mime="text/csv"
            )

    # ======================================================
    # TIME EVOLUTION
    # ======================================================
    else:
        st.header("Time Evolution of Energy Spectra")

        if not sim_groups:
            st.error("Time evolution requires spectrum*.dat files.")
            return

        sim_names = sorted(sim_groups.keys())
        sim_display_names = [_default_labelify(n) for n in sim_names]
        sim_sel_display = st.sidebar.selectbox("Simulation group", sim_display_names, index=0)
        sim_sel = sim_names[sim_display_names.index(sim_sel_display)]

        files = sim_groups[sim_sel]
        iters = [_extract_iter(f) for f in files]
        if all(i is None for i in iters):
            st.error("Could not extract iteration numbers from filenames.")
            return

        every_n = st.sidebar.slider("Show every Nth iteration curve", 1, min(50, len(files)), 5)
        thin_idx = list(range(0, len(files), every_n))
        thin_files = [files[i] for i in thin_idx]
        thin_iters = [iters[i] for i in thin_idx]

        sel_pos = st.sidebar.slider("Highlight curve (thinned index)", 0, len(thin_files)-1, len(thin_files)-1)
        highlight_file = thin_files[sel_pos]
        highlight_iter = thin_iters[sel_pos]

        # Get plot-specific style
        plot_name_evol = "Time Evolution"
        ps_evol = get_plot_style(plot_name_evol)

        figE = go.Figure()

        for f, it in zip(thin_files, thin_iters):
            try:
                k, E = _read_spectrum_cached(str(f))
            except Exception:
                continue
            figE.add_trace(go.Scatter(
                x=k, y=E,
                mode="lines",
                line=dict(width=max(1.0, ps_evol["line_width"] * 0.6)),
                opacity=0.25,
                showlegend=False,
            ))

        try:
            kH, EH = _read_spectrum_cached(str(highlight_file))
            figE.add_trace(go.Scatter(
                x=kH, y=EH,
                mode="lines",
                name=f"Highlighted iter {highlight_iter}",
                line=dict(width=ps_evol["line_width"] * 1.2, color=ps_evol.get("highlight_color", "#E41A1C")),
                opacity=1.0
            ))
        except Exception as e:
            st.warning(f"Highlight read failed: {e}")

        layout_kwargs_evol = dict(
            xaxis_title=st.session_state.axis_labels_raw["x"],
            yaxis_title=st.session_state.axis_labels_raw["y"],
            xaxis_type="log",
            yaxis_type="log",
            width=850,  # Width less than height
            height=550,  # Reduced for time evolution figure
        )
        layout_kwargs_evol = apply_axis_limits(layout_kwargs_evol, ps_evol)
        # Don't apply figure size override for time evolution to keep fixed width
        figE.update_layout(**layout_kwargs_evol)
        figE = apply_plot_style(figE, ps_evol)

        st.plotly_chart(figE, width='content')
        capture_button(figE, title=f"Energy Spectra Time Evolution - {sim_sel}", source_page="Energy Spectra")

        st.subheader("Export time evolution figure")
        export_panel(figE, data_dir, f"energy_spectra_time_evolution_{sim_sel}")

    # ======================================================
    # Theory / Equations
    # ======================================================
    with st.expander("Theory / Equations", expanded=False):
        st.markdown("**3D kinetic energy spectrum (Fourier space)**")
        st.latex(r"E(\kappa)=\sum_{\kappa\le |\mathbf{k}|<\kappa+\Delta \kappa} \frac{1}{2}\left(|\hat{u}(\mathbf{k})|^2+|\hat{v}(\mathbf{k})|^2+|\hat{w}(\mathbf{k})|^2\right)")
        
        st.markdown("**Total kinetic energy and RMS velocity**")
        st.latex(r"\mathrm{TKE}=\sum_{\kappa}E(\kappa), \qquad u_{\mathrm{rms}}=\sqrt{\frac{2}{3}\mathrm{TKE}}")

        st.markdown("**Kolmogorov inertial-range scaling**")
        st.latex(r"E(\kappa)\propto \kappa^{-5/3}")

        st.markdown("**Pope model spectrum (HIT validation)**")
        st.latex(r"E_{\text{pope}}(\kappa)=C\,\varepsilon^{2/3}\kappa^{-5/3}f_L(\kappa L)f_\eta(\kappa\eta)")
        st.markdown("with $C=1.5$, $c_L=6.78$, $c_\eta=0.40$, $\\beta=5.2$.")


if __name__ == "__main__":
    main()

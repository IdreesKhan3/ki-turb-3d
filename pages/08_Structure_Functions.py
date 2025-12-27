"""
Structure Functions Page (Streamlit)
ESS analysis + scaling exponents + full persistent UI controls

Features:
- Reads structure functions from:
    * Binary: structure_funcs*_t*.bin  (primary)
    * Text (optional if available): structure_functions*_t*.txt
- Groups by simulation prefix (structure_funcs1, structure_funcs2, ...)
- Time-averages over selected time window
- Plots:
    1) S_p(r) vs r (orders selectable)
    2) ESS: S_p vs S_ref (ref order selectable)
    3) Optional anomalies plot (xi_p - p/3) with SL94 + experimental overlay
- Computes ESS scaling exponents xi_p with user-fit range control
- FULL user controls (in-memory session state):
    * Legends, axis labels
    * Fonts, tick style, major/minor grids, background colors, theme
    * Palette / custom colors
    * Per-simulation overrides: color/width/dash/marker/marker size
- Research-grade export:
    * User can export to: PNG, PDF, SVG, JPG/JPEG, WEBP
    * scale (DPI-like), width/height override

Requires kaleido for static export:
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
import glob

# --- Project imports ---
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
from utils.file_detector import (
    detect_simulation_files,
    group_files_by_simulation,
    natural_sort_key
)
from utils.report_builder import capture_button
from utils.export_figs import export_panel
from utils.plot_style import (
    default_plot_style, apply_plot_style as apply_plot_style_base,
    render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size,
    render_axis_scale_ui, render_tick_format_ui, render_axis_borders_ui,
    render_plot_title_ui, _get_palette, _normalize_plot_name,
    resolve_line_style, render_per_sim_style_ui, ensure_per_sim_defaults, convert_superscript
)
from pages.StructureFunctions.ess_inset import add_ess_inset

# Binary/text readers (binary is required by plan, text is optional)
from data_readers.binary_reader import read_structure_function_file
st.set_page_config(page_icon="⚫")
try:
    from data_readers.text_reader import read_structure_function_txt
except Exception:
    read_structure_function_txt = None


# ==========================================================
# Theory curves
# ==========================================================
def zeta_p_she_leveque(p):
    return p/9 + 2*(1 - (2/3)**(p/3))

TABLE_P = [2, 3, 4, 5, 6]
EXP_ZETA = [0.71, 1.00, 1.28, 1.53, 1.78]


# ==========================================================
# Cached readers / averaging
# ==========================================================
@st.cache_data(show_spinner=False)
def _read_structure_bin_cached(fname: str):
    return read_structure_function_file(fname)

@st.cache_data(show_spinner=False)
def _read_structure_txt_cached(fname: str):
    if read_structure_function_txt is None:
        raise RuntimeError("Text reader not available.")
    return read_structure_function_txt(fname)

def _extract_iter(fname: str):
    stem = Path(fname).stem
    nums = re.findall(r"(\d+)", stem)
    return int(nums[-1]) if nums else None

@st.cache_data(show_spinner=False)
def _compute_time_avg_structure(files: tuple, kind: str):
    """
    Time-average structure functions over selected files.
    Matches auxiliary script approach: sum S_p, r, u_rms then divide by num_files.
    Assumes all files from same simulation have same r grid (physically correct).
    kind: "bin" or "txt".
    Returns r, S_p_mean dict, S_p_std dict, u_rms_mean, ps(list)
    """
    sum_sp = None
    sum_r = None
    total_u_rms = 0.0
    num_files = 0
    ps = None
    max_dr = None

    for f in files:
        try:
            data = _read_structure_bin_cached(str(f)) if kind == "bin" else _read_structure_txt_cached(str(f))
        except Exception as e:
            # Silently skip files that can't be read (may be corrupted or wrong format)
            continue

        r = np.asarray(data.get("r", []), float)
        S_p = data.get("S_p", {})
        if r.size == 0 or not S_p:
            continue

        # Initialize on first file (assumes all files have same structure)
        if sum_sp is None:
            max_dr = len(r)
            ps = sorted(S_p.keys())
            sum_sp = {p: np.zeros(max_dr, dtype=float) for p in ps}
            sum_r = np.zeros(max_dr, dtype=float)

        # Sum S_p and r (matching auxiliary script approach)
        for p in ps:
            if p in S_p:
                sp_arr = np.asarray(S_p[p], float)
                min_len = min(len(sum_sp[p]), len(sp_arr))
                sum_sp[p][:min_len] += sp_arr[:min_len]
        
        min_r_len = min(len(sum_r), len(r))
        sum_r[:min_r_len] += r[:min_r_len]
        total_u_rms += float(data.get("u_rms", 0.0))
        num_files += 1

    if num_files == 0 or sum_sp is None:
        return None, None, None, None, None

    # Average by dividing by num_files (matching auxiliary script)
    r_mean = sum_r / num_files
    Sp_mean_dict = {p: sum_sp[p] / num_files for p in ps}
    u_rms_mean = total_u_rms / num_files

    # Compute std for error bars (not in auxiliary script, but useful for Streamlit)
    # Re-read files to compute std
    Sp_list = []
    for f in files:
        try:
            data = _read_structure_bin_cached(str(f)) if kind == "bin" else _read_structure_txt_cached(str(f))
            S_p = data.get("S_p", {})
            if S_p:
                Sp_mat = np.vstack([np.asarray(S_p[p], float)[:max_dr] for p in ps])
                Sp_list.append(Sp_mat)
        except Exception:
            continue
    
    if Sp_list:
        Sp_arr = np.stack(Sp_list, axis=0)
        Sp_std = np.std(Sp_arr, axis=0)
        # Map p values to their index in ps (not p-1, which assumes consecutive 1,2,3,...)
        Sp_std_dict = {p: Sp_std[idx, :] for idx, p in enumerate(ps)}
    else:
        Sp_std_dict = {p: np.zeros(max_dr) for p in ps}

    return r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, list(ps)


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
            l=ps.get("margin_left", 50),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 30),
            b=ps.get("margin_bottom", 50)
        ))
    
    # Always set title with correct font color if show_plot_title is True
    if ps.get("show_plot_title", False) and ps.get("plot_title"):
        fig.update_layout(title=_get_title_dict(ps, ps["plot_title"]))
    
    return fig

def _normalize_plot_name_local(plot_name: str) -> str:
    """Normalize plot name to a valid key format (handles special characters)."""
    # First replace special characters, then use the centralized function
    cleaned = plot_name.replace("ₚ", "p").replace("ξ", "xi").replace("/", "_")
    return _normalize_plot_name(cleaned)

def get_plot_style(plot_name: str):
    """Get plot-specific style, merging defaults with plot-specific overrides."""
    default = default_plot_style()
    default.update({
        "line_width": 2.4,
        "marker_size": 6,
        "margin_left": 50,
        "margin_right": 20,
        "margin_top": 30,
        "margin_bottom": 50,
        "std_alpha": 0.18,
        "per_sim_style_structure": {},
        "she_leveque_color": "#000000",  # Default black for light theme
        "experimental_b93_color": "#00BFC4",  # Default cyan for light theme
    })
    
    # Set default axis types based on plot
    if plot_name == "Anomalies (ξₚ − p/3)" or plot_name == "ESS Inset":
        default.update({
            "x_axis_type": "linear",
            "y_axis_type": "linear",
        })
    else:
        # Default to log for S_p and ESS plots
        default.update({
            "x_axis_type": "log",
            "y_axis_type": "log",
        })
    
    plot_styles = st.session_state.get("plot_styles", {})
    plot_style = plot_styles.get(plot_name, {})
    
    # Apply theme first to get theme defaults
    current_theme = st.session_state.get("theme", "Light Scientific")
    merged = default.copy()
    merged = apply_theme_to_plot_style(merged, current_theme)
    
    # Store theme-determined properties before applying user overrides
    theme_plot_bgcolor = merged["plot_bgcolor"]
    theme_paper_bgcolor = merged["paper_bgcolor"]
    theme_font_color = merged.get("font_color")
    theme_axis_line_color = merged.get("axis_line_color")
    
    # Then apply user overrides (from plot_style) - this ensures user settings override theme
    for key, value in plot_style.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merged[key].copy()
            merged[key].update(value)
        else:
            merged[key] = value
    
    # Update theme-dependent colors if using default values
    if "plot_bgcolor" in plot_style:
        if plot_style["plot_bgcolor"] in ["#1e1e1e", "#FFFFFF", "#F5F5F5"]:
            merged["plot_bgcolor"] = theme_plot_bgcolor
    else:
        merged["plot_bgcolor"] = theme_plot_bgcolor
    
    if "paper_bgcolor" in plot_style:
        if plot_style["paper_bgcolor"] in ["#1e1e1e", "#FFFFFF", "#F5F5F5"]:
            merged["paper_bgcolor"] = theme_paper_bgcolor
    else:
        merged["paper_bgcolor"] = theme_paper_bgcolor
    
    if "font_color" in plot_style:
        if plot_style["font_color"] in [None, "#000000", "#d4d4d4", "#FFFFFF"]:
            merged["font_color"] = theme_font_color
    else:
        merged["font_color"] = theme_font_color
    
    if "axis_line_color" in plot_style:
        if plot_style["axis_line_color"] in ["#000000", "#FFFFFF", "#d4d4d4"]:
            merged["axis_line_color"] = theme_axis_line_color
    else:
        merged["axis_line_color"] = theme_axis_line_color
    
    # Tick color defaults to axis_line_color if not explicitly set
    if "tick_color" not in plot_style or plot_style.get("tick_color") is None:
        merged["tick_color"] = None  # Will use axis_line_color in apply_plot_style
    
    # Update She-Leveque and Experimental B93 curve colors for dark theme if they're still at light theme defaults
    if "Dark" in current_theme and (plot_name == "Anomalies (ξₚ − p/3)" or plot_name == "ESS Inset"):
        if merged.get("she_leveque_color") == "#000000":
            merged["she_leveque_color"] = "#569cd6"  # Blue - visible on dark background
        if merged.get("experimental_b93_color") == "#00BFC4":
            merged["experimental_b93_color"] = "#4ec9b0"  # Cyan/turquoise - visible on dark background
    
    return merged

def plot_style_sidebar(data_dir: Path, sim_groups, plot_names: list):
    # Plot selector
    selected_plot = st.sidebar.selectbox(
        "Select plot to configure",
        plot_names,
        key="structure_plot_selector"
    )
    
    # Get or create plot-specific style
    if "plot_styles" not in st.session_state:
        st.session_state.plot_styles = {}
    if selected_plot not in st.session_state.plot_styles:
        st.session_state.plot_styles[selected_plot] = {}
    
    # Start with defaults, merge with plot-specific overrides
    ps = get_plot_style(selected_plot)
    plot_key = _normalize_plot_name_local(selected_plot)
    
    # Ensure per-sim defaults
    ensure_per_sim_defaults(ps, sim_groups, style_key="per_sim_style_structure", include_marker=True)
    
    # Create unique key prefix for all widgets
    key_prefix = f"structure_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        font_idx = fonts.index(ps.get("font_family", "Arial")) if ps.get("font_family", "Arial") in fonts else 0
        ps["font_family"] = st.selectbox("Font family", fonts, index=font_idx,
                                         key=f"{key_prefix}_font_family")
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
        ps["marker_size"] = st.slider("Global marker size", 0, 18, int(ps.get("marker_size", 6)),
                                       key=f"{key_prefix}_marker_size")
        ps["std_alpha"] = st.slider("Std band opacity", 0.05, 0.6, float(ps.get("std_alpha", 0.18)),
                                    key=f"{key_prefix}_std_alpha")

        # Reference line colors for Anomalies plot and ESS Inset
        if selected_plot == "Anomalies (ξₚ − p/3)" or selected_plot == "ESS Inset":
            ps["she_leveque_color"] = st.color_picker(
                "She–Leveque curve color",
                ps.get("she_leveque_color", "#000000"),
                key=f"{key_prefix}_she_leveque_color"
            )
            ps["experimental_b93_color"] = st.color_picker(
                "Experimental B93 color",
                ps.get("experimental_b93_color", "#00BFC4"),
                key=f"{key_prefix}_experimental_b93_color"
            )
        
        # Zero line color for ESS Inset
        if selected_plot == "ESS Inset":
            ps["zero_line_color"] = st.color_picker(
                "Zero line (Kolmogorov reference) color",
                ps.get("zero_line_color", ps.get("axis_line_color", "#000000")),
                key=f"{key_prefix}_zero_line_color"
            )
            ps["zero_line_width"] = st.slider(
                "Zero line width",
                0.5, 3.0,
                float(ps.get("zero_line_width", ps.get("axis_line_width", 0.8))),
                key=f"{key_prefix}_zero_line_width"
            )

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
                                                  value=int(ps.get("margin_right", 20)), 
                                                  step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, 
                                                   value=int(ps.get("margin_bottom", 50)), 
                                                   step=5, key=f"{key_prefix}_margin_bottom")
        st.markdown("---")
        render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_structure", 
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
                    # Reference line colors (for Anomalies plot and ESS Inset)
                    f"{key_prefix}_she_leveque_color",
                    f"{key_prefix}_experimental_b93_color",
                    # Zero line (for ESS Inset)
                    f"{key_prefix}_zero_line_color",
                    f"{key_prefix}_zero_line_width",
                    # Colors
                    f"{key_prefix}_palette",
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
            
            st.toast(f"Reset style for '{selected_plot}'.")
            reset_pressed = True
            st.rerun()

    # Auto-save plot style changes (applies immediately) - but not if reset was pressed
    if not reset_pressed:
        # Make a copy to avoid modifying the original dict reference
        ps_copy = ps.copy()
        # Deep copy nested dicts
        for key, value in ps.items():
            if isinstance(value, dict):
                ps_copy[key] = value.copy()
        st.session_state.plot_styles[selected_plot] = ps_copy


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
# Page main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("Structure Functions")

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

    # Defaults
    st.session_state.setdefault("structure_legend_names", {})
    st.session_state.setdefault("axis_labels_structure", {
        "x_r": "Separation distance r",
        "y_sp": "Structure functions S<sub>p</sub>(r)",
        "x_ess": "S<sub>3</sub>(r)",
        "y_ess": "S<sub>p</sub>(r)",
        "x_anom": "p",
        "y_anom": "ξ<sub>p</sub> - p/3",
        "x_inset": "p",
        "y_inset": "ξ<sub>p</sub> - p/3",
        "inset_legend_sl": "SL94",
        "inset_legend_b93": "B93",
    })
    st.session_state.setdefault("plot_styles", {})


    # Collect files from all directories
    all_bin_files = []
    all_txt_files = []
    
    for data_dir_path in data_dirs:
        # Resolve path to ensure it works regardless of how it was stored
        try:
            data_dir_obj = Path(data_dir_path).resolve()
            if data_dir_obj.exists() and data_dir_obj.is_dir():
                # Process each directory independently
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_bin = files_dict.get("structure_functions_bin", [])
                dir_txt = files_dict.get("structure_functions_txt", [])
                all_bin_files.extend([str(f) for f in dir_bin])
                all_txt_files.extend([str(f) for f in dir_txt])
        except Exception:
            continue  # Skip invalid directories
    
    if not all_bin_files and not all_txt_files:
        st.info("No structure function files found. Expected `structure_funcs*_t*.bin` or `structure_functions*_t*.txt`.")
        return

    # Group by simulation prefix, with directory name when multiple directories
    if len(data_dirs) > 1:
        # Multiple directories: group by directory name + simulation pattern
        sim_groups_bin = {}
        sim_groups_txt = {}
        
        for data_dir_path in data_dirs:
            data_dir_obj = Path(data_dir_path).resolve()
            dir_name = data_dir_obj.name
            
            # Get files from this directory - use string comparison for robustness
            data_dir_str = str(data_dir_obj)
            dir_bin = [f for f in all_bin_files if str(Path(f).resolve().parent) == data_dir_str]
            dir_txt = [f for f in all_txt_files if str(Path(f).resolve().parent) == data_dir_str]
            
            # If no files found, re-check this directory directly (in case they weren't in all_bin_files)
            if not dir_bin and not dir_txt:
                files_dict = detect_simulation_files(str(data_dir_obj))
                dir_bin = [str(f) for f in files_dict.get("structure_functions_bin", [])]
                dir_txt = [str(f) for f in files_dict.get("structure_functions_txt", [])]
            
            if dir_bin:
                # Try pattern with number: structure_funcs1_t*.bin
                dir_sim_groups_bin = group_files_by_simulation(
                    sorted([str(f) for f in dir_bin], key=natural_sort_key),
                    r"(structure_funcs\d+)_t\d+\.bin"
                )
                # If that fails, try pattern with data: structure_funcs_data4_t*.bin
                if not dir_sim_groups_bin:
                    dir_sim_groups_bin = group_files_by_simulation(
                        sorted([str(f) for f in dir_bin], key=natural_sort_key),
                        r"(structure_funcs_data\d+)_t\d+\.bin"
                    )
                if dir_sim_groups_bin:
                    # Files matched pattern - use pattern-based grouping
                    for key, files in dir_sim_groups_bin.items():
                        new_key = f"{dir_name}_{key}" if key else dir_name
                        sim_groups_bin[new_key] = files
                else:
                    # Files didn't match pattern - treat entire directory as one simulation
                    sim_groups_bin[dir_name] = sorted([str(f) for f in dir_bin], key=natural_sort_key)
            
            if dir_txt:
                dir_sim_groups_txt = group_files_by_simulation(
                    sorted([str(f) for f in dir_txt], key=natural_sort_key),
                    r"(structure_functions\d+)_t\d+\.txt"
                )
                if dir_sim_groups_txt:
                    # Files matched pattern - use pattern-based grouping
                    for key, files in dir_sim_groups_txt.items():
                        new_key = f"{dir_name}_{key}" if key else dir_name
                        sim_groups_txt[new_key] = files
                else:
                    # Files didn't match pattern - treat entire directory as one simulation
                    sim_groups_txt[dir_name] = sorted([str(f) for f in dir_txt], key=natural_sort_key)
    else:
        # Single directory - group by simulation prefix
        sim_groups_bin = group_files_by_simulation(
            sorted([str(f) for f in all_bin_files], key=natural_sort_key),
            r"(structure_funcs\d+)_t\d+\.bin"
        ) if all_bin_files else {}
        # If that fails, try pattern with data: structure_funcs_data4_t*.bin
        if not sim_groups_bin and all_bin_files:
            sim_groups_bin = group_files_by_simulation(
                sorted([str(f) for f in all_bin_files], key=natural_sort_key),
                r"(structure_funcs_data\d+)_t\d+\.bin"
            )
        
        sim_groups_txt = group_files_by_simulation(
            sorted([str(f) for f in all_txt_files], key=natural_sort_key),
            r"(structure_functions\d+)_t\d+\.txt"
        ) if all_txt_files else {}
        
        # If grouping failed in single directory, treat all files as one simulation
        if not sim_groups_bin and not sim_groups_txt:
            if all_bin_files:
                sim_groups_bin["structure_funcs"] = sorted([str(f) for f in all_bin_files], key=natural_sort_key)
            elif all_txt_files:
                sim_groups_txt["structure_funcs"] = sorted([str(f) for f in all_txt_files], key=natural_sort_key)

    # Combine binary and text groups
    sim_groups = {}
    for k, v in sim_groups_bin.items():
        sim_groups[k] = {"kind": "bin", "files": v}
    for k, v in sim_groups_txt.items():
        if k not in sim_groups:
            sim_groups[k] = {"kind": "txt", "files": v}
    
    if not sim_groups:
        st.error("No structure function files found or could not group files.")
        return

    # Sidebar time window
    st.sidebar.subheader("Time Window")
    file_lengths = {k: len(v["files"]) for k, v in sim_groups.items()}
    if not file_lengths:
        st.error("No files found in any simulation group.")
        return
    min_len = min(file_lengths.values())
    
    # Show file counts per simulation (helpful for debugging)
    if len(sim_groups) > 1:
        with st.sidebar.expander("File counts", expanded=False):
            for sim_prefix in sorted(file_lengths.keys()):
                count = file_lengths[sim_prefix]
                st.text(f"{sim_prefix}: {count} files")
    
    start_idx = st.sidebar.slider("Start file index", 1, min_len, 1)
    end_idx = st.sidebar.slider("End file index", start_idx, min_len, min_len)

    # Sidebar data options
    st.sidebar.subheader("Orders / Normalization")
    sample_key = sorted(sim_groups.keys())[0]
    sample_files = tuple(sim_groups[sample_key]["files"][start_idx-1:end_idx])
    r_s, Sp_m_s, Sp_sd_s, urms_s, ps_list = _compute_time_avg_structure(sample_files, sim_groups[sample_key]["kind"])
    if ps_list is None:
        st.error("Could not read structure function data from the selected range.")
        return

    max_p = max(ps_list)
    selected_ps = st.sidebar.multiselect(
        "Orders p to plot (S_p and ESS)",
        options=list(range(1, max_p + 1)),
        default=list(range(1, min(7, max_p + 1)))
    )
    ref_p = st.sidebar.selectbox(
        "ESS reference order (x-axis)",
        options=ps_list,
        index=ps_list.index(3) if 3 in ps_list else 0
    )
    normalize_by_urms = st.sidebar.checkbox("Normalize S_p by u_rms^p", value=True)

    st.sidebar.subheader("Error band / Theory")
    error_display = st.sidebar.radio(
        "Error display",
        ["Shaded band", "Error bars", "Both", "None"],
        index=0,
        help="Choose how to display ±1σ uncertainty (applies to both S_p and ESS plots)"
    )
    show_std_band = error_display in ["Shaded band", "Both"]
    show_error_bars = error_display in ["Error bars", "Both"]
    show_sl_theory = st.sidebar.checkbox("Show She-Leveque anomalies", value=True)
    show_exp_anom = st.sidebar.checkbox("Show experimental anomalies (B93)", value=True)
    show_inset = st.sidebar.checkbox("Show ESS inset (anomalies)", value=True)

    st.sidebar.subheader("Fit range for ESS exponents")
    if r_s is not None and np.any(r_s > 0):
        r_pos = r_s[r_s > 0]
        r_min_default = float(np.percentile(r_pos, 10))
        r_max_default = float(np.percentile(r_pos, 60))
    else:
        r_min_default, r_max_default = 1e-3, 1e-1

    fit_rmin = st.sidebar.number_input("Fit r_min", value=r_min_default, min_value=0.0, format="%.6g")
    fit_rmax = st.sidebar.number_input("Fit r_max", value=r_max_default, min_value=fit_rmin + 1e-12, format="%.6g")

    # Sidebar legends + axis labels (persistent)
    with st.sidebar.expander("🏷️ Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        for sim_prefix in sorted(sim_groups.keys()):
            st.session_state.structure_legend_names.setdefault(sim_prefix, sim_prefix.replace("_", " ").title())
            st.session_state.structure_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.structure_legend_names[sim_prefix],
                key=f"legend_struct_{sim_prefix}"
            )

        st.markdown("---")
        st.markdown("### Axis labels")
        st.session_state.axis_labels_structure["x_r"] = st.text_input(
            "S_p plot x-label", st.session_state.axis_labels_structure.get("x_r", "Separation distance r"), key="ax_struct_xr"
        )
        st.session_state.axis_labels_structure["y_sp"] = st.text_input(
            "S_p plot y-label", st.session_state.axis_labels_structure.get("y_sp", "Structure functions S<sub>p</sub>(r)"), key="ax_struct_ysp"
        )
        st.session_state.axis_labels_structure["x_ess"] = st.text_input(
            "ESS x-label", st.session_state.axis_labels_structure.get("x_ess", "S<sub>3</sub>(r)"), key="ax_struct_xess"
        )
        st.session_state.axis_labels_structure["y_ess"] = st.text_input(
            "ESS y-label", st.session_state.axis_labels_structure.get("y_ess", "S<sub>p</sub>(r)"), key="ax_struct_yess"
        )
        st.session_state.axis_labels_structure["x_anom"] = st.text_input(
            "Anomaly x-label", st.session_state.axis_labels_structure.get("x_anom", "p"), key="ax_struct_xanom"
        )
        st.session_state.axis_labels_structure["y_anom"] = st.text_input(
            "Anomaly y-label", st.session_state.axis_labels_structure.get("y_anom", "ξ<sub>p</sub> - p/3"), key="ax_struct_yanom"
        )
        
        st.markdown("---")
        st.markdown("### Inset labels")
        st.session_state.axis_labels_structure["x_inset"] = st.text_input(
            "Inset x-label", st.session_state.axis_labels_structure.get("x_inset", "p"), key="ax_struct_x_inset"
        )
        st.session_state.axis_labels_structure["y_inset"] = st.text_input(
            "Inset y-label", st.session_state.axis_labels_structure.get("y_inset", "ξ<sub>p</sub> - p/3"), key="ax_struct_y_inset"
        )
        st.session_state.axis_labels_structure["inset_legend_sl"] = st.text_input(
            "Inset She-Leveque legend", st.session_state.axis_labels_structure.get("inset_legend_sl", "SL94"), key="ax_struct_legend_sl"
        )
        st.session_state.axis_labels_structure["inset_legend_b93"] = st.text_input(
            "Inset B93 legend", st.session_state.axis_labels_structure.get("inset_legend_b93", "B93"), key="ax_struct_legend_b93"
        )

        if st.button("♻️ Reset labels/legends"):
            st.session_state.structure_legend_names = {k: k.replace("_", " ").title() for k in sim_groups.keys()}
            st.session_state.axis_labels_structure.update({
                "x_r": "Separation distance r",
                "y_sp": "Structure functions S<sub>p</sub>(r)",
                "x_ess": "S<sub>3</sub>(r)",
                "y_ess": "S<sub>p</sub>(r)",
                "x_anom": "p",
                "y_anom": "ξ<sub>p</sub> - p/3",
                "x_inset": "p",
                "y_inset": "ξ<sub>p</sub> - p/3",
                "inset_legend_sl": "SL94",
                "inset_legend_b93": "B93",
            })
            st.toast("Reset.")
            st.rerun()

    # Full style sidebar
    plot_names = ["S_p(r) vs r", "ESS (S_p vs S_3)", "ESS Inset", "Anomalies (ξₚ − p/3)"]
    plot_style_sidebar(data_dir, sim_groups, plot_names)

    tabs = st.tabs(["Sₚ(r) vs r", "ESS (Sₚ vs S₃)", "Scaling Exponents Table"])

    # ============================================
    # Tab 1: S_p(r) vs r
    # ============================================
    with tabs[0]:
        st.subheader("Time-averaged Structure Functions")
        
        # Get plot-specific style
        plot_name_sp = "S_p(r) vs r"
        ps_sp = get_plot_style(plot_name_sp)
        colors_sp = _get_palette(ps_sp)
        
        fig_sp = go.Figure()
        plotted_any = False

        for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
            kind = sim_groups[sim_prefix]["kind"]
            files = sim_groups[sim_prefix]["files"][start_idx-1:end_idx]
            if not files:
                st.warning(f"No files found for {sim_prefix} in selected time range.")
                continue
            r, Sp_mean, Sp_std, urms, ps_here = _compute_time_avg_structure(tuple(files), kind)
            if r is None:
                st.warning(f"Could not read structure function data for {sim_prefix}. Check file format.")
                continue
            if not Sp_mean:
                st.warning(f"No structure function data found for {sim_prefix}.")
                continue

            legend_base = st.session_state.structure_legend_names.get(sim_prefix, sim_prefix.replace("_", " ").title())
            color_base, lw_base, dash_base, marker_base, msize_base, override_on = resolve_line_style(
                sim_prefix, idx, colors_sp, ps_sp,
                style_key="per_sim_style_structure",
                include_marker=True,
                default_marker="circle"
            )
            plotted_any = True

            for j, p in enumerate(selected_ps):
                if p not in Sp_mean:
                    continue
                y = Sp_mean[p]
                ystd = Sp_std[p]

                if normalize_by_urms and np.isfinite(urms):
                    y = y / (urms ** p)
                    ystd = ystd / (urms ** p)

                # Use same color for all orders of the same simulation (like ESS plot)
                line_color = color_base
                
                # Determine mode and marker based on override
                if override_on and marker_base and msize_base > 0:
                    mode = "lines+markers"
                    marker_dict = dict(symbol=marker_base, size=msize_base)
                else:
                    mode = "lines"
                    marker_dict = None

                trace_kwargs = dict(
                    x=r, y=y,
                    mode=mode,
                    name=f"{legend_base}  (p={p})",
                    line=dict(color=line_color, width=lw_base, dash=dash_base),
                    hovertemplate="r=%{x:.3g}<br>S_p=%{y:.3g}<extra></extra>"
                )
                if marker_dict:
                    trace_kwargs["marker"] = marker_dict
                if show_error_bars and ystd is not None:
                    trace_kwargs["error_y"] = dict(
                        type="data",
                        array=ystd,
                        visible=True,
                        thickness=1,
                        color=line_color
                    )
                fig_sp.add_trace(go.Scatter(**trace_kwargs))

                if show_std_band and ystd is not None:
                    rgb = _color_to_rgb_tuple(line_color)
                    fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps_sp['std_alpha']})"
                    fig_sp.add_trace(go.Scatter(
                        x=np.concatenate([r, r[::-1]]),
                        y=np.concatenate([y - ystd, (y + ystd)[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

        if not plotted_any:
            st.info("No valid structure function data in selected range.")
        else:
            layout_kwargs = dict(
                xaxis_title=st.session_state.axis_labels_structure.get("x_r", "Separation distance r"),
                yaxis_title=st.session_state.axis_labels_structure.get("y_sp", "Structure functions S<sub>p</sub>(r)"),
                legend_title="Simulation / Order",
                height=500,  # Default, will be overridden if custom size is enabled
            )
            layout_kwargs = apply_axis_limits(layout_kwargs, ps_sp)
            layout_kwargs = apply_figure_size(layout_kwargs, ps_sp)
            fig_sp.update_layout(**layout_kwargs)
            fig_sp = apply_plot_style(fig_sp, ps_sp)
            st.plotly_chart(fig_sp, width='stretch')
            capture_button(fig_sp, title="Structure Functions S_p(r)", source_page="Structure Functions")
            export_panel(fig_sp, data_dir, base_name="structure_functions_sp")

    # ============================================
    # Tab 2: ESS plot + anomalies below
    # ============================================
    with tabs[1]:
        st.subheader("Extended Self-Similarity (ESS)")
        
        # Get plot-specific style
        plot_name_ess = "ESS (S_p vs S_3)"
        ps_ess = get_plot_style(plot_name_ess)
        colors_ess = _get_palette(ps_ess)
        
        fig_ess = go.Figure()
        plotted_any = False

        xi_all = {}
        xi_err_all = {}
        anom_all = {}

        for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
            kind = sim_groups[sim_prefix]["kind"]
            files = sim_groups[sim_prefix]["files"][start_idx-1:end_idx]
            if not files:
                continue
            r, Sp_mean, Sp_std, urms, ps_here = _compute_time_avg_structure(tuple(files), kind)
            if r is None:
                continue
            if ref_p not in Sp_mean:
                st.warning(f"Reference order p={ref_p} not available for {sim_prefix}. Available orders: {sorted(Sp_mean.keys()) if Sp_mean else 'none'}")
                continue

            legend_base = st.session_state.structure_legend_names.get(sim_prefix, sim_prefix.replace("_", " ").title())
            color, lw, dash, marker, msize, override_on = resolve_line_style(
                sim_prefix, idx, colors_ess, ps_ess,
                style_key="per_sim_style_structure",
                include_marker=True,
                default_marker="circle"
            )
            plotted_any = True

            xi_all[sim_prefix] = {}
            xi_err_all[sim_prefix] = {}
            anom_all[sim_prefix] = {}

            def _norm(p, arr):
                if normalize_by_urms and np.isfinite(urms):
                    return arr / (urms ** p)
                return arr

            x = _norm(ref_p, Sp_mean[ref_p])

            for p in selected_ps:
                if p not in Sp_mean:
                    continue

                y = _norm(p, Sp_mean[p])
                y_std = _norm(p, Sp_std[p]) if p in Sp_std else None
                x_std = _norm(ref_p, Sp_std[ref_p]) if ref_p in Sp_std else None

                # robust fit mask: r-range + positive finite x,y
                rmask = (
                    (r >= fit_rmin) & (r <= fit_rmax) &
                    np.isfinite(x) & (x > 0) &
                    np.isfinite(y) & (y > 0)
                )

                trace_kwargs = dict(
                    x=x, y=y,
                    mode="lines+markers",
                    name=f"{legend_base} (p={p})",
                    line=dict(color=color, width=lw, dash=dash),
                    marker=dict(symbol=marker, size=msize),
                    hovertemplate=f"S_{ref_p}=%{{x:.3g}}<br>S_{p}=%{{y:.3g}}<extra></extra>"
                )
                
                # Add error bars if requested
                if show_error_bars:
                    error_dict = {}
                    if x_std is not None:
                        error_dict["error_x"] = dict(
                            type="data",
                            array=x_std,
                            visible=True,
                            thickness=1,
                            color=color
                        )
                    if y_std is not None:
                        error_dict["error_y"] = dict(
                            type="data",
                            array=y_std,
                            visible=True,
                            thickness=1,
                            color=color
                        )
                    if error_dict:
                        trace_kwargs.update(error_dict)
                
                fig_ess.add_trace(go.Scatter(**trace_kwargs))
                
                # Add shaded band if requested (for y-direction uncertainty)
                if show_std_band and y_std is not None:
                    rgb = _color_to_rgb_tuple(color)
                    fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps_ess['std_alpha']})"
                    fig_ess.add_trace(go.Scatter(
                        x=np.concatenate([x, x[::-1]]),
                        y=np.concatenate([y - y_std, (y + y_std)[::-1]]),
                        fill="toself",
                        fillcolor=fill_rgba,
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip"
                    ))

                if np.count_nonzero(rmask) >= 3:
                    logx = np.log(x[rmask])
                    logy = np.log(y[rmask])
                    valid = np.isfinite(logx) & np.isfinite(logy)
                    if np.count_nonzero(valid) >= 3:
                        slope, intercept = np.polyfit(logx[valid], logy[valid], 1)

                        yfit = slope * logx[valid] + intercept
                        resid = logy[valid] - yfit
                        dof = max(len(resid) - 2, 1)
                        stderr = np.sqrt(np.sum(resid**2) / dof) / np.sqrt(len(resid))

                        xi_all[sim_prefix][p] = float(slope)
                        xi_err_all[sim_prefix][p] = float(stderr)
                        anom_all[sim_prefix][p] = float(slope - p/3)

        if not plotted_any:
            st.info("No valid ESS data to plot.")
        else:
            layout_kwargs = dict(
                xaxis_title=st.session_state.axis_labels_structure.get("x_ess", "S<sub>3</sub>(r)"),
                yaxis_title=st.session_state.axis_labels_structure.get("y_ess", "S<sub>p</sub>(r)"),
                legend_title="Simulation / Order",
                height=500,  # Default, will be overridden if custom size is enabled
            )
            layout_kwargs = apply_axis_limits(layout_kwargs, ps_ess)
            layout_kwargs = apply_figure_size(layout_kwargs, ps_ess)
            fig_ess.update_layout(**layout_kwargs)
            fig_ess = apply_plot_style(fig_ess, ps_ess)
            
            # Add inset for anomalies (ξ_p - p/3) vs p
            if show_inset:
                # Get inset-specific plot style, fallback to ESS style if not configured
                plot_name_inset = "ESS Inset"
                ps_inset = get_plot_style(plot_name_inset)
                # If inset style hasn't been customized, use ESS style as base
                if not st.session_state.get("plot_styles", {}).get(plot_name_inset):
                    # Merge ESS style into inset style as defaults
                    for key, value in ps_ess.items():
                        if key not in ps_inset or ps_inset[key] == default_plot_style().get(key):
                            ps_inset[key] = value
                
                fig_ess = add_ess_inset(
                    fig=fig_ess,
                    xi_all=xi_all,
                    anom_all=anom_all,
                    xi_err_all=xi_err_all,
                    sim_groups=sim_groups,
                    legend_names=st.session_state.structure_legend_names,
                    colors_palette=colors_ess,
                    plot_style=ps_inset,
                    show_sl_theory=show_sl_theory,
                    show_exp_anom=show_exp_anom,
                    inset_x_label=st.session_state.axis_labels_structure.get("x_inset", "p"),
                    inset_y_label=st.session_state.axis_labels_structure.get("y_inset", "ξ<sub>p</sub> - p/3"),
                    inset_legend_sl=st.session_state.axis_labels_structure.get("inset_legend_sl", "SL94"),
                    inset_legend_b93=st.session_state.axis_labels_structure.get("inset_legend_b93", "B93")
                )
            
            st.plotly_chart(fig_ess, width='stretch')
            capture_button(fig_ess, title="Structure Functions ESS", source_page="Structure Functions")
            export_panel(fig_ess, data_dir, base_name="structure_functions_ess")

            st.markdown("#### Anomalies (ξₚ − p/3)")
            
            # Get plot-specific style
            plot_name_anom = "Anomalies (ξₚ − p/3)"
            ps_anom = get_plot_style(plot_name_anom)
            colors_anom = _get_palette(ps_anom)
            
            fig_anom = go.Figure()

            for idx, sim_prefix in enumerate(sorted(xi_all.keys())):
                color, lw, dash, marker, msize, override_on = resolve_line_style(
                    sim_prefix, idx, colors_anom, ps_anom,
                    style_key="per_sim_style_structure",
                    include_marker=True,
                    default_marker="circle"
                )
                ps_show = sorted(xi_all[sim_prefix].keys())
                yvals = [anom_all[sim_prefix][p] for p in ps_show]
                yerr = [xi_err_all[sim_prefix].get(p, 0.0) for p in ps_show]

                fig_anom.add_trace(go.Scatter(
                    x=ps_show, y=yvals,
                    mode="lines+markers",
                    name=st.session_state.structure_legend_names.get(sim_prefix, sim_prefix.replace("_", " ").title()),
                    line=dict(color=color, width=max(1.0, lw*0.7)),
                    marker=dict(symbol=marker, size=max(4, int(msize*0.7))),
                    error_y=dict(type="data", array=yerr, visible=True, thickness=1),
                ))

            if show_sl_theory:
                ps_theory = list(range(1, max(selected_ps)+1))
                theory_anom = [zeta_p_she_leveque(p) - p/3 for p in ps_theory]
                # Use color from plot style (automatically adjusted for dark theme)
                sl_color = ps_anom.get("she_leveque_color", "#000000")
                fig_anom.add_trace(go.Scatter(
                    x=ps_theory, y=theory_anom,
                    mode="lines+markers",
                    name="She–Leveque 1994",
                    line=dict(color=sl_color, dash="dash", width=1.5),
                    marker=dict(symbol="diamond", size=5),
                ))

            if show_exp_anom:
                exp_anom = [EXP_ZETA[i] - TABLE_P[i]/3 for i in range(len(TABLE_P))]
                # Use color from plot style (automatically adjusted for dark theme)
                exp_color = ps_anom.get("experimental_b93_color", "#00BFC4")
                fig_anom.add_trace(go.Scatter(
                    x=TABLE_P, y=exp_anom,
                    mode="lines+markers",
                    name="Experiment (B93)",
                    line=dict(color=exp_color, width=1.5),
                    marker=dict(symbol="x", size=6),
                ))

            fig_anom.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

            layout_kwargs_anom = dict(
                xaxis_title=st.session_state.axis_labels_structure.get("x_anom", "p"),
                yaxis_title=st.session_state.axis_labels_structure.get("y_anom", "ξ<sub>p</sub> - p/3"),
                height=360,
                legend_title="",
            )
            layout_kwargs_anom = apply_axis_limits(layout_kwargs_anom, ps_anom)
            layout_kwargs_anom = apply_figure_size(layout_kwargs_anom, ps_anom)
            fig_anom.update_layout(**layout_kwargs_anom)
            fig_anom = apply_plot_style(fig_anom, ps_anom)
            st.plotly_chart(fig_anom, width='stretch')
            export_panel(fig_anom, data_dir, base_name="structure_functions_anomalies")

            st.session_state["_xi_all"] = xi_all
            st.session_state["_anom_all"] = anom_all
            st.session_state["_xi_err_all"] = xi_err_all

    # ============================================
    # Tab 3: table
    # ============================================
    with tabs[2]:
        st.subheader("Computed ESS Scaling Exponents")
        xi_all = st.session_state.get("_xi_all", {})
        xi_err_all = st.session_state.get("_xi_err_all", {})
        if not xi_all:
            st.info("Run ESS tab first to populate exponents.")
        else:
            import pandas as pd
            
            # Get all available simulations
            all_simulations = sorted(xi_all.keys())
            
            # Interactive selector for which simulations to show
            if len(all_simulations) > 1:
                selected_sims = st.multiselect(
                    "Select simulations to display:",
                    options=all_simulations,
                    default=all_simulations,
                    key="table_sim_selector"
                )
            else:
                selected_sims = all_simulations
            
            if not selected_sims:
                st.info("Please select at least one simulation to display.")
            else:
                rows = []
                for sim_prefix in selected_sims:
                    if sim_prefix not in xi_all:
                        continue
                    xi_dict = xi_all[sim_prefix]
                    for p, xi in xi_dict.items():
                        rows.append({
                            "simulation": st.session_state.structure_legend_names.get(sim_prefix, sim_prefix),
                            "p": p,
                            "xi_p": f"{xi:.6f}",
                            "stderr": f"{xi_err_all.get(sim_prefix, {}).get(p, np.nan):.6f}",
                            "xi_p - p/3": f"{xi - p/3:.6f}",
                            "She–Leveque ζ_p": f"{zeta_p_she_leveque(p):.6f}",
                            "xi_p - ζ_p": f"{xi - zeta_p_she_leveque(p):.6f}",
                        })
                
                if rows:
                    df = pd.DataFrame(rows).sort_values(["simulation", "p"])
                    
                    # Display table with better formatting
                    st.dataframe(
                        df,
                        width='stretch',
                        hide_index=True,
                        height=min(400, 50 + len(df) * 35)  # Dynamic height based on rows
                    )
                    
                    # Download button
                    col1, col2 = st.columns([1, 4])
                    with col1:
                        st.download_button(
                            "📥 Download CSV",
                            df.to_csv(index=False).encode("utf-8"),
                            file_name="ess_scaling_exponents.csv",
                            mime="text/csv",
                            key="download_ess_table"
                        )
                    with col2:
                        st.caption(f"Showing {len(selected_sims)} simulation(s) with {len(df)} total rows")
                else:
                    st.warning("No data available for selected simulations.")

    # ============================================
    # Theory section
    # ============================================
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("**Structure functions:**")
        st.latex(r"""
        S_p(r) = \langle |\delta u_L(r)|^p \rangle
        """)
        st.markdown(r"""
        where $\delta u_L(r) = u_L(\mathbf{x} + r\mathbf{e}_L) - u_L(\mathbf{x})$ is the longitudinal velocity increment.
        """)
        
        st.markdown("**Extended Self-Similarity (ESS):** ([Benzi et al., 1993](/Citation#benzi1993))")
        st.latex(r"""
        S_p(r) \propto S_3(r)^{\xi_p}
        """)
        st.markdown(r"""
        The scaling exponent $\xi_p$ is obtained from the slope of $\log S_p$ vs $\log S_3$.
        """)
        
        st.markdown("**She–Leveque 1994 scaling (theoretical):** ([She & Leveque, 1994](/Citation#she1994))")
        st.latex(r"""
        \zeta_p = \frac{p}{9} + 2\left(1 - \left(\frac{2}{3}\right)^{p/3}\right)
        """)
        st.markdown(r"""
        Anomalies are plotted as $\xi_p - p/3$ to compare with theoretical predictions.
        """)


if __name__ == "__main__":
    main()

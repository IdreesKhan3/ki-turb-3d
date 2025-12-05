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
- FULL user controls (persistent per dataset in legend_names.json):
    * Legends, axis labels
    * Fonts, tick style, major/minor grids, background colors, theme
    * Palette / custom colors
    * Per-simulation overrides: color/width/dash/marker/marker size
- Research-grade export:
    * PNG/PDF/SVG/EPS/JPG/WEBP/TIFF + HTML
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
import json
import glob

# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.file_detector import (
    detect_simulation_files,
    group_files_by_simulation,
    natural_sort_key
)
from utils.report_builder import capture_button
from utils.plot_style import resolve_line_style, render_per_sim_style_ui, render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size

# Binary/text readers (binary is required by plan, text is optional)
from data_readers.binary_reader import read_structure_function_file
st.set_page_config(page_icon="⚫")
try:
    from data_readers.text_reader import read_structure_function_txt
except Exception:
    read_structure_function_txt = None


# ==========================================================
# JSON persistence (dataset-local)
# ==========================================================
def _legend_json_path(data_dir: Path) -> Path:
    return data_dir / "legend_names.json"

def _default_labelify(name: str) -> str:
    return name.replace("_", " ").title()

def _load_ui_metadata(data_dir: Path):
    """Load structure legends + axis labels + plot_styles from legend_names.json."""
    path = _legend_json_path(data_dir)
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        st.session_state.structure_legend_names = meta.get("structure_legends", {})
        st.session_state.axis_labels_structure = meta.get("axis_labels_structure", {})
        
        # Load plot_styles (per-plot system)
        st.session_state.plot_styles = meta.get("plot_styles", {})
        # Backward compatibility: if plot_styles doesn't exist, migrate from old plot_style
        if not st.session_state.plot_styles and "plot_style" in meta:
            # Migrate old single plot_style to all plots
            old_style = meta.get("plot_style", {})
            st.session_state.plot_styles = {
                "S_p(r) vs r": old_style.copy(),
                "ESS (S_p vs S_3)": old_style.copy(),
                "Anomalies (ξₚ − p/3)": old_style.copy(),
            }
    except Exception:
        st.toast("legend_names.json exists but could not be read. Using defaults.", icon="⚠️")

def _save_ui_metadata(data_dir: Path):
    """Merge-save UI metadata without clobbering other pages."""
    path = _legend_json_path(data_dir)
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            old = {}

    old.update({
        "structure_legends": st.session_state.get("structure_legend_names", {}),
        "axis_labels_structure": st.session_state.get("axis_labels_structure", {}),
        "plot_styles": st.session_state.get("plot_styles", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json (read-only folder?): {e}")


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
        Sp_std_dict = {p: Sp_std[p-1, :] for p in ps}
    else:
        Sp_std_dict = {p: np.zeros(max_dr) for p in ps}

    return r_mean, Sp_mean_dict, Sp_std_dict, u_rms_mean, list(ps)


# ==========================================================
# Research-grade export system
# ==========================================================
_EXPORT_FORMATS = {
    "PNG (raster)": "png",
    "PDF (vector)": "pdf",
    "SVG (vector)": "svg",
    "EPS (vector)": "eps",
    "JPG/JPEG (raster)": "jpg",
    "WEBP (raster)": "webp",
    "TIFF (raster)": "tiff",
    "HTML (interactive)": "html",
}

def export_panel(fig, out_dir: Path, base_name: str):
    with st.expander(f"📤 Export figure: {base_name}", expanded=False):
        fmts = st.multiselect(
            "Select export format(s)",
            list(_EXPORT_FORMATS.keys()),
            default=["PNG (raster)", "PDF (vector)", "SVG (vector)"],
            key=f"{base_name}_fmts"
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            scale = st.slider("Scale (DPI-like)", 1.0, 6.0, 3.0, 0.5, key=f"{base_name}_scale")
        with c2:
            width_px = st.number_input("Width px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_wpx")
        with c3:
            height_px = st.number_input("Height px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_hpx")

        if st.button("Export selected formats", key=f"{base_name}_doexport"):
            if not fmts:
                st.warning("Please select at least one format.")
                return

            errors = []
            chrome_error = False
            for f_label in fmts:
                ext = _EXPORT_FORMATS[f_label]
                out = out_dir / f"{base_name}.{ext}"
                try:
                    if ext == "html":
                        fig.write_html(str(out))
                        continue

                    kwargs = {}
                    if width_px > 0:
                        kwargs["width"] = int(width_px)
                    if height_px > 0:
                        kwargs["height"] = int(height_px)

                    fig.write_image(str(out), scale=scale, **kwargs)
                except Exception as e:
                    error_msg = str(e)
                    errors.append((out.name, error_msg))
                    if "Chrome" in error_msg or "chrome" in error_msg.lower():
                        chrome_error = True

            if errors:
                error_text = "Some exports failed.\n\n"
                
                if chrome_error:
                    error_text += (
                        "**Issue:** Kaleido (the image export library) requires Google Chrome to be installed on your system.\n\n"
                        "**What you need:** Google Chrome browser installed on your computer.\n\n"
                        "**Solution - Choose one:**\n\n"
                        "**Option 1 (Recommended):** Auto-install Chrome via command:\n"
                        "```bash\n"
                        "kaleido_get_chrome\n"
                        "```\n"
                        "This command downloads and installs Chrome automatically.\n\n"
                        "**Option 2:** Install Chrome manually:\n"
                        "1. Download Chrome from: https://www.google.com/chrome/\n"
                        "2. Install it following the installer instructions\n"
                        "3. Restart your application\n\n"
                        "**Note:** After installing Chrome, you may need to update kaleido:\n"
                        "```bash\n"
                        "pip install -U kaleido\n"
                        "```\n\n"
                    )
                else:
                    error_text += "Ensure kaleido is installed:\n```bash\npip install -U kaleido\n```\n\n"
                
                error_text += "**Failed exports:**\n" + "\n".join([f"- {n}: {msg}" for n, msg in errors])
                st.error(error_text)
            else:
                st.success("All selected exports saved to dataset folder.")


# ==========================================================
# Plot styling system
# ==========================================================
def _default_plot_style():
    return {
        "font_family": "Arial",
        "font_size": 14,
        "title_size": 16,
        "legend_size": 12,
        "tick_font_size": 12,
        "axis_title_size": 14,

        "tick_len": 6,
        "tick_w": 1.2,
        "ticks_outside": True,

        "plot_bgcolor": "#FFFFFF",
        "paper_bgcolor": "#FFFFFF",

        "show_grid": True,
        "grid_on_x": True,
        "grid_on_y": True,
        "grid_w": 0.6,
        "grid_dash": "dot",
        "grid_color": "#B0B0B0",
        "grid_opacity": 0.6,

        "show_minor_grid": False,
        "minor_grid_w": 0.4,
        "minor_grid_dash": "dot",
        "minor_grid_color": "#D0D0D0",
        "minor_grid_opacity": 0.45,

        "line_width": 2.4,
        "marker_size": 6,
        "std_alpha": 0.18,

        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",

        "enable_per_sim_style": False,
        "per_sim_style_structure": {},
        
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
        "margin_right": 20,
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
            r=ps.get("margin_right", 20),
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
    return plot_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("ₚ", "p").replace("ξ", "xi").replace("/", "_")

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
    ps.setdefault("per_sim_style_structure", {})
    for k in sim_groups.keys():
        ps["per_sim_style_structure"].setdefault(k, {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": None,
            "marker": "circle",
            "msize": None,
        })

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
    plot_key = _normalize_plot_name(selected_plot)
    _ensure_per_sim_defaults(ps, sim_groups)
    
    # Create unique key prefix for all widgets
    key_prefix = f"structure_{plot_key}"

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown(f"**Configuring: {selected_plot}**")
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        ps["font_family"] = st.selectbox("Font family", fonts, index=fonts.index(ps.get("font_family", "Arial")),
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
        ps["marker_size"] = st.slider("Global marker size", 0, 14, int(ps.get("marker_size", 6)),
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
                                                  value=int(ps.get("margin_right", 20)), 
                                                  step=5, key=f"{key_prefix}_margin_right")
            ps["margin_bottom"] = st.number_input("Bottom margin (px)", min_value=0, max_value=200, 
                                                   value=int(ps.get("margin_bottom", 50)), 
                                                   step=5, key=f"{key_prefix}_margin_bottom")
        st.markdown("---")
        render_per_sim_style_ui(ps, sim_groups, style_key="per_sim_style_structure", 
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
    
    st.title("📊 Structure Functions")

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

    # Defaults
    st.session_state.setdefault("structure_legend_names", {})
    st.session_state.setdefault("axis_labels_structure", {
        "x_r": "Separation distance $r$",
        "y_sp": "Structure functions $S_p(r)$",
        "x_ess": r"$S_3(r)$",
        "y_ess": r"$S_p(r)$",
        "y_anom": r"$\xi_p - p/3$",
    })
    st.session_state.setdefault("plot_styles", {})

    # Load json once per dataset change
    if st.session_state.get("_last_struct_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        if "plot_styles" not in st.session_state:
            st.session_state.plot_styles = {}
        st.session_state["_last_struct_dir"] = str(data_dir)

    # Collect files from all directories
    all_bin_files = []
    all_txt_files = []
    
    for data_dir_path in data_dirs:
        data_dir_obj = Path(data_dir_path)
        if data_dir_obj.exists():
            files_dict = detect_simulation_files(str(data_dir_obj))
            dir_bin = files_dict.get("structure_functions_bin", [])
            dir_txt = files_dict.get("structure_functions_txt", [])
            all_bin_files.extend([str(f) for f in dir_bin])
            all_txt_files.extend([str(f) for f in dir_txt])
    
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
        with st.sidebar.expander("📊 File counts", expanded=False):
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
    show_sl_theory = st.sidebar.checkbox("Show She–Leveque anomalies", value=True)
    show_exp_anom = st.sidebar.checkbox("Show experimental anomalies (B93)", value=True)

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
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        for sim_prefix in sorted(sim_groups.keys()):
            st.session_state.structure_legend_names.setdefault(sim_prefix, _default_labelify(sim_prefix))
            st.session_state.structure_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.structure_legend_names[sim_prefix],
                key=f"legend_struct_{sim_prefix}"
            )

        st.markdown("---")
        st.markdown("### Axis labels")
        st.session_state.axis_labels_structure["x_r"] = st.text_input(
            "S_p plot x-label", st.session_state.axis_labels_structure.get("x_r", "Separation distance $r$"), key="ax_struct_xr"
        )
        st.session_state.axis_labels_structure["y_sp"] = st.text_input(
            "S_p plot y-label", st.session_state.axis_labels_structure.get("y_sp", "Structure functions $S_p(r)$"), key="ax_struct_ysp"
        )
        st.session_state.axis_labels_structure["x_ess"] = st.text_input(
            "ESS x-label", st.session_state.axis_labels_structure.get("x_ess", r"$S_3(r)$"), key="ax_struct_xess"
        )
        st.session_state.axis_labels_structure["y_ess"] = st.text_input(
            "ESS y-label", st.session_state.axis_labels_structure.get("y_ess", r"$S_p(r)$"), key="ax_struct_yess"
        )
        st.session_state.axis_labels_structure["y_anom"] = st.text_input(
            "Anomaly y-label", st.session_state.axis_labels_structure.get("y_anom", r"$\xi_p - p/3$"), key="ax_struct_yanom"
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save labels/legends"):
                _save_ui_metadata(data_dir)
                st.success("Saved to legend_names.json")
        with b2:
            if st.button("♻️ Reset labels/legends"):
                st.session_state.structure_legend_names = {k: _default_labelify(k) for k in sim_groups.keys()}
                st.session_state.axis_labels_structure.update({
                    "x_r": "Separation distance $r$",
                    "y_sp": "Structure functions $S_p(r)$",
                    "x_ess": r"$S_3(r)$",
                    "y_ess": r"$S_p(r)$",
                    "y_anom": r"$\xi_p - p/3$",
                })
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")
                st.rerun()

    # Full style sidebar
    plot_names = ["S_p(r) vs r", "ESS (S_p vs S_3)", "Anomalies (ξₚ − p/3)"]
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

            legend_base = st.session_state.structure_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
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
                xaxis_title=st.session_state.axis_labels_structure.get("x_r", "Separation distance $r$"),
                yaxis_title=st.session_state.axis_labels_structure.get("y_sp", "Structure functions $S_p(r)$"),
                xaxis_type="log",
                yaxis_type="log",
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

            legend_base = st.session_state.structure_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
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
                xaxis_title=st.session_state.axis_labels_structure.get("x_ess", r"$S_3(r)$"),
                yaxis_title=st.session_state.axis_labels_structure.get("y_ess", r"$S_p(r)$"),
                xaxis_type="log",
                yaxis_type="log",
                legend_title="Simulation / Order",
                height=500,  # Default, will be overridden if custom size is enabled
            )
            layout_kwargs = apply_axis_limits(layout_kwargs, ps_ess)
            layout_kwargs = apply_figure_size(layout_kwargs, ps_ess)
            fig_ess.update_layout(**layout_kwargs)
            fig_ess = apply_plot_style(fig_ess, ps_ess)
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
                    name=st.session_state.structure_legend_names.get(sim_prefix, _default_labelify(sim_prefix)),
                    line=dict(color=color, width=max(1.0, lw*0.7)),
                    marker=dict(symbol=marker, size=max(4, int(msize*0.7))),
                    error_y=dict(type="data", array=yerr, visible=True, thickness=1),
                ))

            if show_sl_theory:
                ps_theory = list(range(1, max(selected_ps)+1))
                theory_anom = [zeta_p_she_leveque(p) - p/3 for p in ps_theory]
                fig_anom.add_trace(go.Scatter(
                    x=ps_theory, y=theory_anom,
                    mode="lines+markers",
                    name="She–Leveque 1994",
                    line=dict(color="black", dash="dash", width=1.5),
                    marker=dict(symbol="diamond", size=5),
                ))

            if show_exp_anom:
                exp_anom = [EXP_ZETA[i] - TABLE_P[i]/3 for i in range(len(TABLE_P))]
                fig_anom.add_trace(go.Scatter(
                    x=TABLE_P, y=exp_anom,
                    mode="lines+markers",
                    name="Experiment (B93)",
                    line=dict(color="#00BFC4", width=1.5),
                    marker=dict(symbol="x", size=6),
                ))

            fig_anom.add_hline(y=0, line_dash="dot", line_color="black", line_width=1)

            layout_kwargs_anom = dict(
                xaxis_title=r"$p$",
                yaxis_title=st.session_state.axis_labels_structure.get("y_anom", r"$\xi_p - p/3$"),
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
        st.markdown("**Structure functions**")
        st.latex(r"S_p(r)=\langle |\delta u_L(r)|^p\rangle")
        
        st.markdown("**Extended Self-Similarity (ESS)**")
        st.latex(r"S_p(r)\propto S_3(r)^{\xi_p}")
        st.markdown("So $\\xi_p$ is obtained from the slope of $\\log S_p$ vs $\\log S_3$.")
        
        st.markdown("**She–Leveque 1994 scaling**")
        st.latex(r"\zeta_p=\frac{p}{9}+2\left(1-\left(\frac{2}{3}\right)^{p/3}\right)")
        st.markdown("Anomalies are plotted as $\\xi_p - p/3$.")


if __name__ == "__main__":
    main()

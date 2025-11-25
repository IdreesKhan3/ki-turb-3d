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

# Binary/text readers (binary is required by plan, text is optional)
from data_readers.binary_reader import read_structure_function_file
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
    """Load structure legends + axis labels + plot_style from legend_names.json."""
    path = _legend_json_path(data_dir)
    if not path.exists():
        return
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        st.session_state.structure_legend_names = meta.get("structure_legends", {})
        st.session_state.axis_labels_structure = meta.get("axis_labels_structure", {})
        st.session_state.plot_style = meta.get("plot_style", st.session_state.get("plot_style", {}))
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
        "plot_style": st.session_state.get("plot_style", {}),
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
    kind: "bin" or "txt".
    Returns r, S_p_mean dict, S_p_std dict, u_rms_mean, ps(list)
    """
    Rs = []
    Sp_list = []
    urms_list = []

    for f in files:
        try:
            data = _read_structure_bin_cached(str(f)) if kind == "bin" else _read_structure_txt_cached(str(f))
        except Exception:
            continue

        r = np.asarray(data.get("r", []), float)
        S_p = data.get("S_p", {})
        if r.size == 0 or not S_p:
            continue

        ps = sorted(S_p.keys())
        Sp_mat = np.vstack([np.asarray(S_p[p], float) for p in ps])

        Rs.append(r)
        Sp_list.append(Sp_mat)
        urms_list.append(float(data.get("u_rms", np.nan)))

    if not Sp_list:
        return None, None, None, None, None

    min_len = min(mat.shape[1] for mat in Sp_list)
    r0 = Rs[0][:min_len]
    Sp_arr = np.stack([mat[:, :min_len] for mat in Sp_list], axis=0)

    Sp_mean = np.mean(Sp_arr, axis=0)
    Sp_std = np.std(Sp_arr, axis=0)

    ps = list(range(1, Sp_mean.shape[0] + 1))
    Sp_mean_dict = {p: Sp_mean[p-1, :] for p in ps}
    Sp_std_dict = {p: Sp_std[p-1, :] for p in ps}

    u_rms_mean = np.nanmean(urms_list) if urms_list else np.nan
    return r0, Sp_mean_dict, Sp_std_dict, u_rms_mean, ps


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
                    errors.append((out.name, str(e)))

            if errors:
                st.error(
                    "Some exports failed. Ensure kaleido is installed:\n"
                    "pip install -U kaleido\n\n"
                    + "\n".join([f"- {n}: {msg}" for n, msg in errors])
                )
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
        title=dict(font=dict(size=ps["title_size"])),
        hovermode="x unified",
        plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
        paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"),
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

def plot_style_sidebar(data_dir: Path, sim_groups):
    ps = dict(st.session_state.plot_style)
    _ensure_per_sim_defaults(ps, sim_groups)

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        ps["font_family"] = st.selectbox("Font family", fonts, index=fonts.index(ps.get("font_family", "Arial")))
        ps["font_size"] = st.slider("Base/global font size", 8, 26, int(ps.get("font_size", 14)))
        ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)))
        ps["legend_size"] = st.slider("Legend font size", 8, 24, int(ps.get("legend_size", 12)))
        ps["tick_font_size"] = st.slider("Tick label font size", 6, 24, int(ps.get("tick_font_size", 12)))
        ps["axis_title_size"] = st.slider("Axis title font size", 8, 28, int(ps.get("axis_title_size", 14)))

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background (inside axes)", ps.get("plot_bgcolor", "#FFFFFF"))
        ps["paper_bgcolor"] = st.color_picker("Paper background (outside axes)", ps.get("paper_bgcolor", "#FFFFFF"))

        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)))
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)))
        ps["ticks_outside"] = st.checkbox("Ticks outside", bool(ps.get("ticks_outside", True)))

        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox("Show major grid", bool(ps.get("show_grid", True)))
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)))
        with gcol2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)))
        ps["grid_w"] = st.slider("Major grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)))
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        ps["grid_dash"] = st.selectbox("Major grid type", grid_styles,
                                       index=grid_styles.index(ps.get("grid_dash", "dot")))
        ps["grid_color"] = st.color_picker("Major grid color", ps.get("grid_color", "#B0B0B0"))
        ps["grid_opacity"] = st.slider("Major grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)))

        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)))
        ps["minor_grid_w"] = st.slider("Minor grid width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)))
        ps["minor_grid_dash"] = st.selectbox("Minor grid type", grid_styles,
                                             index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
                                             key="minor_grid_dash_struct")
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"))
        ps["minor_grid_opacity"] = st.slider("Minor grid opacity", 0.0, 1.0,
                                             float(ps.get("minor_grid_opacity", 0.45)))

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.4)))
        ps["marker_size"] = st.slider("Global marker size", 0, 14, int(ps.get("marker_size", 6)))
        ps["std_alpha"] = st.slider("Std band opacity", 0.05, 0.6, float(ps.get("std_alpha", 0.18)))

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox("Palette", palettes,
                                     index=palettes.index(ps.get("palette", "Plotly")))
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"cust_color_struct_{i}"))
            ps["custom_colors"] = new_cols

        st.markdown("---")
        st.markdown("**Theme**")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox("Template", templates,
                                      index=templates.index(ps.get("template", "plotly_white")))

        st.markdown("---")
        st.markdown("**Per-simulation overrides (optional)**")
        ps["enable_per_sim_style"] = st.checkbox("Enable per-simulation overrides",
                                                 bool(ps.get("enable_per_sim_style", False)))

        if ps["enable_per_sim_style"]:
            dash_opts = ["solid", "dot", "dash", "dashdot", "longdash", "longdashdot"]
            marker_opts = ["circle", "square", "diamond", "cross", "x", "triangle-up",
                           "triangle-down", "star", "hexagon"]

            with st.container(border=True):
                for sim_prefix in sorted(sim_groups.keys()):
                    s = ps["per_sim_style_structure"][sim_prefix]
                    st.markdown(f"`{sim_prefix}`")
                    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
                    with c1:
                        s["enabled"] = st.checkbox("Override", value=s.get("enabled", False),
                                                  key=f"struct_over_on_{sim_prefix}")
                    with c2:
                        s["color"] = st.color_picker("Color", value=s.get("color") or "#000000",
                                                     key=f"struct_over_color_{sim_prefix}",
                                                     disabled=not s["enabled"])
                    with c3:
                        s["width"] = st.slider("Width", 0.5, 8.0,
                                               float(s.get("width") or ps["line_width"]),
                                               key=f"struct_over_width_{sim_prefix}",
                                               disabled=not s["enabled"])
                    with c4:
                        s["dash"] = st.selectbox("Dash", dash_opts,
                                                 index=dash_opts.index(s.get("dash") or "solid"),
                                                 key=f"struct_over_dash_{sim_prefix}",
                                                 disabled=not s["enabled"])
                    with c5:
                        s["marker"] = st.selectbox("Marker", marker_opts,
                                                   index=marker_opts.index(s.get("marker") or "circle"),
                                                   key=f"struct_over_marker_{sim_prefix}",
                                                   disabled=not s["enabled"])
                    s["msize"] = st.slider("Marker size", 0, 18,
                                           int(s.get("msize") or ps["marker_size"]),
                                           key=f"struct_over_msize_{sim_prefix}",
                                           disabled=not s["enabled"])

        st.markdown("---")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save Plot Style"):
                st.session_state.plot_style = ps
                _save_ui_metadata(data_dir)
                st.success("Saved plot style.")
        with b2:
            if st.button("♻️ Reset Plot Style"):
                st.session_state.plot_style = _default_plot_style()
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")

    st.session_state.plot_style = ps


def _resolve_line_style(sim_prefix, idx, colors, ps):
    default_color = colors[idx % len(colors)]
    default_width = ps["line_width"]
    default_dash = "solid"
    default_marker = "circle"
    default_msize = ps["marker_size"]

    if not ps.get("enable_per_sim_style", False):
        return default_color, default_width, default_dash, default_marker, default_msize, False

    s = ps.get("per_sim_style_structure", {}).get(sim_prefix, {})
    if not s.get("enabled", False):
        return default_color, default_width, default_dash, default_marker, default_msize, False

    color = s.get("color") or default_color
    width = float(s.get("width") or default_width)
    dash = s.get("dash") or default_dash
    marker = s.get("marker") or default_marker
    msize = int(s.get("msize") or default_msize)
    return color, width, dash, marker, msize, True


# ==========================================================
# Page main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("📊 Structure Functions")

    data_dir = st.session_state.get("data_directory", None)
    if not data_dir:
        st.warning("Please select a data directory from the Overview page.")
        return
    data_dir = Path(data_dir)

    # Defaults
    st.session_state.setdefault("structure_legend_names", {})
    st.session_state.setdefault("axis_labels_structure", {
        "x_r": "Separation distance $r$",
        "y_sp": "Structure functions $S_p(r)$",
        "x_ess": r"$S_3(r)$",
        "y_ess": r"$S_p(r)$",
        "y_anom": r"$\xi_p - p/3$",
    })
    st.session_state.setdefault("plot_style", _default_plot_style())

    # Load json once per dataset change
    if st.session_state.get("_last_struct_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        merged = _default_plot_style()
        merged.update(st.session_state.plot_style or {})
        st.session_state.plot_style = merged
        st.session_state["_last_struct_dir"] = str(data_dir)

    ps = st.session_state.plot_style

    # Detect files
    files_dict = detect_simulation_files(str(data_dir))

    # Primary binary list (your naming)
    bin_files = files_dict.get("structure_bin", [])
    if not bin_files:
        bin_files = glob.glob(str(data_dir / "structure_funcs*_t*.bin"))

    txt_files = files_dict.get("structure_txt", [])
    if not txt_files and read_structure_function_txt is not None:
        txt_files = glob.glob(str(data_dir / "structure_functions*_t*.txt"))

    if not bin_files and not txt_files:
        st.info("No structure function files found. Expected `structure_funcs*_t*.bin`.")
        return

    # Grouping
    sim_groups_bin = group_files_by_simulation(
        sorted([str(f) for f in bin_files], key=natural_sort_key),
        r"(structure_funcs\d+)_t\d+\.bin"
    ) if bin_files else {}

    sim_groups_txt = {}
    if txt_files:
        # relaxed pattern to catch more text variants
        sim_groups_txt = group_files_by_simulation(
            sorted([str(f) for f in txt_files], key=natural_sort_key),
            r"(structure_functions\w*\d+)_t\d+\.txt"
        )

    sim_groups = {}
    for k, v in sim_groups_bin.items():
        sim_groups[k] = {"kind": "bin", "files": v}
    for k, v in sim_groups_txt.items():
        if k not in sim_groups:
            sim_groups[k] = {"kind": "txt", "files": v}

    if not sim_groups:
        st.warning("Could not group structure files by simulation prefix.")
        return

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
            "S_p plot x-label", st.session_state.axis_labels_structure["x_r"], key="ax_struct_xr"
        )
        st.session_state.axis_labels_structure["y_sp"] = st.text_input(
            "S_p plot y-label", st.session_state.axis_labels_structure["y_sp"], key="ax_struct_ysp"
        )
        st.session_state.axis_labels_structure["x_ess"] = st.text_input(
            "ESS x-label", st.session_state.axis_labels_structure["x_ess"], key="ax_struct_xess"
        )
        st.session_state.axis_labels_structure["y_ess"] = st.text_input(
            "ESS y-label", st.session_state.axis_labels_structure["y_ess"], key="ax_struct_yess"
        )
        st.session_state.axis_labels_structure["y_anom"] = st.text_input(
            "Anomaly y-label", st.session_state.axis_labels_structure["y_anom"], key="ax_struct_yanom"
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

    # Sidebar time window
    st.sidebar.subheader("Time Window")
    min_len = min(len(v["files"]) for v in sim_groups.values())
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
    show_std_band = st.sidebar.checkbox("Show ±1σ band (S_p plot)", value=True)
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

    # Full style sidebar
    plot_style_sidebar(data_dir, sim_groups)
    ps = st.session_state.plot_style
    colors = _get_palette(ps)

    tabs = st.tabs(["Sₚ(r) vs r", "ESS (Sₚ vs S₃)", "Scaling Exponents Table"])

    # ============================================
    # Tab 1: S_p(r) vs r
    # ============================================
    with tabs[0]:
        st.subheader("Time-averaged Structure Functions")
        fig_sp = go.Figure()
        plotted_any = False

        for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
            kind = sim_groups[sim_prefix]["kind"]
            files = sim_groups[sim_prefix]["files"][start_idx-1:end_idx]
            r, Sp_mean, Sp_std, urms, ps_here = _compute_time_avg_structure(tuple(files), kind)
            if r is None:
                continue

            legend_base = st.session_state.structure_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
            color_base, lw_base, dash_base, marker_base, msize_base, override_on = _resolve_line_style(
                sim_prefix, idx, colors, ps
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

                # Per-order color unless override is enabled
                per_order_color = colors[(idx + j) % len(colors)]
                line_color = color_base if override_on else per_order_color

                fig_sp.add_trace(go.Scatter(
                    x=r, y=y,
                    mode="lines",
                    name=f"{legend_base}  (p={p})",
                    line=dict(color=line_color, width=lw_base, dash=dash_base),
                    hovertemplate="r=%{x:.3g}<br>S_p=%{y:.3g}<extra></extra>"
                ))

                if show_std_band and ystd is not None:
                    rgb = hex_to_rgb(line_color)
                    fill_rgba = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{ps['std_alpha']})"
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
            fig_sp.update_layout(
                xaxis_title=st.session_state.axis_labels_structure["x_r"],
                yaxis_title=st.session_state.axis_labels_structure["y_sp"],
                xaxis_type="log",
                yaxis_type="log",
                legend_title="Simulation / Order",
                height=620,
                margin=dict(l=50, r=20, t=30, b=50),
            )
            fig_sp = apply_plot_style(fig_sp, ps)
            st.plotly_chart(fig_sp, use_container_width=True)
            export_panel(fig_sp, data_dir, base_name="structure_functions_sp")

    # ============================================
    # Tab 2: ESS plot + anomalies below
    # ============================================
    with tabs[1]:
        st.subheader("Extended Self-Similarity (ESS)")
        fig_ess = go.Figure()
        plotted_any = False

        xi_all = {}
        xi_err_all = {}
        anom_all = {}

        for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
            kind = sim_groups[sim_prefix]["kind"]
            files = sim_groups[sim_prefix]["files"][start_idx-1:end_idx]
            r, Sp_mean, Sp_std, urms, ps_here = _compute_time_avg_structure(tuple(files), kind)
            if r is None or ref_p not in Sp_mean:
                continue

            legend_base = st.session_state.structure_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
            color, lw, dash, marker, msize, override_on = _resolve_line_style(sim_prefix, idx, colors, ps)
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

                # robust fit mask: r-range + positive finite x,y
                rmask = (
                    (r >= fit_rmin) & (r <= fit_rmax) &
                    np.isfinite(x) & (x > 0) &
                    np.isfinite(y) & (y > 0)
                )

                fig_ess.add_trace(go.Scatter(
                    x=x, y=y,
                    mode="lines+markers",
                    name=f"{legend_base} (p={p})",
                    line=dict(color=color, width=lw, dash=dash),
                    marker=dict(symbol=marker, size=msize),
                    hovertemplate=f"S_{ref_p}=%{{x:.3g}}<br>S_{p}=%{{y:.3g}}<extra></extra>"
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
            fig_ess.update_layout(
                xaxis_title=st.session_state.axis_labels_structure["x_ess"],
                yaxis_title=st.session_state.axis_labels_structure["y_ess"],
                xaxis_type="log",
                yaxis_type="log",
                legend_title="Simulation / Order",
                height=620,
                margin=dict(l=50, r=20, t=30, b=50),
            )
            fig_ess = apply_plot_style(fig_ess, ps)
            st.plotly_chart(fig_ess, use_container_width=True)
            export_panel(fig_ess, data_dir, base_name="structure_functions_ess")

            st.markdown("#### Anomalies (ξₚ − p/3)")
            fig_anom = go.Figure()

            for idx, sim_prefix in enumerate(sorted(xi_all.keys())):
                color, lw, dash, marker, msize, override_on = _resolve_line_style(sim_prefix, idx, colors, ps)
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

            fig_anom.update_layout(
                xaxis_title=r"$p$",
                yaxis_title=st.session_state.axis_labels_structure["y_anom"],
                height=360,
                margin=dict(l=50, r=20, t=30, b=50),
                legend_title="",
            )
            fig_anom = apply_plot_style(fig_anom, ps)
            st.plotly_chart(fig_anom, use_container_width=True)
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
            rows = []
            for sim_prefix, xi_dict in xi_all.items():
                for p, xi in xi_dict.items():
                    rows.append({
                        "simulation": st.session_state.structure_legend_names.get(sim_prefix, sim_prefix),
                        "p": p,
                        "xi_p": xi,
                        "stderr": xi_err_all.get(sim_prefix, {}).get(p, np.nan),
                        "xi_p - p/3": xi - p/3,
                        "She–Leveque ζ_p": zeta_p_she_leveque(p),
                        "xi_p - ζ_p": xi - zeta_p_she_leveque(p),
                    })
            import pandas as pd
            df = pd.DataFrame(rows).sort_values(["simulation", "p"])
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download exponents CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="ess_scaling_exponents.csv",
                mime="text/csv"
            )

    # ============================================
    # Theory section
    # ============================================
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(r"""
**Structure functions**
\[
S_p(r)=\langle |\delta u_L(r)|^p\rangle
\]

**Extended Self-Similarity (ESS)**
\[
S_p(r)\propto S_3(r)^{\xi_p}
\]

So \(\xi_p\) is obtained from the slope of \(\log S_p\) vs \(\log S_3\).

**She–Leveque 1994 scaling**
\[
\zeta_p=\frac{p}{9}+2\left(1-\left(\frac{2}{3}\right)^{p/3}\right)
\]
Anomalies are plotted as \(\xi_p - p/3\).
        """)


if __name__ == "__main__":
    main()

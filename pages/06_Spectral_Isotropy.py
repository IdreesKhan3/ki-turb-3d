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
        st.session_state.spec_iso_legends = meta.get("spec_iso_legends", {})
        st.session_state.axis_labels_spec_iso = meta.get("axis_labels_spec_iso", {})
        st.session_state.plot_style = meta.get("plot_style", st.session_state.get("plot_style", {}))
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
        "plot_style": st.session_state.get("plot_style", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json: {e}")


# ==========================================================
# Plot styling system (same keys as other pages)
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
        "minor_grid_opacity": 0.4,

        "line_width": 2.2,
        "marker_size": 6,

        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",

        "enable_per_curve_style": False,
        "per_curve_style_spec_iso": {}  # {curve: {enabled,color,width,dash,marker,msize}}
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
        margin=dict(l=60, r=20, t=40, b=50),
    )

    tick_dir = "outside" if ps["ticks_outside"] else "inside"

    show_x_grid = ps["show_grid"] and ps.get("grid_on_x", True)
    show_y_grid = ps["show_grid"] and ps.get("grid_on_y", True)
    grid_rgba = f"rgba{hex_to_rgb(ps['grid_color']) + (ps['grid_opacity'],)}"

    show_minor = ps.get("show_minor_grid", False)
    minor_rgba = f"rgba{hex_to_rgb(ps['minor_grid_color']) + (ps['minor_grid_opacity'],)}"

    fig.update_xaxes(
        ticks=tick_dir, ticklen=ps["tick_len"], tickwidth=ps["tick_w"],
        tickfont=_tick_font(ps), title_font=_axis_title_font(ps),
        showgrid=show_x_grid, gridwidth=ps["grid_w"],
        griddash=ps["grid_dash"], gridcolor=grid_rgba,
        minor=dict(showgrid=show_minor, gridwidth=ps["minor_grid_w"],
                   griddash=ps["minor_grid_dash"], gridcolor=minor_rgba)
    )
    fig.update_yaxes(
        ticks=tick_dir, ticklen=ps["tick_len"], tickwidth=ps["tick_w"],
        tickfont=_tick_font(ps), title_font=_axis_title_font(ps),
        showgrid=show_y_grid, gridwidth=ps["grid_w"],
        griddash=ps["grid_dash"], gridcolor=grid_rgba,
        minor=dict(showgrid=show_minor, gridwidth=ps["minor_grid_w"],
                   griddash=ps["minor_grid_dash"], gridcolor=minor_rgba)
    )
    return fig

def _ensure_curve_defaults(ps, curves):
    ps.setdefault("per_curve_style_spec_iso", {})
    for c in curves:
        ps["per_curve_style_spec_iso"].setdefault(c, {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": "solid",
            "marker": "circle",
            "msize": None
        })

def plot_style_sidebar(data_dir: Path, curves):
    ps = dict(st.session_state.plot_style)
    _ensure_curve_defaults(ps, curves)

    with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
        st.markdown("**Fonts**")
        fonts = ["Arial", "Helvetica", "Times New Roman", "Computer Modern", "Courier New"]
        ps["font_family"] = st.selectbox("Font family", fonts, index=fonts.index(ps.get("font_family", "Arial")))
        ps["font_size"] = st.slider("Base font size", 8, 26, int(ps.get("font_size", 14)))
        ps["title_size"] = st.slider("Title size", 10, 32, int(ps.get("title_size", 16)))
        ps["legend_size"] = st.slider("Legend size", 8, 24, int(ps.get("legend_size", 12)))
        ps["tick_font_size"] = st.slider("Tick label size", 6, 24, int(ps.get("tick_font_size", 12)))
        ps["axis_title_size"] = st.slider("Axis title size", 8, 28, int(ps.get("axis_title_size", 14)))

        st.markdown("---")
        st.markdown("**Backgrounds**")
        ps["plot_bgcolor"] = st.color_picker("Plot background", ps.get("plot_bgcolor", "#FFFFFF"))
        ps["paper_bgcolor"] = st.color_picker("Paper background", ps.get("paper_bgcolor", "#FFFFFF"))

        st.markdown("---")
        st.markdown("**Ticks**")
        ps["tick_len"] = st.slider("Tick length", 2, 14, int(ps.get("tick_len", 6)))
        ps["tick_w"] = st.slider("Tick width", 0.5, 3.5, float(ps.get("tick_w", 1.2)))
        ps["ticks_outside"] = st.checkbox("Ticks outside", bool(ps.get("ticks_outside", True)))

        st.markdown("---")
        st.markdown("**Grid (Major)**")
        ps["show_grid"] = st.checkbox("Show major grid", bool(ps.get("show_grid", True)))
        c1, c2 = st.columns(2)
        with c1:
            ps["grid_on_x"] = st.checkbox("Grid on X", bool(ps.get("grid_on_x", True)))
        with c2:
            ps["grid_on_y"] = st.checkbox("Grid on Y", bool(ps.get("grid_on_y", True)))
        ps["grid_w"] = st.slider("Grid width", 0.2, 2.5, float(ps.get("grid_w", 0.6)))
        grid_styles = ["solid", "dot", "dash", "dashdot"]
        ps["grid_dash"] = st.selectbox("Grid type", grid_styles,
                                       index=grid_styles.index(ps.get("grid_dash", "dot")))
        ps["grid_color"] = st.color_picker("Grid color", ps.get("grid_color", "#B0B0B0"))
        ps["grid_opacity"] = st.slider("Grid opacity", 0.0, 1.0, float(ps.get("grid_opacity", 0.6)))

        st.markdown("---")
        st.markdown("**Grid (Minor)**")
        ps["show_minor_grid"] = st.checkbox("Show minor grid", bool(ps.get("show_minor_grid", False)))
        ps["minor_grid_w"] = st.slider("Minor width", 0.1, 2.0, float(ps.get("minor_grid_w", 0.4)))
        ps["minor_grid_dash"] = st.selectbox("Minor type", grid_styles,
                                             index=grid_styles.index(ps.get("minor_grid_dash", "dot")),
                                             key="minor_grid_dash_speciso")
        ps["minor_grid_color"] = st.color_picker("Minor color", ps.get("minor_grid_color", "#D0D0D0"))
        ps["minor_grid_opacity"] = st.slider("Minor opacity", 0.0, 1.0, float(ps.get("minor_grid_opacity", 0.4)))

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.2)))
        ps["marker_size"] = st.slider("Global marker size", 0, 14, int(ps.get("marker_size", 6)))

        st.markdown("---")
        st.markdown("**Colors**")
        palettes = ["Plotly", "D3", "G10", "T10", "Dark2", "Set1", "Set2",
                    "Pastel1", "Bold", "Prism", "Custom"]
        ps["palette"] = st.selectbox("Palette", palettes,
                                     index=palettes.index(ps.get("palette", "Plotly")))
        if ps["palette"] == "Custom":
            st.caption("Custom hex colors:")
            current = ps.get("custom_colors", []) or ["#1f77b4", "#ff7f0e", "#2ca02c"]
            new_cols = []
            cols_ui = st.columns(3)
            for i, c in enumerate(current):
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c,
                                                          key=f"cust_color_speciso_{i}"))
            ps["custom_colors"] = new_cols

        st.markdown("---")
        st.markdown("**Theme**")
        templates = ["plotly_white", "simple_white", "plotly_dark"]
        ps["template"] = st.selectbox("Template", templates,
                                      index=templates.index(ps.get("template", "plotly_white")))

        st.markdown("---")
        st.markdown("**Per-curve overrides (optional)**")
        ps["enable_per_curve_style"] = st.checkbox("Enable per-curve overrides",
                                                   bool(ps.get("enable_per_curve_style", False)))
        if ps["enable_per_curve_style"]:
            dash_opts = ["solid", "dot", "dash", "dashdot", "longdash"]
            marker_opts = ["circle", "square", "diamond", "cross", "x",
                           "triangle-up", "triangle-down", "star"]
            with st.container(border=True):
                for c in curves:
                    s = ps["per_curve_style_spec_iso"][c]
                    st.markdown(f"`{c}`")
                    o1, o2, o3, o4, o5 = st.columns([1,1,1,1,1])
                    with o1:
                        s["enabled"] = st.checkbox("Override", value=s["enabled"],
                                                   key=f"speciso_over_on_{c}")
                    with o2:
                        s["color"] = st.color_picker("Color", value=s["color"] or "#000000",
                                                     key=f"speciso_over_color_{c}",
                                                     disabled=not s["enabled"])
                    with o3:
                        s["width"] = st.slider("Width", 0.5, 8.0,
                                               float(s["width"] or ps["line_width"]),
                                               key=f"speciso_over_width_{c}",
                                               disabled=not s["enabled"])
                    with o4:
                        s["dash"] = st.selectbox("Dash", dash_opts,
                                                 index=dash_opts.index(s["dash"] or "solid"),
                                                 key=f"speciso_over_dash_{c}",
                                                 disabled=not s["enabled"])
                    with o5:
                        s["marker"] = st.selectbox("Marker", marker_opts,
                                                   index=marker_opts.index(s["marker"] or "circle"),
                                                   key=f"speciso_over_marker_{c}",
                                                   disabled=not s["enabled"])
                    s["msize"] = st.slider("Marker size", 0, 18,
                                           int(s["msize"] or ps["marker_size"]),
                                           key=f"speciso_over_msize_{c}",
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

def _resolve_curve_style(curve, idx, colors, ps):
    default_color = colors[idx % len(colors)]
    default_width = ps["line_width"]
    default_dash = "solid"
    default_marker = "circle"
    default_msize = ps["marker_size"]

    if not ps.get("enable_per_curve_style", False):
        return default_color, default_width, default_dash, default_marker, default_msize

    s = ps.get("per_curve_style_spec_iso", {}).get(curve, {})
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
# Export system
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
        fmts = st.multiselect("Formats", list(_EXPORT_FORMATS.keys()),
                              default=["PNG (raster)", "PDF (vector)", "SVG (vector)"],
                              key=f"{base_name}_fmts")
        c1, c2, c3 = st.columns(3)
        with c1:
            scale = st.slider("Scale", 1.0, 6.0, 3.0, 0.5, key=f"{base_name}_scale")
        with c2:
            width_px = st.number_input("Width px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_wpx")
        with c3:
            height_px = st.number_input("Height px (0=auto)", 0, 6000, 0, 100, key=f"{base_name}_hpx")

        if st.button("Export", key=f"{base_name}_doexport"):
            errors = []
            for fl in fmts:
                ext = _EXPORT_FORMATS[fl]
                out = out_dir / f"{base_name}.{ext}"
                try:
                    if ext == "html":
                        fig.write_html(str(out))
                    else:
                        kwargs = {}
                        if width_px > 0: kwargs["width"] = int(width_px)
                        if height_px > 0: kwargs["height"] = int(height_px)
                        fig.write_image(str(out), scale=scale, **kwargs)
                except Exception as e:
                    errors.append((out.name, str(e)))

            if errors:
                st.error("Some exports failed (install kaleido):\n" +
                         "\n".join([f"- {n}: {m}" for n, m in errors]))
            else:
                st.success("Exports saved in dataset folder.")


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

        if d.shape[1] >= 7:
            IC = d[:, 6]   # derivative IC from Fortran
        else:
            IC = np.divide(E22, E11, out=np.zeros_like(E22), where=E11 != 0)

        valid = (k > 0) & np.isfinite(IC)
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
    st.title("📈 Isotropy Validation — Spectral")

    data_dir = st.session_state.get("data_directory", None)
    if not data_dir:
        st.warning("Please select a data directory from the Overview page.")
        return
    data_dir = Path(data_dir)

    st.session_state.setdefault("spec_iso_legends", {
        "IC": r"$IC(k)$ (time-avg)",
        "IC_snap": r"$IC(k)$ snapshots",
        "E11": r"$E_{11}(k)$",
        "E22": r"$E_{22}(k)$",
        "E33": r"$E_{33}(k)$",
    })
    st.session_state.setdefault("axis_labels_spec_iso", {
        "k": r"$k$",
        "ic": r"$IC(k)$",
        "ek": r"$E_{ii}(k)$",
    })
    st.session_state.setdefault("plot_style", _default_plot_style())

    if st.session_state.get("_last_speciso_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        merged = _default_plot_style()
        merged.update(st.session_state.plot_style or {})
        st.session_state.plot_style = merged
        st.session_state["_last_speciso_dir"] = str(data_dir)

    # Find isotropy files
    files = detect_simulation_files(str(data_dir))
    ic_files = files.get("isotropy_coeff", [])
    if not ic_files:
        ic_files = glob.glob(str(data_dir / "isotropy_coeff_*.dat"))
    ic_files = sorted(ic_files, key=natural_sort_key)

    if not ic_files:
        st.info("No isotropy_coeff_*.dat files found.")
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
                    "IC": r"$IC(k)$ (time-avg)",
                    "IC_snap": r"$IC(k)$ snapshots",
                    "E11": r"$E_{11}(k)$",
                    "E22": r"$E_{22}(k)$",
                    "E33": r"$E_{33}(k)$",
                }
                st.session_state.axis_labels_spec_iso = {
                    "k": r"$k$",
                    "ic": r"$IC(k)$",
                    "ek": r"$E_{ii}(k)$",
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
    plot_style_sidebar(data_dir, curves)
    ps = st.session_state.plot_style
    colors = _get_palette(ps)

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
        st.subheader("Time-averaged Isotropy Coefficient")

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
                    IC0 = np.divide(d[:,2], d[:,1], out=np.zeros_like(d[:,2]), where=d[:,1]!=0)

                fig_ic.add_trace(go.Scatter(
                    x=k0, y=IC0, mode="lines",
                    name=st.session_state.spec_iso_legends["IC_snap"],
                    line=dict(color="rgba(0,0,0,0.15)", width=1),
                    showlegend=(i==0)
                ))

        c, lw, dash, mk, ms = _resolve_curve_style("IC", 0, colors, ps)
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

        fig_ic.update_layout(
            xaxis_title=st.session_state.axis_labels_spec_iso["k"],
            yaxis_title=st.session_state.axis_labels_spec_iso["ic"],
            xaxis_type="log",
            height=600,
        )
        fig_ic = apply_plot_style(fig_ic, ps)
        st.plotly_chart(fig_ic, use_container_width=True)
        export_panel(fig_ic, data_dir, "spectral_isotropy_IC")

    # ======================================================
    # Tab 2: Component spectra
    # ======================================================
    with tabs[1]:
        st.subheader("Component Spectra (time-avg)")

        if not show_component_spectra or avg["E11_mean"] is None:
            st.info("Component spectra not available (missing columns or disabled).")
        else:
            fig_eii = go.Figure()
            for i, curve in enumerate(["E11","E22","E33"]):
                arr = avg[f"{curve}_mean"]
                c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors, ps)
                fig_eii.add_trace(go.Scatter(
                    x=k, y=arr, mode="lines",
                    name=st.session_state.spec_iso_legends[curve],
                    line=dict(color=c, width=lw, dash=dash),
                ))

            fig_eii.update_layout(
                xaxis_title=st.session_state.axis_labels_spec_iso["k"],
                yaxis_title=st.session_state.axis_labels_spec_iso["ek"],
                xaxis_type="log",
                yaxis_type="log",
                height=600,
            )
            fig_eii = apply_plot_style(fig_eii, ps)
            st.plotly_chart(fig_eii, use_container_width=True)
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
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download summary CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="spectral_isotropy_summary.csv",
            mime="text/csv"
        )

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(r"""
Isotropy coefficient in spectral space:
\[
IC(k)=\frac{E_{22}(k)}{E_{11}(k)}
\]
For isotropic turbulence:
\[
E_{11}(k)=E_{22}(k)=E_{33}(k) \;\Rightarrow\; IC(k)\approx 1
\]
Derivative-based IC from Fortran provides a more robust estimate when available.
        """)


if __name__ == "__main__":
    main()

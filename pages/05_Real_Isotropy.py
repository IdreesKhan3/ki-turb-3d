"""
Isotropy Validation (Real Space) Page — Streamlit

High-standard features:
- Reads real-space isotropy files:
    * eps_real_validation.csv (required)
    * reynolds_stress_validation.csv (optional)
- Computes anisotropy tensor b_ij and Pope/Lumley invariants
- Produces 6 interactive subplots like your simple script:
    (a) Energy fractions vs t/t0 + moving averages + tolerance bands
    (b) Lumley triangle (xi, eta) trajectory
    (c) b11, b22, b33 vs t/t0
    (d) |b12|, |b13|, |b23| + anisotropy index
    (e) energy-fraction deviations from isotropy
    (f) convergence (running std)
- FULL persistent UI controls (same system as other pages)
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
import json
import sys


# --- Project imports ---
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.file_detector import detect_simulation_files
from utils.theme_config import inject_theme_css
from utils.report_builder import capture_button
from utils.plot_style import render_axis_limits_ui, apply_axis_limits, render_figure_size_ui, apply_figure_size
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
            "lumley_x": "ξ = (III<sub>b</sub>/2)<sup>1/3</sup>",
            "lumley_y": "η = (-II<sub>b</sub>/3)<sup>1/2</sup>",
        }
        # Start with defaults, then update with saved values
        loaded_legends = default_legends.copy()
        loaded_legends.update(meta.get("real_iso_legends", {}))
        st.session_state.real_iso_legends = loaded_legends
        
        loaded_axis_labels = default_axis_labels.copy()
        loaded_axis_labels.update(meta.get("axis_labels_real_iso", {}))
        st.session_state.axis_labels_real_iso = loaded_axis_labels
        
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
        "real_iso_legends": st.session_state.get("real_iso_legends", {}),
        "axis_labels_real_iso": st.session_state.get("axis_labels_real_iso", {}),
        "plot_styles": st.session_state.get("plot_styles", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json: {e}")


# ==========================================================
# Plot styling system (shared keys with other pages)
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
        "margin_left": 60,
        "margin_right": 20,
        "margin_top": 40,
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
            l=ps.get("margin_left", 60),
            r=ps.get("margin_right", 20),
            t=ps.get("margin_top", 40),
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

def _normalize_plot_name(plot_name: str) -> str:
    """Normalize plot name to a valid key format."""
    return plot_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace("/", "_")

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
# Physics / isotropy computations
# ==========================================================
def load_turbulence_data(csv_path: Path):
    df = pd.read_csv(csv_path)

    # robust numeric parse
    for col in ["iter", "iter_norm", "TKE_real", "u_rms_real", "eps_real",
                "frac_x", "frac_y", "frac_z"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # fallback mapping if columns are different
    # your eps_real_validation.csv mapping:
    # iter,iter_norm,eps_real,eps_spectral,TKE_real,u_rms_real,...,frac_x,frac_y,frac_z
    cols = df.columns.tolist()
    if "frac_x" not in cols and len(cols) >= 20:
        df["frac_x"] = pd.to_numeric(df.iloc[:, 17], errors="coerce")
        df["frac_y"] = pd.to_numeric(df.iloc[:, 18], errors="coerce")
        df["frac_z"] = pd.to_numeric(df.iloc[:, 19], errors="coerce")

    data = {
        "iter": df["iter"].to_numpy(),
        "iter_norm": df.get("iter_norm", df["iter"]).to_numpy(),
        "TKE": df.get("TKE_real", df.iloc[:, 4]).to_numpy(),
        "u_rms": df.get("u_rms_real", df.iloc[:, 5]).to_numpy(),
        "eps0": df.get("eps_real", df.iloc[:, 2]).to_numpy(),
        "frac_x": df["frac_x"].to_numpy(),
        "frac_y": df["frac_y"].to_numpy(),
        "frac_z": df["frac_z"].to_numpy(),
    }
    return data

def load_reynolds_stress(stress_path: Path, turb):
    if not stress_path.exists():
        return compute_reynolds_from_fractions(turb)

    df = pd.read_csv(stress_path)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna()
    n = min(len(df), len(turb["iter"]))

    R11, R22, R33 = df.iloc[:n, 1], df.iloc[:n, 2], df.iloc[:n, 3]
    R12, R13, R23 = df.iloc[:n, 4], df.iloc[:n, 5], df.iloc[:n, 6]
    TKE_from_R = 0.5 * (R11 + R22 + R33)

    return {
        "R11": R11.to_numpy(),
        "R22": R22.to_numpy(),
        "R33": R33.to_numpy(),
        "R12": R12.to_numpy(),
        "R13": R13.to_numpy(),
        "R23": R23.to_numpy(),
        "TKE": TKE_from_R.to_numpy(),
    }

def compute_reynolds_from_fractions(turb):
    TKE = turb["TKE"]
    R11 = turb["frac_x"] * 2 * TKE
    R22 = turb["frac_y"] * 2 * TKE
    R33 = turb["frac_z"] * 2 * TKE
    n = len(TKE)
    return dict(R11=R11, R22=R22, R33=R33,
                R12=np.zeros(n), R13=np.zeros(n), R23=np.zeros(n),
                TKE=TKE)

def anisotropy_tensor(R):
    k = R["TKE"]
    k_safe = np.where(k > 1e-10, k, 1e-10)

    b11 = R["R11"]/(2*k_safe) - 1/3
    b22 = R["R22"]/(2*k_safe) - 1/3
    b33 = R["R33"]/(2*k_safe) - 1/3
    b12 = R["R12"]/(2*k_safe)
    b13 = R["R13"]/(2*k_safe)
    b23 = R["R23"]/(2*k_safe)

    return dict(b11=b11, b22=b22, b33=b33, b12=b12, b13=b13, b23=b23)

def invariants(b):
    II_b = -0.5 * (
        b["b11"]**2 + b["b22"]**2 + b["b33"]**2 +
        2*(b["b12"]**2 + b["b13"]**2 + b["b23"]**2)
    )
    III_b = (1/3) * (
        b["b11"]**3 + b["b22"]**3 + b["b33"]**3 +
        3*b["b11"]*(b["b12"]**2 + b["b13"]**2) +
        3*b["b22"]*(b["b12"]**2 + b["b23"]**2) +
        3*b["b33"]*(b["b13"]**2 + b["b23"]**2) +
        6*b["b12"]*b["b13"]*b["b23"]
    )
    anis_index = np.sqrt(-2*II_b)
    eta = np.sqrt(-II_b/3)
    xi = np.cbrt(III_b/2)
    return dict(II_b=II_b, III_b=III_b, anis_index=anis_index, xi=xi, eta=eta)


# ==========================================================
# Page main
# ==========================================================
def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    
    st.title("⚖️ Isotropy Validation — Real Space")

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

    if st.session_state.get("_last_realiso_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        # After loading, ensure all required keys are present
        for key, value in default_legends.items():
            if key not in st.session_state.real_iso_legends:
                st.session_state.real_iso_legends[key] = value
        for key, value in default_axis_labels.items():
            if key not in st.session_state.axis_labels_real_iso:
                st.session_state.axis_labels_real_iso[key] = value
        if "plot_styles" not in st.session_state:
            st.session_state.plot_styles = {}
        st.session_state["_last_realiso_dir"] = str(data_dir)

    # locate required file
    files = detect_simulation_files(str(data_dir))
    eps_file = None
    
    # First, check files detected by file_detector (eps_validation key)
    for f in files.get("eps_validation", []):
        if Path(f).name == "eps_real_validation.csv" or Path(f).name.startswith("eps_real_validation"):
            eps_file = Path(f)
            break
    
    # If not found, check for exact filename in directory
    if eps_file is None:
        exact_file = data_dir / "eps_real_validation.csv"
        if exact_file.exists():
            eps_file = exact_file
    
    # If still not found, check for any eps_real_validation*.csv file
    if eps_file is None:
        import glob
        pattern = str(data_dir / "eps_real_validation*.csv")
        matches = glob.glob(pattern)
        if matches:
            eps_file = Path(matches[0])  # Use first match
    
    if eps_file is None or not eps_file.exists():
        st.error(f"eps_real_validation.csv not found in dataset folder: {data_dir}")
        st.info(f"💡 Looking for files matching: eps_real_validation*.csv")
        st.info(f"📂 Current directory: {data_dir}")
        # Show what files are actually in the directory
        csv_files = list(data_dir.glob("*.csv"))
        if csv_files:
            st.write("Available CSV files in directory:")
            for f in csv_files:
                st.write(f"  - {f.name}")
        return

    stress_file = data_dir / "reynolds_stress_validation.csv"

    turb = load_turbulence_data(eps_file)
    R = load_reynolds_stress(stress_file, turb)
    b = anisotropy_tensor(R)
    inv = invariants(b)

    t0_raw = turb["iter"][0] if turb["iter"][0] != 0 else 1.0
    time_norm = turb["iter"] / t0_raw

    # Sidebar: labels/legends persistence
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Curve names")
        for k in st.session_state.real_iso_legends:
            st.session_state.real_iso_legends[k] = st.text_input(
                k, st.session_state.real_iso_legends[k], key=f"realiso_leg_{k}"
            )

        st.markdown("---")
        st.markdown("### Axis labels")
        for k in st.session_state.axis_labels_real_iso:
            st.session_state.axis_labels_real_iso[k] = st.text_input(
                k, st.session_state.axis_labels_real_iso[k], key=f"realiso_ax_{k}"
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save labels/legends"):
                _save_ui_metadata(data_dir)
                st.success("Saved to legend_names.json")
        with c2:
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
                }
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")

    # Sidebar: analysis controls
    st.sidebar.subheader("Analysis Controls")
    stationary_iter = st.sidebar.number_input("Stationarity iteration", value=50000.0, step=5000.0)
    stationary_t = stationary_iter / t0_raw

    tol_list = st.sidebar.multiselect("Tolerance bands (fraction)", [0.005, 0.01, 0.02],
                                      default=[0.005, 0.01, 0.02])

    ma_win = st.sidebar.slider("Moving average window (0=off)", 0, 500, 0, 5)

    # curve list for overrides
    curves = ["Ex","Ey","Ez","b11","b22","b33","b12","b13","b23","anis","devx","devy","devz","maxdev"]
    plot_names = ["Energy Fractions (A)", "Lumley Triangle (B)", "Diagonal b_ii (C)", 
                  "Cross-correlations (D)", "Deviations (E)", "Convergence (F)"]
    plot_style_sidebar(data_dir, curves, plot_names)

    # Layout - 3 rows x 2 cols like your matplotlib GridSpec
    st.markdown("### Real-space isotropy diagnostics")
    colA, colB = st.columns([3,1])

    # ======================================================
    # (a) Temporal energy fractions
    # ======================================================
    with colA:
        # Get plot-specific style
        plot_name_a = "Energy Fractions (A)"
        ps_a = get_plot_style(plot_name_a)
        colors_a = _get_palette(ps_a)
        
        fig_a = go.Figure()
        E_x, E_y, E_z = turb["frac_x"], turb["frac_y"], turb["frac_z"]

        for i, (curve, arr) in enumerate([("Ex",E_x),("Ey",E_y),("Ez",E_z)]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_a, ps_a, plot_name_a)
            fig_a.add_trace(go.Scatter(
                x=time_norm, y=arr, mode="lines",
                name=st.session_state.real_iso_legends[curve],
                line=dict(color=c, width=lw, dash=dash),
            ))

        # Moving average (optional)
        if ma_win and ma_win > 1 and len(E_x) > ma_win:
            def _ma(x):
                k = np.ones(ma_win)/ma_win
                return np.convolve(x, k, mode="valid")
            t_ma = time_norm[ma_win//2: ma_win//2 + len(_ma(E_x))]

            for i, (curve, arr) in enumerate([("Ex",E_x),("Ey",E_y),("Ez",E_z)]):
                c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_a, ps_a, plot_name_a)
                fig_a.add_trace(go.Scatter(
                    x=t_ma, y=_ma(arr), mode="lines",
                    showlegend=False,
                    line=dict(color=c, width=max(1,lw*1.4)),
                ))

        # isotropic line + tolerance
        fig_a.add_hline(y=1/3, line_dash="dash", line_color="red", line_width=1.5)
        for tol in tol_list:
            fig_a.add_hrect(y0=1/3-tol, y1=1/3+tol, fillcolor="red", opacity=0.12, line_width=0)

        fig_a.add_vline(x=stationary_t, line_dash="dash", line_color="purple", line_width=1.2)

        layout_kwargs_a = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["energy_frac"],
            height=420,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs_a = apply_axis_limits(layout_kwargs_a, ps_a)
        layout_kwargs_a = apply_figure_size(layout_kwargs_a, ps_a)
        fig_a.update_layout(**layout_kwargs_a)
        fig_a = apply_plot_style(fig_a, ps_a)
        st.plotly_chart(fig_a, width='stretch')
        capture_button(fig_a, title="Real-Space Isotropy Analysis (Part A)", source_page="Real Isotropy")
        export_panel(fig_a, data_dir, "real_iso_energy_fractions")

    # ======================================================
    # (b) Lumley triangle
    # ======================================================
    with colB:
        # Get plot-specific style
        plot_name_b = "Lumley Triangle (B)"
        ps_b = get_plot_style(plot_name_b)
        
        fig_b = go.Figure()
        xi, eta = inv["xi"], inv["eta"]

        # realizability boundaries
        xi_vals = np.linspace(-1/6, 1/3, 300)
        eta_two = np.sqrt(1/27 + 2*xi_vals**3)
        eta_low = np.where(xi_vals < 0, -xi_vals, xi_vals)

        fig_b.add_trace(go.Scatter(x=xi_vals, y=eta_two, mode="lines", line=dict(color="red", width=2),
                                   name="Two-comp limit", showlegend=False))
        fig_b.add_trace(go.Scatter(x=xi_vals, y=eta_low, mode="lines", line=dict(color="black", width=2),
                                   name="Axisym limits", showlegend=False))

        fig_b.add_trace(go.Scatter(
            x=xi, y=eta, mode="lines+markers",
            marker=dict(size=4, color=np.linspace(0,1,len(xi)), colorscale="Viridis"),
            line=dict(color="rgba(0,0,0,0.4)", width=1),
            name="DNS trajectory"
        ))
        fig_b.add_trace(go.Scatter(x=[xi[0]], y=[eta[0]], mode="markers",
                                   marker=dict(size=9, color="red", line=dict(width=1,color="black")),
                                   name="Start"))
        fig_b.add_trace(go.Scatter(x=[xi[-1]], y=[eta[-1]], mode="markers",
                                   marker=dict(size=9, color="green", line=dict(width=1,color="black")),
                                   name="End"))

        layout_kwargs_b = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["lumley_x"],
            yaxis_title=st.session_state.axis_labels_real_iso["lumley_y"],
            height=420,  # Default, will be overridden if custom size is enabled
            showlegend=False,
        )
        layout_kwargs_b = apply_axis_limits(layout_kwargs_b, ps_b)
        layout_kwargs_b = apply_figure_size(layout_kwargs_b, ps_b)
        fig_b.update_layout(**layout_kwargs_b)
        fig_b = apply_plot_style(fig_b, ps_b)
        st.plotly_chart(fig_b, width='stretch')
        capture_button(fig_b, title="Real-Space Isotropy Analysis (Part B)", source_page="Real Isotropy")
        export_panel(fig_b, data_dir, "real_iso_lumley_triangle")


    # ======================================================
    # Second row (c) + (d)
    # ======================================================
    colC, colD = st.columns(2)

    with colC:
        # Get plot-specific style
        plot_name_c = "Diagonal b_ii (C)"
        ps_c = get_plot_style(plot_name_c)
        colors_c = _get_palette(ps_c)
        
        fig_c = go.Figure()
        for i, curve in enumerate(["b11","b22","b33"]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_c, ps_c, plot_name_c)
            fig_c.add_trace(go.Scatter(
                x=time_norm, y=b[curve], mode="lines",
                name=st.session_state.real_iso_legends[curve],
                line=dict(color=c, width=lw, dash=dash),
            ))
        fig_c.add_hline(y=0, line_dash="dash", line_color="black")
        layout_kwargs_c = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["bij"],
            height=360,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs_c = apply_axis_limits(layout_kwargs_c, ps_c)
        layout_kwargs_c = apply_figure_size(layout_kwargs_c, ps_c)
        fig_c.update_layout(**layout_kwargs_c)
        fig_c = apply_plot_style(fig_c, ps_c)
        st.plotly_chart(fig_c, width='stretch')
        export_panel(fig_c, data_dir, "real_iso_bii_diag")

    with colD:
        # Get plot-specific style
        plot_name_d = "Cross-correlations (D)"
        ps_d = get_plot_style(plot_name_d)
        colors_d = _get_palette(ps_d)
        
        fig_d = go.Figure()
        for i, curve in enumerate(["b12","b13","b23"]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_d, ps_d, plot_name_d)
            fig_d.add_trace(go.Scatter(
                x=time_norm, y=np.abs(b[curve]), mode="lines",
                name=st.session_state.real_iso_legends[curve],
                line=dict(color=c, width=lw, dash=dash),
            ))
        c, lw, dash, mk, ms = _resolve_curve_style("anis", 3, colors_d, ps_d, plot_name_d)
        fig_d.add_trace(go.Scatter(
            x=time_norm, y=inv["anis_index"], mode="lines",
            name=st.session_state.real_iso_legends["anis"],
            line=dict(color=c, width=max(2,lw*1.2)),
        ))
        layout_kwargs_d = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["cross"],
            yaxis_type="log",
            height=360,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs_d = apply_axis_limits(layout_kwargs_d, ps_d)
        layout_kwargs_d = apply_figure_size(layout_kwargs_d, ps_d)
        fig_d.update_layout(**layout_kwargs_d)
        fig_d = apply_plot_style(fig_d, ps_d)
        st.plotly_chart(fig_d, width='stretch')
        export_panel(fig_d, data_dir, "real_iso_cross_corr")


    # ======================================================
    # Third row (e) + (f)
    # ======================================================
    colE, colF = st.columns(2)

    with colE:
        # Get plot-specific style
        plot_name_e = "Deviations (E)"
        ps_e = get_plot_style(plot_name_e)
        colors_e = _get_palette(ps_e)
        
        fig_e = go.Figure()
        devx = np.abs(E_x - 1/3)
        devy = np.abs(E_y - 1/3)
        devz = np.abs(E_z - 1/3)
        maxdev = np.maximum(np.maximum(devx, devy), devz)

        for i,(curve,arr) in enumerate([("devx",devx),("devy",devy),("devz",devz)]):
            c, lw, dash, mk, ms = _resolve_curve_style(curve, i, colors_e, ps_e, plot_name_e)
            fig_e.add_trace(go.Scatter(
                x=time_norm, y=arr, mode="lines",
                name=curve,
                line=dict(color=c, width=lw, dash=dash)
            ))

        c, lw, dash, mk, ms = _resolve_curve_style("maxdev", 3, colors_e, ps_e, plot_name_e)
        fig_e.add_trace(go.Scatter(
            x=time_norm, y=maxdev, mode="lines",
            name="Max deviation",
            line=dict(color=c, width=max(2,lw))
        ))

        fig_e.add_vline(x=stationary_t, line_dash="dash", line_color="purple", line_width=1)

        layout_kwargs_e = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title=st.session_state.axis_labels_real_iso["dev"],
            yaxis_type="log",
            height=360,  # Default, will be overridden if custom size is enabled
        )
        layout_kwargs_e = apply_axis_limits(layout_kwargs_e, ps_e)
        layout_kwargs_e = apply_figure_size(layout_kwargs_e, ps_e)
        fig_e.update_layout(**layout_kwargs_e)
        fig_e = apply_plot_style(fig_e, ps_e)
        st.plotly_chart(fig_e, width='stretch')
        export_panel(fig_e, data_dir, "real_iso_deviation")

    with colF:
        # Get plot-specific style
        plot_name_f = "Convergence (F)"
        ps_f = get_plot_style(plot_name_f)
        
        fig_f = go.Figure()
        if len(E_x) > 20:
            conv_win = st.sidebar.slider("Convergence window", 10, max(20,len(E_x)//2), max(20,len(E_x)//10), 5)
            running_stds = []
            for i in range(conv_win, len(E_x)+1):
                sx = np.std(E_x[i-conv_win:i])
                sy = np.std(E_y[i-conv_win:i])
                sz = np.std(E_z[i-conv_win:i])
                running_stds.append((sx+sy+sz)/3.0)
            running_stds = np.asarray(running_stds)
            t_conv = time_norm[conv_win-1: conv_win-1+len(running_stds)]

            fig_f.add_trace(go.Scatter(
                x=t_conv, y=running_stds, mode="lines",
                name=f"Running σ (win={conv_win})",
                line=dict(color="black", width=2.2)
            ))

        layout_kwargs_f = dict(
            xaxis_title=st.session_state.axis_labels_real_iso["time"],
            yaxis_title="Running standard deviation",
            yaxis_type="log",
            height=360,  # Default, will be overridden if custom size is enabled
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
        st.markdown("**Energy fractions:**")
        st.latex(r"E_x/E_{tot}, \quad E_y/E_{tot}, \quad E_z/E_{tot}")
        st.markdown("Isotropy implies each approaches $1/3$.")
        
        st.markdown("**Reynolds stress anisotropy tensor:**")
        st.latex(r"b_{ij} = \frac{R_{ij}}{2k} - \frac{1}{3}\delta_{ij}")
        
        st.markdown("**Invariants (Pope 2000):**")
        st.latex(r"II_b = -\frac{1}{2}\mathrm{tr}(b^2), \qquad III_b = \frac{1}{3}\mathrm{tr}(b^3)")
        
        st.markdown("**Lumley coordinates:**")
        st.latex(r"\eta = \left(-\frac{II_b}{3}\right)^{1/2}, \quad \xi = \left(\frac{III_b}{2}\right)^{1/3}")
        
        st.markdown("**Anisotropy index:**")
        st.latex(r"A = \sqrt{-2 II_b}")


if __name__ == "__main__":
    main()

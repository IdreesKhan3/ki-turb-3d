"""
Energy Balance / Residual Error Page (Streamlit)

High-standard features:
- Reads eps_real_validation CSV files from multiple simulations
- Groups by simulation prefix (eps_real_validation_data1, data2, ...)
- Plots energy_balance_ratio residual error vs t/t0
- Full persistent UI controls:
    * Legend names, axis labels
    * Fonts, tick style, major/minor grids, background colors, theme
    * Palette / custom colors
    * Per-simulation overrides: color/width/dash
- Research-grade export:
    * PNG/PDF/SVG/EPS/JPG/WEBP/TIFF + HTML
- Robust to missing columns/files
"""

import streamlit as st
import numpy as np
import pandas as pd
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

from utils.file_detector import (
    detect_simulation_files,
    group_files_by_simulation,
    natural_sort_key
)

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
        st.session_state.energy_legend_names = meta.get("energy_legends", {})
        st.session_state.axis_labels_energy = meta.get("axis_labels_energy", {})
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
        "energy_legends": st.session_state.get("energy_legend_names", {}),
        "axis_labels_energy": st.session_state.get("axis_labels_energy", {}),
        "plot_style": st.session_state.get("plot_style", {}),
    })

    try:
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
    except Exception as e:
        st.error(f"Could not save legend_names.json (read-only folder?): {e}")


# ==========================================================
# Plot styling (shared keys with other pages)
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

        "line_width": 2.2,

        "palette": "Plotly",
        "custom_colors": ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
                          "#8c564b", "#e377c2", "#7f7f7f"],
        "template": "plotly_white",

        "enable_per_sim_style": False,
        "per_sim_style_energy": {},  # {sim: {enabled,color,width,dash}}
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
    ps.setdefault("per_sim_style_energy", {})
    for k in sim_groups.keys():
        ps["per_sim_style_energy"].setdefault(k, {
            "enabled": False,
            "color": None,
            "width": None,
            "dash": "solid",
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
                                             key="minor_grid_dash_energy")
        ps["minor_grid_color"] = st.color_picker("Minor grid color", ps.get("minor_grid_color", "#D0D0D0"))
        ps["minor_grid_opacity"] = st.slider("Minor grid opacity", 0.0, 1.0,
                                             float(ps.get("minor_grid_opacity", 0.45)))

        st.markdown("---")
        st.markdown("**Curves**")
        ps["line_width"] = st.slider("Global line width", 0.5, 7.0, float(ps.get("line_width", 2.2)))

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
                new_cols.append(cols_ui[i % 3].text_input(f"Color {i+1}", c, key=f"cust_color_energy_{i}"))
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
            with st.container(border=True):
                for sim_prefix in sorted(sim_groups.keys()):
                    s = ps["per_sim_style_energy"][sim_prefix]
                    st.markdown(f"`{sim_prefix}`")
                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        s["enabled"] = st.checkbox("Override", value=s.get("enabled", False),
                                                   key=f"energy_over_on_{sim_prefix}")
                    with c2:
                        s["color"] = st.color_picker("Color", value=s.get("color") or "#000000",
                                                     key=f"energy_over_color_{sim_prefix}",
                                                     disabled=not s["enabled"])
                    with c3:
                        s["width"] = st.slider("Width", 0.5, 8.0,
                                               float(s.get("width") or ps["line_width"]),
                                               key=f"energy_over_width_{sim_prefix}",
                                               disabled=not s["enabled"])
                    s["dash"] = st.selectbox("Dash", dash_opts,
                                             index=dash_opts.index(s.get("dash") or "solid"),
                                             key=f"energy_over_dash_{sim_prefix}",
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

    if not ps.get("enable_per_sim_style", False):
        return default_color, default_width, default_dash

    s = ps.get("per_sim_style_energy", {}).get(sim_prefix, {})
    if not s.get("enabled", False):
        return default_color, default_width, default_dash

    color = s.get("color") or default_color
    width = float(s.get("width") or default_width)
    dash = s.get("dash") or default_dash
    return color, width, dash


# ==========================================================
# Export (research formats)
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
# Page main
# ==========================================================
def main():
    st.title("⚡ Energy Balance / Residual Error")

    data_dir = st.session_state.get("data_directory", None)
    if not data_dir:
        st.warning("Please select a data directory from the Overview page.")
        return
    data_dir = Path(data_dir)

    # Defaults
    st.session_state.setdefault("energy_legend_names", {})
    st.session_state.setdefault("axis_labels_energy", {
        "x": r"$t/t_0$",
        "y": r"$\left| \frac{\varepsilon}{-dE_k/dt + \langle f\cdot u \rangle} - 1 \right|$",
        "title": "Energy balance residual",
    })
    st.session_state.setdefault("plot_style", _default_plot_style())

    # Load json once per dataset change
    if st.session_state.get("_last_energy_dir") != str(data_dir):
        _load_ui_metadata(data_dir)
        merged = _default_plot_style()
        merged.update(st.session_state.plot_style or {})
        st.session_state.plot_style = merged
        st.session_state["_last_energy_dir"] = str(data_dir)

    ps = st.session_state.plot_style

    # Detect files
    files_dict = detect_simulation_files(str(data_dir))
    eps_files = files_dict.get("eps_real", [])
    if not eps_files:
        eps_files = glob.glob(str(data_dir / "eps_real_validation*.csv"))

    if not eps_files:
        st.info("No eps_real_validation*.csv found.")
        return

    # Group by simulation name
    sim_groups = group_files_by_simulation(
        sorted([str(f) for f in eps_files], key=natural_sort_key),
        r"(eps_real_validation[_\w]*\d+)\.csv"
    )

    if not sim_groups:
        # fallback: treat each file as its own sim
        sim_groups = {Path(f).stem: [f] for f in eps_files}

    # Sidebar legends + axis labels (persistent)
    with st.sidebar.expander("Legend & Axis Labels (persistent)", expanded=False):
        st.markdown("### Legend names")
        for sim_prefix in sorted(sim_groups.keys()):
            st.session_state.energy_legend_names.setdefault(sim_prefix, _default_labelify(sim_prefix))
            st.session_state.energy_legend_names[sim_prefix] = st.text_input(
                f"Name for `{sim_prefix}`",
                value=st.session_state.energy_legend_names[sim_prefix],
                key=f"legend_energy_{sim_prefix}"
            )

        st.markdown("---")
        st.markdown("### Axis labels")
        st.session_state.axis_labels_energy["x"] = st.text_input(
            "x-label", st.session_state.axis_labels_energy["x"], key="ax_energy_x"
        )
        st.session_state.axis_labels_energy["y"] = st.text_input(
            "y-label", st.session_state.axis_labels_energy["y"], key="ax_energy_y"
        )
        st.session_state.axis_labels_energy["title"] = st.text_input(
            "plot title", st.session_state.axis_labels_energy["title"], key="ax_energy_title"
        )

        b1, b2 = st.columns(2)
        with b1:
            if st.button("💾 Save labels/legends"):
                _save_ui_metadata(data_dir)
                st.success("Saved to legend_names.json")
        with b2:
            if st.button("♻️ Reset labels/legends"):
                st.session_state.energy_legend_names = {k: _default_labelify(k) for k in sim_groups.keys()}
                st.session_state.axis_labels_energy.update({
                    "x": r"$t/t_0$",
                    "y": r"$\left| \frac{\varepsilon}{-dE_k/dt + \langle f\cdot u \rangle} - 1 \right|$",
                    "title": "Energy balance residual",
                })
                _save_ui_metadata(data_dir)
                st.toast("Reset + saved.", icon="♻️")

    # Sidebar time normalization & window
    st.sidebar.subheader("Time / Window")
    T0 = st.sidebar.number_input("t₀ normalization constant", value=1000.0, min_value=1.0, step=100.0)

    # Determine max common length for window indexing
    min_len = min(len(v) for v in sim_groups.values())
    start_idx = st.sidebar.slider("Start file index", 1, min_len, 1)
    end_idx = st.sidebar.slider("End file index", start_idx, min_len, min_len)

    st.sidebar.subheader("Plot Options")
    y_max = st.sidebar.slider("Y max", 0.05, 2.0, 0.5, 0.05)
    show_abs = st.sidebar.checkbox("Use absolute residual", value=True)
    smooth_win = st.sidebar.slider("Optional moving average window (0=off)", 0, 500, 0, 10)

    # Full style sidebar
    plot_style_sidebar(data_dir, sim_groups)
    ps = st.session_state.plot_style
    colors = _get_palette(ps)

    # Main plot
    st.subheader("Residual error across simulations")

    fig = go.Figure()
    plotted_any = False

    summary_rows = []

    for idx, sim_prefix in enumerate(sorted(sim_groups.keys())):
        files = sim_groups[sim_prefix][start_idx-1:end_idx]
        if not files:
            continue

        # load & concat in time
        frames = []
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            frames.append(df)
        if not frames:
            continue

        df_all = pd.concat(frames, ignore_index=True)

        # robust numeric conversion
        for col in ["iter", "energy_balance_ratio"]:
            if col in df_all.columns:
                df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

        if "iter" not in df_all.columns or "energy_balance_ratio" not in df_all.columns:
            continue

        df_all = df_all.dropna(subset=["iter", "energy_balance_ratio"])
        if df_all.empty:
            continue

        t = df_all["iter"].values / float(T0)
        y = df_all["energy_balance_ratio"].values
        if show_abs:
            y = np.abs(y)

        # smoothing
        if smooth_win and smooth_win > 1 and len(y) > smooth_win:
            kernel = np.ones(int(smooth_win)) / int(smooth_win)
            y_sm = np.convolve(y, kernel, mode="valid")
            t_sm = t[int(smooth_win)//2: int(smooth_win)//2 + len(y_sm)]
            t_plot, y_plot = t_sm, y_sm
        else:
            t_plot, y_plot = t, y

        legend = st.session_state.energy_legend_names.get(sim_prefix, _default_labelify(sim_prefix))
        c, lw, dash = _resolve_line_style(sim_prefix, idx, colors, ps)

        fig.add_trace(go.Scatter(
            x=t_plot, y=y_plot,
            mode="lines",
            name=legend,
            line=dict(color=c, width=lw, dash=dash),
            hovertemplate="t/t0=%{x:.4g}<br>residual=%{y:.4g}<extra></extra>"
        ))
        plotted_any = True

        summary_rows.append({
            "simulation": legend,
            "mean_residual": float(np.mean(y_plot)),
            "std_residual": float(np.std(y_plot)),
            "final_residual": float(y_plot[-1]),
        })

    if not plotted_any:
        st.info("No valid energy balance data could be plotted.")
        return

    fig.update_layout(
        title=st.session_state.axis_labels_energy["title"],
        xaxis_title=st.session_state.axis_labels_energy["x"],
        yaxis_title=st.session_state.axis_labels_energy["y"],
        height=620,
        margin=dict(l=60, r=20, t=40, b=55),
        legend_title="Simulation",
    )
    fig.update_yaxes(range=[0, y_max])
    fig.update_xaxes(range=[0, None])

    fig = apply_plot_style(fig, ps)
    st.plotly_chart(fig, use_container_width=True)

    export_panel(fig, data_dir, base_name="energy_balance_residual")

    # Summary table
    st.markdown("### Summary (selected window)")
    if summary_rows:
        df_sum = pd.DataFrame(summary_rows).sort_values("mean_residual")
        st.dataframe(df_sum, use_container_width=True)

        st.download_button(
            "Download summary CSV",
            df_sum.to_csv(index=False).encode("utf-8"),
            file_name="energy_balance_summary.csv",
            mime="text/csv"
        )

    # Theory section
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(r"""
Energy balance residual is defined as:

\[
\mathrm{Res}(t) = \left| \frac{\varepsilon}{-dE_k/dt + \langle f\cdot u \rangle} - 1 \right|
\]

For statistically stationary forced HIT, the denominator should balance the dissipation, so
\(\mathrm{Res}(t)\to 0\) as stationarity and resolution improve.
        """)


if __name__ == "__main__":
    main()

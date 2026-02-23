"""
Other Turbulence Stats — Custom plot and tables views.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Any

from utils.report_builder import capture_button
from utils.export_figs import export_panel
from utils.plot_style import (
    resolve_line_style,
    _get_palette,
    get_tick_format,
    apply_axis_limits,
    apply_figure_size,
    default_plot_style,
    render_legend_axis_labels_ui,
)

from .plot_style import apply_plot_style, plot_style_sidebar


def render_custom_plot_section(
    data_dir: Path,
    all_dataframes: Dict[str, pd.DataFrame],
) -> None:
    """Render the custom multi-trace plotting section."""
    st.header("Custom Plotting")

    if not all_dataframes:
        st.info("No CSV files found. Please load data from the Overview page.")
        return

    sim_groups = {key: [] for key in all_dataframes.keys()}
    if sim_groups:
        plot_style_sidebar(data_dir, sim_groups)

    if "custom_plot_traces" not in st.session_state:
        st.session_state.custom_plot_traces = []
    if "custom_plot_legend_names" not in st.session_state:
        st.session_state.custom_plot_legend_names = {}
    if "custom_plot_axis_labels" not in st.session_state:
        st.session_state.custom_plot_axis_labels = {"x": "X", "y": "Y"}

    def _reset_legend_axis():
        st.session_state.custom_plot_legend_names = {}
        st.session_state.custom_plot_axis_labels = {"x": "X", "y": "Y"}

    legend_names, axis_labels = render_legend_axis_labels_ui(
        data_dir=None,
        traces=st.session_state.custom_plot_traces if st.session_state.custom_plot_traces else None,
        legend_names_key="custom_plot_legend_names",
        axis_labels_key="custom_plot_axis_labels",
        trace_key_func=lambda trace: f"{trace['data_source']}_{trace['x_col']}_{trace['y_col']}",
        save_callback=None,
        reset_callback=_reset_legend_axis,
        key_prefix="custom",
    )

    st.sidebar.subheader("Plot Options")
    use_abs = st.sidebar.checkbox("Use absolute value (Y-axis)", value=False, key="plot_use_abs")
    smooth_window = st.sidebar.slider(
        "Moving average window (0=off)",
        0,
        500,
        0,
        10,
        key="plot_smooth",
        help="Smooths the curve by averaging over N consecutive points. Reduces noise but also reduces the number of data points by (N-1). Example: window=5 averages every 5 points into 1 smoothed point.",
    )
    normalize_x = st.sidebar.checkbox("Normalize X-axis", value=False, key="plot_norm_x")
    x_norm = st.sidebar.number_input(
        "X normalization constant",
        value=1000.0,
        min_value=1.0,
        step=100.0,
        disabled=not normalize_x,
        key="plot_x_norm",
    )
    normalize_y = st.sidebar.checkbox(
        "Normalize Y-axis by maximum",
        value=False,
        key="plot_norm_y",
        help="Normalize each trace's Y values by its maximum value",
    )

    st.subheader("Add Traces to Plot")

    with st.expander("➕ Add New Trace", expanded=len(st.session_state.custom_plot_traces) == 0):
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1:
            data_source = st.selectbox(
                "📁 Data Source",
                options=list(all_dataframes.keys()),
                help="Select which CSV file to plot from",
                key="new_trace_source",
            )
        df_plot = all_dataframes[data_source]
        numeric_cols = [col for col in df_plot.columns if pd.api.types.is_numeric_dtype(df_plot[col])]

        if len(numeric_cols) < 2:
            st.warning(f"Not enough numeric columns in {data_source} for plotting.")
        else:
            with col2:
                x_col = st.selectbox(
                    "X-axis Column",
                    options=numeric_cols,
                    index=0,
                    help="Select column for X-axis",
                    key="new_trace_x",
                )
            with col3:
                y_col = st.selectbox(
                    "Y-axis Column",
                    options=[c for c in numeric_cols if c != x_col],
                    index=0,
                    help="Select column for Y-axis",
                    key="new_trace_y",
                )
            with col4:
                trace_label = st.text_input(
                    "Label",
                    value=f"{data_source.split('_')[-1]}: {y_col}",
                    help="Trace label for legend",
                    key="new_trace_label",
                )
            if st.button("Add Trace", key="add_trace_btn"):
                trace_config = {
                    "data_source": data_source,
                    "x_col": x_col,
                    "y_col": y_col,
                    "label": trace_label,
                }
                st.session_state.custom_plot_traces.append(trace_config)
                st.rerun()

    if st.session_state.custom_plot_traces:
        st.subheader("Current Traces")
        for idx, trace in enumerate(st.session_state.custom_plot_traces):
            with st.expander(f"Trace {idx+1}: {trace['label']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"Source: {trace['data_source']}")
                    st.text(f"X: {trace['x_col']} | Y: {trace['y_col']}")
                with col2:
                    if st.button("Remove", key=f"remove_trace_{idx}"):
                        st.session_state.custom_plot_traces.pop(idx)
                        st.rerun()
        if st.button("Clear All Traces", key="clear_all_traces"):
            st.session_state.custom_plot_traces = []
            st.rerun()
        st.markdown("---")

        _render_plot(
            all_dataframes=all_dataframes,
            legend_names=legend_names,
            axis_labels=axis_labels,
            use_abs=use_abs,
            smooth_window=smooth_window,
            normalize_x=normalize_x,
            x_norm=x_norm,
            normalize_y=normalize_y,
            data_dir=data_dir,
        )
    else:
        st.info(
            "👆 Add traces above to create a multi-trace plot. Each trace can use different files and columns."
        )


def _render_plot(
    all_dataframes: Dict[str, pd.DataFrame],
    legend_names: Dict[str, str],
    axis_labels: Dict[str, str],
    use_abs: bool,
    smooth_window: int,
    normalize_x: bool,
    x_norm: float,
    normalize_y: bool,
    data_dir: Path,
) -> None:
    """Build and display the multi-trace plot."""
    ps = st.session_state.get("plot_style", default_plot_style())
    colors = _get_palette(ps)
    fig = go.Figure()
    all_x_labels = set()
    all_y_labels = set()

    for idx, trace in enumerate(st.session_state.custom_plot_traces):
        data_source = trace["data_source"]
        x_col = trace["x_col"]
        y_col = trace["y_col"]
        trace_key = f"{data_source}_{x_col}_{y_col}"
        label = legend_names.get(
            trace_key,
            trace.get("label", f"{data_source.split('_')[-1]}: {y_col}"),
        )

        if data_source not in all_dataframes:
            continue
        df_plot = all_dataframes[data_source]
        if x_col not in df_plot.columns or y_col not in df_plot.columns:
            continue

        x_data = df_plot[x_col].values
        y_data = df_plot[y_col].values
        x_data = pd.to_numeric(x_data, errors="coerce")
        y_data = pd.to_numeric(y_data, errors="coerce")
        valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]

        if len(x_data) == 0 or len(y_data) == 0:
            continue

        if normalize_x:
            x_data = x_data / float(x_norm)
        if use_abs:
            y_data = np.abs(y_data)
        if normalize_y:
            y_max = np.max(np.abs(y_data)) if len(y_data) > 0 else 1.0
            if y_max > 0:
                y_data = y_data / y_max

        x_axis_type = ps.get("x_axis_type", "linear")
        y_axis_type = ps.get("y_axis_type", "linear")
        if x_axis_type == "log":
            log_x_mask = x_data > 0
            x_data = x_data[log_x_mask]
            y_data = y_data[log_x_mask]
        if y_axis_type == "log":
            log_y_mask = y_data > 0
            x_data = x_data[log_y_mask]
            y_data = y_data[log_y_mask]

        if len(x_data) == 0 or len(y_data) == 0:
            continue

        hover_x_label = x_col
        if normalize_x:
            hover_x_label = f"{x_col} (normalized)"
        hover_y_label = y_col
        if use_abs:
            hover_y_label = f"|{hover_y_label}|"
        if normalize_y:
            hover_y_label = f"{hover_y_label} / max"

        sim_prefix = data_source
        color, width, dash, marker, marker_size, override_on = resolve_line_style(
            sim_prefix,
            idx,
            colors,
            ps,
            style_key="per_sim_style_turb_stats",
            include_marker=True,
        )
        line_style = dict(width=width, color=color)
        if dash and dash != "solid":
            line_style["dash"] = dash
        mode = "lines"
        marker_dict = None
        if override_on and marker and marker != "none":
            mode = "lines+markers"
            marker_dict = dict(symbol=marker, size=marker_size)

        if smooth_window > 1 and len(y_data) > smooth_window:
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=y_data,
                    mode="lines",
                    name=f"{label} (original)",
                    line=dict(width=1.0, color=color),
                    opacity=0.3,
                    hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label} (original)=%{{y:.4g}}<extra></extra>",
                    showlegend=False,
                )
            )
            kernel = np.ones(int(smooth_window)) / int(smooth_window)
            y_smooth = np.convolve(y_data, kernel, mode="valid")
            x_smooth = x_data[int(smooth_window) // 2 : int(smooth_window) // 2 + len(y_smooth)]
            x_plot, y_plot = x_smooth, y_smooth
            hover_y_label_smooth = f"{hover_y_label} (smoothed)"
            scatter_kwargs = dict(
                x=x_plot,
                y=y_plot,
                mode=mode,
                name=label,
                line=line_style,
                hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label_smooth}=%{{y:.4g}}<extra></extra>",
            )
        else:
            x_plot, y_plot = x_data, y_data
            scatter_kwargs = dict(
                x=x_plot,
                y=y_plot,
                mode=mode,
                name=label,
                line=line_style,
                hovertemplate=f"{hover_x_label}=%{{x:.4g}}<br>{hover_y_label}=%{{y:.4g}}<extra></extra>",
                showlegend=True,
            )
        if marker_dict:
            scatter_kwargs["marker"] = marker_dict
        fig.add_trace(go.Scatter(**scatter_kwargs))
        all_x_labels.add(x_col)
        all_y_labels.add(y_col)

    if len(fig.data) == 0:
        st.warning("No valid traces to plot. Please add traces with valid data.")
        return

    custom_x_label = axis_labels.get("x", "X")
    custom_y_label = axis_labels.get("y", "Y")
    if custom_x_label and custom_x_label != "X":
        x_label = custom_x_label
        if normalize_x:
            x_label = f"{x_label} / {x_norm}"
    else:
        x_label = list(all_x_labels)[0] if len(all_x_labels) == 1 else "X"
        if normalize_x and len(all_x_labels) == 1:
            x_label = f"{x_label} / {x_norm}"
        elif len(all_x_labels) > 1:
            x_label = "X (multiple columns)"

    if custom_y_label and custom_y_label != "Y":
        y_label = custom_y_label
        if use_abs:
            y_label = f"|{y_label}|"
        if normalize_y:
            y_label = f"{y_label} / max"
    else:
        y_label = list(all_y_labels)[0] if len(all_y_labels) == 1 else "Y"
        if len(all_y_labels) > 1:
            y_label = "Y (multiple columns)"
        else:
            if use_abs:
                y_label = f"|{y_label}|"
            if normalize_y:
                y_label = f"{y_label} / max({y_label})"

    fig = apply_plot_style(fig, ps)
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    layout_kwargs = dict(
        height=400,
        margin=dict(l=60, r=20, t=40, b=55),
    )
    layout_kwargs = apply_axis_limits(layout_kwargs, ps)
    layout_kwargs = apply_figure_size(layout_kwargs, ps)
    if ps.get("show_legend", True):
        layout_kwargs["showlegend"] = True
    fig.update_layout(**layout_kwargs)
    fig.update_layout(
        plot_bgcolor=ps.get("plot_bgcolor", "#FFFFFF"),
        paper_bgcolor=ps.get("paper_bgcolor", "#FFFFFF"),
    )
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(zeroline=False)
    if ps.get("show_legend", True) and len(fig.data) > 0:
        fig.update_layout(showlegend=True)

    x_tick_format = get_tick_format(
        ps.get("x_tick_format", "auto"), ps.get("x_tick_decimals", 2), normalize_x
    )
    y_tick_format = get_tick_format(
        ps.get("y_tick_format", "auto"), ps.get("y_tick_decimals", 2), normalize_y
    )
    fig.update_xaxes(tickformat=x_tick_format, separatethousands=False)
    fig.update_yaxes(tickformat=y_tick_format, separatethousands=False)

    use_container = not ps.get("enable_custom_size", False)
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        st.plotly_chart(fig, width="stretch" if use_container else "content")
    capture_button(fig, title="Custom Multi-Trace Plot", source_page="Other Turbulence Stats")
    export_panel(fig, data_dir, base_name="custom_multi_trace_plot")


def render_tables_section(
    data_dirs: List[str],
    table_data: Dict[str, Dict[str, Any]],
) -> None:
    """Render turbulence statistics tables."""
    if not table_data:
        return
    turbulence_tables = {k: v for k, v in table_data.items() if v["type"] == "turbulence_stats"}
    if not turbulence_tables:
        return

    st.header("Turbulence Statistics")
    if len(data_dirs) > 1:
        for key, table_info in turbulence_tables.items():
            df_stats = table_info["df"]
            dir_name = table_info["dir_name"]
            st.subheader(f"📁 {dir_name}")
            st.markdown("**Latest Values:**")
            latest = df_stats.iloc[-1]
            latest_df = latest.to_frame().T
            st.dataframe(latest_df, width="stretch")
            capture_button(
                df=latest_df,
                title=f"Latest Values - {dir_name}",
                source_page="Other Turbulence Stats",
            )
            st.markdown("**Time Series Data:**")
            st.dataframe(df_stats, width="stretch", height=300)
            capture_button(
                df=df_stats,
                title=f"Time Series - {dir_name}",
                source_page="Other Turbulence Stats",
            )
            st.markdown("---")
    else:
        key = list(turbulence_tables.keys())[0]
        df_stats = turbulence_tables[key]["df"]
        st.subheader("Latest Values")
        latest = df_stats.iloc[-1]
        latest_df = latest.to_frame().T
        st.dataframe(latest_df, width="stretch")
        capture_button(df=latest_df, title="Latest Values", source_page="Other Turbulence Stats")
        st.subheader("Time Series Data")
        st.dataframe(df_stats, width="stretch", height=400)
        capture_button(df=df_stats, title="Time Series Data", source_page="Other Turbulence Stats")
        st.markdown("---")

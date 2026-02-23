"""
3D Volume Viewer — Main view: load data, compute field, build 3D plot, theory, export.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from plotly.colors import hex_to_rgb

from data_readers.vti_reader import compute_velocity_magnitude, compute_vorticity
from utils.iso_surfaces import compute_qs_s, compute_q_invariant, compute_r_invariant
from pages.PDFs.pdf_params import get_grid_spacing_options
from utils.export_figs import export_panel
from utils.plot_style import apply_figure_size, convert_superscript
from utils.report_builder import capture_button

from .data_helpers import (
    load_velocity_file,
    safe_minmax,
    downsample3d,
    make_grid,
    apply_clip,
    colormap_options,
    create_slice_surface,
)
from .plot_style import render_plot_style_sidebar, get_plot_style_3d


def _compute_field(velocity, field_type: str, dx: float = 1.0, dy: float = 1.0, dz: float = 1.0):
    """Compute scalar field from velocity based on field_type. dx,dy,dz used for gradient-based fields."""
    if field_type == "Velocity Magnitude":
        return compute_velocity_magnitude(velocity)
    elif field_type == "ux":
        return velocity[:, :, :, 0]
    elif field_type == "uy":
        return velocity[:, :, :, 1]
    elif field_type == "uz":
        return velocity[:, :, :, 2]
    elif field_type == "Vorticity Magnitude":
        vort = compute_vorticity(velocity, dx=dx, dy=dy, dz=dz)
        return np.sqrt(
            vort[:, :, :, 0] ** 2 + vort[:, :, :, 1] ** 2 + vort[:, :, :, 2] ** 2
        )
    elif field_type.startswith("ω"):
        vort = compute_vorticity(velocity, dx=dx, dy=dy, dz=dz)
        if field_type == "ωx":
            return vort[:, :, :, 0]
        elif field_type == "ωy":
            return vort[:, :, :, 1]
        else:
            return vort[:, :, :, 2]
    elif field_type == "Q_S^S":
        return compute_qs_s(velocity, dx=dx, dy=dy, dz=dz)
    elif field_type == "Q Invariant":
        return compute_q_invariant(velocity, dx=dx, dy=dy, dz=dz)
    elif field_type == "R Invariant":
        return compute_r_invariant(velocity, dx=dx, dy=dy, dz=dz)
    else:
        return velocity[:, :, :, 0]


def render_main_view(state: dict) -> None:
    """
    Render the main 3D volume view: load velocity, compute field, build plot.
    state: dict from load_volume_data() with data_dir, selected_file, file_index, etc.
    """
    data_dir = state["data_dir"]
    all_files = state["all_files"]
    file_index = state["file_index"]
    selected_file = state["selected_file"]
    iterations = state["iterations"]
    iteration = iterations[file_index]
    filename = Path(selected_file).name

    file_ext = Path(selected_file).suffix.lower()
    file_type = "HDF5" if file_ext in [".h5", ".hdf5"] else "VTI"
    abs_selected_file = str(Path(selected_file).resolve())

    try:
        with st.spinner(
            f"Loading {file_type} file {file_index + 1}/{len(all_files)} ({filename})..."
        ):
            vti_data = load_velocity_file(abs_selected_file)

        velocity = vti_data["velocity"]
        if velocity is None or len(velocity.shape) != 4:
            raise ValueError(
                f"Invalid velocity data shape: {velocity.shape if velocity is not None else 'None'}"
            )

        nx, ny, nz = velocity.shape[:3]
        if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
            st.warning(
                f"File {filename} contains NaN or Inf values. Visualization may be incorrect."
            )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"Loaded: {filename}")
        with col2:
            st.info(f"Grid: {nx} × {ny} × {nz}")
        with col3:
            st.info(f"Points: {nx * ny * nz:,}")

        st.sidebar.markdown("---")
        st.sidebar.header("Visualization")
        field_type = st.sidebar.selectbox(
            "Field to visualize:",
            [
                "ux", "uy", "uz", "Velocity Magnitude",
                "Vorticity Magnitude", "ωx", "ωy", "ωz",
                "Q_S^S", "Q Invariant", "R Invariant",
            ],
            index=0,
            key="field_type",
        )

        # Grid spacing for gradient-based fields (LBM vs NS)
        data_dir_path = Path(data_dir).resolve() if data_dir else None
        if data_dir_path and data_dir_path.exists():
            spacing_options = get_grid_spacing_options(data_dir_path)
            with st.sidebar.expander("🔧 Advanced (grid spacing)", expanded=False):
                choice_labels = list(spacing_options.keys())
                default_idx = 0  # LBM first, NS second
                spacing_choice = st.radio(
                    "Grid spacing source",
                    choice_labels,
                    index=min(default_idx, len(choice_labels) - 1),
                    help="LBM: dx=1 (lattice units). NS: dx=L/nx from simulation.json.",
                    key="vol3d_spacing_choice",
                )
                dx_selected, dy_selected, dz_selected = spacing_options[spacing_choice]
                st.caption("Used for vorticity, Q_S^S, Q, R (gradient-based fields).")
                manual_dx = st.number_input(
                    "Or override dx (=dy=dz)",
                    value=dx_selected,
                    min_value=1e-6,
                    step=0.001,
                    format="%.6f",
                    help="Optional: custom dx to override selection above.",
                    key="vol3d_dx_override",
                )
                use_override = not any(
                    abs(manual_dx - v[0]) < 1e-9 for v in spacing_options.values()
                )
                dx, dy, dz = (
                    (manual_dx, manual_dx, manual_dx)
                    if use_override
                    else (dx_selected, dy_selected, dz_selected)
                )
                st.caption(f"Using dx = {dx:.6f}")
        else:
            dx, dy, dz = 1.0, 1.0, 1.0

        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Performance")
        downsample_step = st.sidebar.slider(
            "Downsample step",
            1, 8, 2,
            help="Uses field[::step, ::step, ::step]. Increase for large grids.",
            key="downsample",
        )

        if field_type in ("Q_S^S", "Q Invariant", "R Invariant"):
            with st.spinner(f"Computing {field_type}..."):
                field = _compute_field(velocity, field_type, dx=dx, dy=dy, dz=dz)
        else:
            field = _compute_field(velocity, field_type, dx=dx, dy=dy, dz=dz)

        field_ds = downsample3d(field, downsample_step)
        nx_d, ny_d, nz_d = field_ds.shape
        xg, yg, zg = make_grid(nx_d, ny_d, nz_d)
        vmin, vmax = safe_minmax(field_ds)

        st.sidebar.markdown("---")
        st.sidebar.subheader("👁️ Display Modes")
        show_volume = st.sidebar.checkbox("Volume rendering", value=False, key="show_vol")
        show_slices = st.sidebar.checkbox("Orthogonal slices", value=True, key="show_slices")
        show_surface = st.sidebar.checkbox("Surface", value=False, key="show_surface")
        show_iso = st.sidebar.checkbox("Isosurface", value=False, key="show_iso")

        cmap_opts = colormap_options()
        rdbu_index = cmap_opts.index("rdbu") if "rdbu" in cmap_opts else 0
        cmap = st.sidebar.selectbox("Colormap", cmap_opts, index=rdbu_index, key="colormap")

        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Rendering Controls")
        # Keep actual data range for colormap; expand only for sliders when range is tiny
        plot_vmin, plot_vmax = float(vmin), float(vmax)
        slider_vmin, slider_vmax = plot_vmin, plot_vmax
        data_span = vmax - vmin
        min_span = 0.1
        if data_span < min_span or not np.isfinite(data_span):
            center = (vmin + vmax) / 2 if np.isfinite(vmin + vmax) else 0.0
            slider_vmin = center - min_span / 2
            slider_vmax = center + min_span / 2
        cmax = st.sidebar.slider(
            "Color Max (Contrast)",
            float(slider_vmin), float(slider_vmax),
            float(np.clip(plot_vmax * 0.6, slider_vmin, slider_vmax)),
            help="Lower values reveal turbulent structures by clipping low-energy regions",
            key="color_max",
        )
        step = max((slider_vmax - slider_vmin) / 200, 1e-6)
        vrange = st.sidebar.slider(
            "Value range",
            min_value=float(slider_vmin),
            max_value=float(slider_vmax),
            value=(float(slider_vmin), float(cmax)),
            step=step,
            key="vrange",
        )

        vol_opacity = 0.15
        vol_surface_count = 20
        if show_volume:
            vol_opacity = st.sidebar.slider(
                "Volume opacity", 0.01, 0.8, 0.15, 0.01,
                help="Higher = denser fog-like volume. Lower values (0.1-0.2) work better for turbulence.",
                key="vol_opacity",
            )
            vol_surface_count = st.sidebar.slider(
                "Volume surfaces", 5, 40, 20, 1,
                help="More surfaces = richer volume but heavier.",
                key="vol_surfaces",
            )

        iso_value = 0.0
        iso_opacity = 0.4
        if show_iso:
            if field_type == "Q_S^S":
                st.sidebar.markdown("**Q_S^S Threshold (log10 scale; actual value = 10^slider)**")
                qss_max = float(np.nanmax(field_ds))
                qss_max = max(qss_max, 1e-30)
                log_iso = st.sidebar.slider(
                    "log10(Q_S^S threshold)",
                    min_value=-12.0,
                    max_value=float(np.log10(qss_max)),
                    value=float(np.log10(0.5 * qss_max)),
                    step=0.05,
                    key="log_qss_threshold",
                )
                iso_value = 10.0 ** log_iso
            elif field_type in ("Q Invariant", "R Invariant"):
                # Q and R span many orders of magnitude; use log-scale threshold like Q_S^S
                f_valid = field_ds[np.isfinite(field_ds)]
                f_abs = np.abs(f_valid)
                f_abs = f_abs[f_abs > 1e-30]
                if f_abs.size > 0:
                    abs_max = float(np.nanmax(f_abs))
                    abs_max = max(abs_max, 1e-30)
                    log_min = -12.0
                    log_max = float(np.log10(abs_max))
                    default_log = np.log10(0.5 * abs_max)
                    default_log = np.clip(default_log, log_min, log_max)
                    st.sidebar.markdown(f"**{field_type} Threshold (log10 scale; actual = 10^slider)**")
                    log_iso = st.sidebar.slider(
                        f"log10(|{field_type}| threshold)",
                        min_value=log_min,
                        max_value=log_max,
                        value=float(default_log),
                        step=0.05,
                        key=f"log_{field_type.replace(' ', '_')}_threshold",
                    )
                    iso_value = 10.0 ** log_iso
                else:
                    iso_value = 0.0
            else:
                iso_min, iso_max = float(vrange[0]), float(vrange[1])
                if iso_max <= iso_min:
                    iso_max = iso_min + 1.0 if iso_min >= 0 else iso_min - 1.0
                iso_value = st.sidebar.slider(
                    "Isosurface value",
                    min_value=iso_min,
                    max_value=iso_max,
                    value=float((iso_min + iso_max) / 2),
                    step=(iso_max - iso_min) / 200 if iso_max > iso_min else 1.0,
                    key="iso_value",
                )
            iso_opacity = st.sidebar.slider(
                "Isosurface opacity", 0.05, 1.0, 0.4, 0.05, key="iso_opacity"
            )

        surface_opacity = 0.8
        if show_surface:
            surface_opacity = st.sidebar.slider(
                "Surface opacity", 0.05, 1.0, 0.8, 0.05, key="surface_opacity"
            )

        slice_x, slice_y, slice_z = nx_d // 2, ny_d // 2, nz_d // 2
        slice_opacity = 0.9
        if show_slices:
            st.sidebar.markdown("---")
            st.sidebar.subheader("✂️ Slice Planes")
            slice_x = st.sidebar.slider("X slice", 0, nx_d - 1, nx_d // 2, key="slice_x")
            slice_y = st.sidebar.slider("Y slice", 0, ny_d - 1, ny_d // 2, key="slice_y")
            slice_z = st.sidebar.slider("Z slice", 0, nz_d - 1, nz_d // 2, key="slice_z")
            slice_opacity = st.sidebar.slider(
                "Slice opacity", 0.05, 1.0, 0.9, 0.05, key="slice_opacity"
            )

        st.sidebar.markdown("---")
        st.sidebar.subheader("✂️ Clipping Box")
        use_clip = st.sidebar.checkbox("Enable clipping", value=False, key="use_clip")
        if use_clip:
            cxmin, cxmax = st.sidebar.slider("Clip X", 0, nx_d - 1, (0, nx_d - 1), key="clip_x")
            cymin, cymax = st.sidebar.slider("Clip Y", 0, ny_d - 1, (0, ny_d - 1), key="clip_y")
            czmin, czmax = st.sidebar.slider("Clip Z", 0, nz_d - 1, (0, nz_d - 1), key="clip_z")
        else:
            cxmin, cxmax = 0, nx_d - 1
            cymin, cymax = 0, ny_d - 1
            czmin, czmax = 0, nz_d - 1

        field_clip = (
            apply_clip(field_ds, cxmin, cxmax, cymin, cymax, czmin, czmax)
            if use_clip
            else field_ds
        )

        ps = render_plot_style_sidebar()

        st.sidebar.markdown("---")
        st.sidebar.subheader("📐 Coordinate Axes")
        show_axes = st.sidebar.checkbox("Show coordinate axes", value=False, key="show_axes_3d")
        show_axis_labels = st.sidebar.checkbox("Show axes labels", value=False, key="show_axis_labels_3d")

        st.sidebar.markdown("---")
        st.sidebar.subheader("📷 Camera")
        camera_preset = st.sidebar.selectbox(
            "View preset",
            ["Isometric", "XY", "XZ", "YZ", "Custom"],
            key="camera_preset",
        )

        # Clamp cmax to data range so colormap shows structure for small-valued fields (Q, R)
        plot_cmax = float(np.clip(cmax, plot_vmin, plot_vmax))
        fig = _build_3d_plot(
            xg=xg, yg=yg, zg=zg,
            field_clip=field_clip,
            vmin=plot_vmin, vmax=plot_vmax, cmax=plot_cmax, cmap=cmap,
            field_type=field_type,
            show_volume=show_volume,
            vol_opacity=vol_opacity,
            vol_surface_count=vol_surface_count,
            show_iso=show_iso,
            iso_value=iso_value,
            iso_opacity=iso_opacity,
            show_surface=show_surface,
            surface_opacity=surface_opacity,
            nx_d=nx_d, ny_d=ny_d, nz_d=nz_d,
            show_slices=show_slices,
            slice_x=slice_x, slice_y=slice_y, slice_z=slice_z,
            slice_opacity=slice_opacity,
            show_axes=show_axes,
            show_axis_labels=show_axis_labels,
            create_slice_surface=create_slice_surface,
        )

        camera_dicts = {
            "Isometric": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
            "XY": dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
            "XZ": dict(eye=dict(x=0, y=2.5, z=0), up=dict(x=0, y=0, z=1)),
            "YZ": dict(eye=dict(x=2.5, y=0, z=0), up=dict(x=0, y=1, z=0)),
            "Custom": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
        }

        ps = get_plot_style_3d()
        layout_kwargs = {}
        layout_kwargs = apply_figure_size(layout_kwargs, ps)
        default_height = layout_kwargs.get("height", 600)
        scene_bgcolor = ps.get("plot_bgcolor", "#FFFFFF")
        paper_bgcolor = ps.get("paper_bgcolor", "#FFFFFF")
        grid_color = ps.get("grid_color", "#B0B0B0")
        axis_title_size = ps.get("axis_title_size", 14)
        font_color = ps.get("font_color", "#000000")
        title_size = ps.get("title_size", 16)

        layout_kwargs_title = {}
        if ps.get("show_plot_title", False) and ps.get("plot_title"):
            layout_kwargs_title["title"] = dict(
                text=convert_superscript(ps.get("plot_title")),
                font=dict(
                    family=ps.get("font_family", "Arial"),
                    size=title_size,
                    color=font_color,
                ),
            )

        fig.update_layout(
            height=default_height,
            **layout_kwargs_title,
            scene=dict(
                xaxis_title="X" if show_axis_labels else "",
                yaxis_title="Y" if show_axis_labels else "",
                zaxis_title="Z" if show_axis_labels else "",
                aspectmode="data",
                camera=camera_dicts.get(camera_preset, camera_dicts["Isometric"]),
                bgcolor=scene_bgcolor,
                xaxis=dict(
                    backgroundcolor=scene_bgcolor,
                    gridcolor=grid_color,
                    showbackground=True,
                    showticklabels=show_axis_labels,
                    title_font=dict(size=axis_title_size, color=font_color),
                    tickfont=dict(color=font_color),
                ),
                yaxis=dict(
                    backgroundcolor=scene_bgcolor,
                    gridcolor=grid_color,
                    showbackground=True,
                    showticklabels=show_axis_labels,
                    title_font=dict(size=axis_title_size, color=font_color),
                    tickfont=dict(color=font_color),
                ),
                zaxis=dict(
                    backgroundcolor=scene_bgcolor,
                    gridcolor=grid_color,
                    showbackground=True,
                    showticklabels=show_axis_labels,
                    title_font=dict(size=axis_title_size, color=font_color),
                    tickfont=dict(color=font_color),
                ),
            ),
            legend=dict(
                itemsizing="constant",
                x=1.02,
                y=1,
                bgcolor=f"rgba{tuple(list(hex_to_rgb(paper_bgcolor)) + [0.8])}",
                bordercolor=grid_color,
                borderwidth=1,
                font=dict(color=font_color),
            ),
            margin=dict(
                l=0,
                r=0,
                t=50 if (ps.get("show_plot_title", False) and ps.get("plot_title")) else 0,
                b=0,
            ),
            paper_bgcolor=paper_bgcolor,
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displayModeBar": True,
                "modeBarButtonsToAdd": [
                    "drawline", "drawopenpath", "drawclosedpath",
                    "drawcircle", "drawrect", "eraseshape",
                ],
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"{Path(selected_file).stem}_3d_view",
                    "height": 600,
                    "width": 1200,
                    "scale": 2,
                },
            },
        )

        capture_title = f"3D Volume Viewer - {field_type}"
        if iteration is not None:
            capture_title += f" (Iteration {iteration})"
        else:
            capture_title += f" (Time Step {file_index})"
        capture_button(fig, title=capture_title, source_page="3D Volume Viewer")

        render_theory_section()
        export_panel(fig, data_dir, base_name=f"{Path(selected_file).stem}_3d_view")

    except Exception as e:
        st.error(f"Error loading {file_type} file: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def _build_3d_plot(
    xg, yg, zg,
    field_clip,
    vmin, vmax, cmax, cmap,
    field_type,
    show_volume,
    vol_opacity,
    vol_surface_count,
    show_iso,
    iso_value,
    iso_opacity,
    show_surface,
    surface_opacity,
    nx_d, ny_d, nz_d,
    show_slices,
    slice_x, slice_y, slice_z,
    slice_opacity,
    show_axes,
    show_axis_labels,
    create_slice_surface,
):
    """Build the 3D figure with volume, isosurface, surface, slices, axes."""
    fig = go.Figure()

    if show_volume:
        isomin_val = vmin + (cmax - vmin) * 0.1 if cmax > vmin else vmin
        fig.add_trace(
            go.Volume(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=field_clip.flatten(),
                isomin=isomin_val,
                isomax=cmax,
                opacity=vol_opacity,
                surface_count=vol_surface_count,
                colorscale=cmap,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Volume",
                showscale=True,
                colorbar=dict(
                    title=dict(text=field_type, font=dict(size=14)),
                    len=0.75,
                    y=0.5,
                    thickness=20,
                ),
            )
        )

    if show_iso:
        fig.add_trace(
            go.Isosurface(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=field_clip.flatten(),
                isomin=iso_value,
                isomax=iso_value,
                surface_count=1,
                opacity=iso_opacity,
                colorscale=cmap,
                showscale=False,
                name=f"Isosurface @ {iso_value:.3f}",
            )
        )

    if show_surface:
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, ny_d))
        y_coords = np.ones((nx_d, 1)) * np.arange(ny_d)[None, :]
        z_front = np.zeros((nx_d, ny_d))
        fig.add_trace(
            create_slice_surface(
                x_coords, y_coords, z_front,
                field_clip[:, :, 0], vmin, cmax, cmap, surface_opacity
            )
        )
        z_back = np.full((nx_d, ny_d), nz_d - 1)
        fig.add_trace(
            create_slice_surface(
                x_coords, y_coords, z_back,
                field_clip[:, :, nz_d - 1], vmin, cmax, cmap, surface_opacity
            )
        )
        y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
        x_left = np.zeros((ny_d, nz_d))
        fig.add_trace(
            create_slice_surface(
                x_left, y_coords, z_coords,
                field_clip[0, :, :], vmin, cmax, cmap, surface_opacity
            )
        )
        x_right = np.full((ny_d, nz_d), nx_d - 1)
        fig.add_trace(
            create_slice_surface(
                x_right, y_coords, z_coords,
                field_clip[nx_d - 1, :, :], vmin, cmax, cmap, surface_opacity
            )
        )
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
        y_bottom = np.zeros((nx_d, nz_d))
        fig.add_trace(
            create_slice_surface(
                x_coords, y_bottom, z_coords,
                field_clip[:, 0, :], vmin, cmax, cmap, surface_opacity
            )
        )
        y_top = np.full((nx_d, nz_d), ny_d - 1)
        fig.add_trace(
            create_slice_surface(
                x_coords, y_top, z_coords,
                field_clip[:, ny_d - 1, :], vmin, cmax, cmap, surface_opacity
            )
        )

    if show_slices:
        z_plane = np.full((nx_d, ny_d), slice_z)
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, ny_d))
        y_coords = np.ones((nx_d, 1)) * np.arange(ny_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_coords, y_coords, z_plane,
                field_clip[:, :, slice_z], vmin, cmax, cmap, slice_opacity
            )
        )
        y_plane = np.full((nx_d, nz_d), slice_y)
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_coords, y_plane, z_coords,
                field_clip[:, slice_y, :], vmin, cmax, cmap, slice_opacity
            )
        )
        x_plane = np.full((ny_d, nz_d), slice_x)
        y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_plane, y_coords, z_coords,
                field_clip[slice_x, :, :], vmin, cmax, cmap, slice_opacity
            )
        )

    if show_axes:
        axis_length = max(nx_d, ny_d, nz_d) * 0.15
        fig.add_trace(
            go.Scatter3d(
                x=[0, axis_length], y=[0, 0], z=[0, 0],
                mode="lines+markers",
                line=dict(color="red", width=3),
                marker=dict(size=5, color="red"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, axis_length], z=[0, 0],
                mode="lines+markers",
                line=dict(color="green", width=3),
                marker=dict(size=5, color="green"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[0, axis_length],
                mode="lines+markers",
                line=dict(color="blue", width=3),
                marker=dict(size=5, color="blue"),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        if show_axis_labels:
            fig.add_trace(
                go.Scatter3d(
                    x=[axis_length * 1.1], y=[0], z=[0],
                    mode="text",
                    text=["x"],
                    textfont=dict(size=14, color="red"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[axis_length * 1.1], z=[0],
                    mode="text",
                    text=["y"],
                    textfont=dict(size=14, color="green"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[0], z=[axis_length * 1.1],
                    mode="text",
                    text=["z"],
                    textfont=dict(size=14, color="blue"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    return fig


def render_theory_section() -> None:
    """Render the Theory & Equations expander. Uses shared content/volume_viewer_theory_content."""
    from content.volume_viewer_theory_content import get_volume_viewer_theory_markdown

    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown(get_volume_viewer_theory_markdown())

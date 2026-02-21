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


def _compute_field(velocity, field_type: str):
    """Compute scalar field from velocity based on field_type."""
    if field_type == "Velocity Magnitude":
        return compute_velocity_magnitude(velocity)
    elif field_type == "ux":
        return velocity[:, :, :, 0]
    elif field_type == "uy":
        return velocity[:, :, :, 1]
    elif field_type == "uz":
        return velocity[:, :, :, 2]
    elif field_type == "Vorticity Magnitude":
        vort = compute_vorticity(velocity)
        return np.sqrt(
            vort[:, :, :, 0] ** 2 + vort[:, :, :, 1] ** 2 + vort[:, :, :, 2] ** 2
        )
    elif field_type.startswith("ω"):
        vort = compute_vorticity(velocity)
        if field_type == "ωx":
            return vort[:, :, :, 0]
        elif field_type == "ωy":
            return vort[:, :, :, 1]
        else:
            return vort[:, :, :, 2]
    elif field_type == "Q_S^S":
        return compute_qs_s(velocity)
    elif field_type == "Q Invariant":
        return compute_q_invariant(velocity)
    elif field_type == "R Invariant":
        return compute_r_invariant(velocity)
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
                field = _compute_field(velocity, field_type)
        else:
            field = _compute_field(velocity, field_type)

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
        if vmax <= vmin:
            vmax = vmin + 1.0 if vmin >= 0 else vmin - 1.0
        cmax = st.sidebar.slider(
            "Color Max (Contrast)",
            float(vmin), float(vmax),
            float(vmax) * 0.6,
            help="Lower values reveal turbulent structures by clipping low-energy regions",
            key="color_max",
        )
        vrange = st.sidebar.slider(
            "Value range",
            min_value=float(vmin),
            max_value=float(vmax),
            value=(float(vmin), float(cmax)),
            step=(vmax - vmin) / 200 if vmax > vmin else 1.0,
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

        fig = _build_3d_plot(
            xg=xg, yg=yg, zg=zg,
            field_clip=field_clip,
            vmin=vmin, vmax=vmax, cmax=cmax, cmap=cmap,
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
    """Render the Theory & Equations expander."""
    with st.expander("📚 Theory & Equations", expanded=False):
        st.markdown("### Velocity Fields")
        st.markdown("**Velocity magnitude:**")
        st.latex(r"|\mathbf{u}| = \sqrt{u_x^2 + u_y^2 + u_z^2}")

        st.markdown("### Vorticity")
        st.markdown("**Vorticity vector:**")
        st.latex(r"\boldsymbol{\omega} = \nabla \times \mathbf{u}")
        st.markdown("**Components:**")
        st.latex(
            r"\omega_x = \frac{\partial u_z}{\partial y} - \frac{\partial u_y}{\partial z}, "
            r"\quad \omega_y = \frac{\partial u_x}{\partial z} - \frac{\partial u_z}{\partial x}, "
            r"\quad \omega_z = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}"
        )
        st.markdown("**Vorticity magnitude:**")
        st.latex(r"|\boldsymbol{\omega}| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}")

        st.markdown("### Q_S^S Method for Vortex Visualization")
        st.markdown("**Main equation:**")
        st.latex(r"Q_S^S = \left[(Q_W^3 + Q_S^3) + (\Sigma^2 - R_s^2)\right]^{1/3}")
        st.markdown("**Component equations:**")
        st.markdown("**Rotation Rate Strength:**")
        st.latex(r"Q_W = \frac{1}{2}\Omega_{ij}\Omega_{ij}")
        st.markdown("**Deformation Rate Strength:**")
        st.latex(r"Q_S = -\frac{1}{2}S_{ij}S_{ij}")
        st.markdown("**Enstrophy Production Term:**")
        st.latex(r"\Sigma = \omega_i S_{ij} \omega_j")
        st.markdown("**Strain Rate Production:**")
        st.latex(r"R_s = -\frac{1}{3}S_{ij}S_{jk}S_{ki}")
        st.markdown("**Tensor definitions:**")
        st.markdown("- $\\Omega_{ij}$: Rotation tensor (antisymmetric part of velocity gradient)")
        st.markdown("- $S_{ij}$: Deformation tensor (symmetric part of velocity gradient)")
        st.markdown("- $\\omega_i$: Vorticity vector")
        st.markdown("**Isosurface Thresholds (Paper values):**")
        st.markdown("- $32^3$ resolution: Threshold = 2.5")
        st.markdown("- $64^3$ resolution: Threshold = 3.5")
        st.markdown("- $128^3$ resolution: Threshold = 5.0")
        st.markdown("- $256^3$ resolution: Threshold = 6.5")

        st.markdown("### Velocity Gradient Tensor Invariants")
        st.markdown("**Second Invariant Q:**")
        st.latex(r"Q = -\frac{1}{2}A_{ij}A_{ij} = \frac{1}{4}(\omega_i\omega_i - 2S_{ij}S_{ij})")
        st.markdown("**Third Invariant R:**")
        st.latex(
            r"R = -\frac{1}{3}A_{ij}A_{jk}A_{ki} = "
            r"-\frac{1}{3}\left(S_{ij}S_{jk}S_{ki} + \frac{3}{4}\omega_i\omega_j S_{ij}\right)"
        )
        st.markdown("where $A_{ij} = \\partial u_i/\\partial x_j$ is the velocity gradient tensor.")

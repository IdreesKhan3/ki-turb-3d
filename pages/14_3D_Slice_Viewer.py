"""
3D Slice Viewer Page (Streamlit)
Interactive 3D volume + slicing for VTI files

Features:
- Reads *.vti velocity fields
- Field choices: |u|, ux, uy, uz, |ω|
- Interactive Plotly 3D:
    * Volume rendering (opacity, colormap, value range)
    * Orthogonal slicing planes (x/y/z) as draggable sliders
    * Clipping (cutting) box: x/y/z min-max
    * Optional isosurface overlay
    * User rotates / zooms / pans in browser
- Fast downsampling controls for large grids
"""

import streamlit as st
import numpy as np
import glob
from pathlib import Path
import sys
import plotly.graph_objects as go
import plotly.colors as pc

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.vti_reader import read_vti_file, compute_velocity_magnitude, compute_vorticity
from utils.file_detector import natural_sort_key


# -----------------------------
# Helpers
# -----------------------------
def _safe_minmax(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    return float(a.min()), float(a.max())

def _downsample3d(field, step):
    if step <= 1:
        return field
    return field[::step, ::step, ::step]

def _make_grid(nx, ny, nz):
    # index grid (voxel coordinates)
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    return x, y, z

def _apply_clip(field, xmin, xmax, ymin, ymax, zmin, zmax):
    # clip by setting outside to nan (plotly ignores nan)
    clipped = field.copy()
    mask = np.ones_like(clipped, dtype=bool)
    mask &= (np.arange(clipped.shape[0])[:, None, None] >= xmin)
    mask &= (np.arange(clipped.shape[0])[:, None, None] <= xmax)
    mask &= (np.arange(clipped.shape[1])[None, :, None] >= ymin)
    mask &= (np.arange(clipped.shape[1])[None, :, None] <= ymax)
    mask &= (np.arange(clipped.shape[2])[None, None, :] >= zmin)
    mask &= (np.arange(clipped.shape[2])[None, None, :] <= zmax)
    clipped[~mask] = np.nan
    return clipped

def _colormap_options():
    # good scientific maps (Plotly built-ins)
    return [
        "Viridis", "Cividis", "Plasma", "Magma", "Inferno",
        "Turbo", "Rainbow", "Jet", "Portland", "RdBu",
        "Spectral", "Ice", "Electric"
    ]


# -----------------------------
# Main
# -----------------------------
def main():
    st.title("🔬 3D Slice Viewer (Interactive Volume + Slices)")

    if 'data_directory' not in st.session_state or not st.session_state.data_directory:
        st.warning("Please select a data directory from the main page.")
        return
    data_dir = Path(st.session_state.data_directory)

    vti_files = sorted(glob.glob(str(data_dir / "*.vti")), key=natural_sort_key)
    if not vti_files:
        st.error("No VTI files found in the selected directory.")
        st.info("Expected names like: `velocity_50000.vti` or `*_*.vti`")
        return

    # Sidebar UI
    st.sidebar.header("Volume / Slice Options")

    selected_file = st.sidebar.selectbox(
        "Select VTI file:", vti_files,
        format_func=lambda x: Path(x).name
    )

    field_type = st.sidebar.selectbox(
        "Field to visualize:",
        ["Velocity Magnitude", "ux", "uy", "uz", "Vorticity Magnitude"]
    )

    downsample_step = st.sidebar.slider(
        "Downsample step (for speed)",
        1, 8, 2,
        help="Uses field[::step, ::step, ::step]. Increase for big grids."
    )

    show_volume = st.sidebar.checkbox("Show volume rendering", value=True)
    show_slices = st.sidebar.checkbox("Show orthogonal slices", value=True)
    show_iso = st.sidebar.checkbox("Show isosurface", value=False)

    cmap = st.sidebar.selectbox("Colormap", _colormap_options(), index=0)

    # Load VTI
    try:
        vti_data = read_vti_file(selected_file)
        nx, ny, nz = vti_data['dimensions']
        velocity = vti_data['velocity']
        st.success(f"Loaded: {Path(selected_file).name}")
        st.info(f"Grid: {nx} × {ny} × {nz}")

        # Compute field
        if field_type == "Velocity Magnitude":
            field = compute_velocity_magnitude(velocity)
        elif field_type == "ux":
            field = velocity[:, :, :, 0]
        elif field_type == "uy":
            field = velocity[:, :, :, 1]
        elif field_type == "uz":
            field = velocity[:, :, :, 2]
        else:
            vort = compute_vorticity(velocity)
            field = np.sqrt(vort[:, :, :, 0]**2 + vort[:, :, :, 1]**2 + vort[:, :, :, 2]**2)

        # Downsample
        field_ds = _downsample3d(field, downsample_step)
        nx_d, ny_d, nz_d = field_ds.shape
        xg, yg, zg = _make_grid(nx_d, ny_d, nz_d)

        vmin, vmax = _safe_minmax(field_ds)

        # Value range + opacity controls
        st.sidebar.subheader("Rendering Controls")
        vrange = st.sidebar.slider(
            "Value range to display",
            min_value=float(vmin), max_value=float(vmax),
            value=(float(vmin), float(vmax)),
            step=(vmax - vmin) / 200 if vmax > vmin else 1.0
        )

        vol_opacity = st.sidebar.slider(
            "Volume opacity", 0.01, 0.8, 0.12, 0.01,
            help="Higher = denser fog-like volume."
        )
        vol_surface_count = st.sidebar.slider(
            "Volume surface count", 5, 40, 18, 1,
            help="More surfaces = richer volume but heavier."
        )

        iso_value = st.sidebar.slider(
            "Isosurface value",
            min_value=float(vrange[0]), max_value=float(vrange[1]),
            value=float((vrange[0] + vrange[1]) / 2),
            step=(vrange[1] - vrange[0]) / 200 if vrange[1] > vrange[0] else 1.0,
            disabled=not show_iso
        )
        iso_opacity = st.sidebar.slider(
            "Isosurface opacity", 0.05, 1.0, 0.4, 0.05,
            disabled=not show_iso
        )

        # Slice controls
        st.sidebar.subheader("Slice Planes")
        slice_x = st.sidebar.slider("X slice index", 0, nx_d-1, nx_d//2, disabled=not show_slices)
        slice_y = st.sidebar.slider("Y slice index", 0, ny_d-1, ny_d//2, disabled=not show_slices)
        slice_z = st.sidebar.slider("Z slice index", 0, nz_d-1, nz_d//2, disabled=not show_slices)
        slice_opacity = st.sidebar.slider("Slice opacity", 0.05, 1.0, 0.9, 0.05, disabled=not show_slices)

        # Clipping / cutting box
        st.sidebar.subheader("Cut / Clip Box")
        cxmin, cxmax = st.sidebar.slider("Clip X", 0, nx_d-1, (0, nx_d-1))
        cymin, cymax = st.sidebar.slider("Clip Y", 0, ny_d-1, (0, ny_d-1))
        czmin, czmax = st.sidebar.slider("Clip Z", 0, nz_d-1, (0, nz_d-1))

        field_clip = _apply_clip(field_ds, cxmin, cxmax, cymin, cymax, czmin, czmax)

        # Build Plotly 3D
        fig = go.Figure()

        # Volume
        if show_volume:
            fig.add_trace(go.Volume(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=field_clip.flatten(),
                isomin=vrange[0],
                isomax=vrange[1],
                opacity=vol_opacity,
                surface_count=vol_surface_count,
                colorscale=cmap,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Volume",
                showscale=True,
                colorbar=dict(title=field_type)
            ))

        # Isosurface
        if show_iso:
            fig.add_trace(go.Isosurface(
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
                name="Isosurface"
            ))

        # Orthogonal slice planes as Surface traces
        if show_slices:
            # XY plane at z = slice_z
            z_plane = np.full((nx_d, ny_d), slice_z)
            fig.add_trace(go.Surface(
                x=np.arange(nx_d)[:, None] * np.ones((1, ny_d)),
                y=np.ones((nx_d, 1)) * np.arange(ny_d)[None, :],
                z=z_plane,
                surfacecolor=np.nan_to_num(field_clip[:, :, slice_z], nan=np.nan),
                cmin=vrange[0], cmax=vrange[1],
                colorscale=cmap,
                opacity=slice_opacity,
                showscale=False,
                name=f"XY @ z={slice_z}"
            ))

            # XZ plane at y = slice_y
            y_plane = np.full((nx_d, nz_d), slice_y)
            fig.add_trace(go.Surface(
                x=np.arange(nx_d)[:, None] * np.ones((1, nz_d)),
                y=y_plane,
                z=np.ones((nx_d, 1)) * np.arange(nz_d)[None, :],
                surfacecolor=np.nan_to_num(field_clip[:, slice_y, :], nan=np.nan),
                cmin=vrange[0], cmax=vrange[1],
                colorscale=cmap,
                opacity=slice_opacity,
                showscale=False,
                name=f"XZ @ y={slice_y}"
            ))

            # YZ plane at x = slice_x
            x_plane = np.full((ny_d, nz_d), slice_x)
            fig.add_trace(go.Surface(
                x=x_plane,
                y=np.arange(ny_d)[:, None] * np.ones((1, nz_d)),
                z=np.ones((ny_d, 1)) * np.arange(nz_d)[None, :],
                surfacecolor=np.nan_to_num(field_clip[slice_x, :, :], nan=np.nan),
                cmin=vrange[0], cmax=vrange[1],
                colorscale=cmap,
                opacity=slice_opacity,
                showscale=False,
                name=f"YZ @ x={slice_x}"
            ))

        # Layout / axes / camera
        fig.update_layout(
            height=720,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=1.4, y=1.4, z=1.2))
            ),
            legend=dict(itemsizing="constant"),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "✅ Rotate: left-drag • Pan: right-drag • Zoom: scroll • "
            "Slices/cuts/opacity via sidebar."
        )

    except Exception as e:
        st.error(f"Error loading VTI file: {e}")


if __name__ == "__main__":
    main()

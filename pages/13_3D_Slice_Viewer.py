"""
3D Slice Viewer Page (Streamlit)
Interactive 3D volume visualization with ParaView-like features

Features:
- Reads *.vti velocity fields
- Field choices: |u|, ux, uy, uz, |ω|, individual vorticity components
- Interactive Plotly 3D:
    * Volume rendering (opacity, colormap, value range)
    * Orthogonal slicing planes (x/y/z) with interactive sliders
    * Clipping (cutting) box: x/y/z min-max
    * Isosurface overlay
    * Vector field visualization (velocity arrows)
    * User rotates / zooms / pans in browser (like ParaView)
- Fast downsampling controls for large grids
- Time series animation support
- Export capabilities
"""

import streamlit as st
import numpy as np
import glob
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
import plotly.graph_objects as go
import plotly.colors as pc

from data_readers.vti_reader import read_vti_file, compute_velocity_magnitude, compute_vorticity
from utils.file_detector import natural_sort_key


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def _cached_read_vti(filepath: str):
    """Cached VTI file reading for performance"""
    return read_vti_file(filepath)

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

def _downsample_vectors(velocity, step):
    """Downsample vector field"""
    if step <= 1:
        return velocity
    return velocity[::step, ::step, ::step, :]

def _make_grid(nx, ny, nz):
    x, y, z = np.mgrid[0:nx, 0:ny, 0:nz]
    return x, y, z

def _apply_clip(field, xmin, xmax, ymin, ymax, zmin, zmax):
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
    return [
        "Viridis", "Cividis", "Plasma", "Magma", "Inferno",
        "Turbo", "Rainbow", "Jet", "Portland", "RdBu",
        "Spectral", "Ice", "Electric", "Hot", "Cool",
        "Greys", "YlOrRd", "Blues", "Reds", "Greens"
    ]

def _create_slice_surface(x_coords, y_coords, z_coords, field_slice, vmin, vmax, cmap, opacity):
    """Create a surface trace for a slice plane"""
    return go.Surface(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        surfacecolor=np.nan_to_num(field_slice, nan=np.nan),
        cmin=vmin,
        cmax=vmax,
        colorscale=cmap,
        opacity=opacity,
        showscale=False,
        hovertemplate="Value: %{surfacecolor:.4f}<extra></extra>"
    )


# -----------------------------
# Main
# -----------------------------
def main():
    inject_theme_css()
    
    st.title("🔬 3D Slice Viewer")
    st.markdown("**Interactive 3D Volume Visualization with ParaView-like Controls**")

    # Get data directories
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the main page.")
        return
    
    data_dir = Path(data_dirs[0])
    
    # Find VTI files
    vti_files = sorted(glob.glob(str(data_dir / "*.vti")), key=natural_sort_key)
    if not vti_files:
        st.error("No VTI files found in the selected directory.")
        st.info("Expected names like: `velocity_50000.vti` or `*_*.vti`")
        return

    # Sidebar - File Selection
    st.sidebar.header("📁 File Selection")
    selected_file = st.sidebar.selectbox(
        "Select VTI file:",
        vti_files,
        format_func=lambda x: Path(x).name,
        key="vti_file_selector"
    )
    
    # Time series animation (if multiple files)
    if len(vti_files) > 1:
        auto_play = st.sidebar.checkbox("Auto-play animation", value=False)
        if auto_play:
            frame_delay = st.sidebar.slider("Frame delay (ms)", 100, 2000, 500)
            if st.sidebar.button("▶️ Play"):
                st.session_state.auto_playing = True
            if st.sidebar.button("⏸️ Pause"):
                st.session_state.auto_playing = False

    # Load VTI
    try:
        with st.spinner("Loading VTI file..."):
            vti_data = _cached_read_vti(selected_file)
        
        nx, ny, nz = vti_data['dimensions']
        velocity = vti_data['velocity']
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success(f"✅ Loaded: {Path(selected_file).name}")
        with col2:
            st.info(f"📐 Grid: {nx} × {ny} × {nz}")
        with col3:
            total_points = nx * ny * nz
            st.info(f"📊 Points: {total_points:,}")

        # Sidebar - Visualization Options
        st.sidebar.markdown("---")
        st.sidebar.header("🎨 Visualization")
        
        field_type = st.sidebar.selectbox(
            "Field to visualize:",
            ["Velocity Magnitude", "ux", "uy", "uz", 
             "Vorticity Magnitude", "ωx", "ωy", "ωz"],
            key="field_type"
        )

        # Performance settings
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Performance")
        downsample_step = st.sidebar.slider(
            "Downsample step",
            1, 8, 2,
            help="Uses field[::step, ::step, ::step]. Increase for large grids.",
            key="downsample"
        )

        # Compute field
        if field_type == "Velocity Magnitude":
            field = compute_velocity_magnitude(velocity)
        elif field_type == "ux":
            field = velocity[:, :, :, 0]
        elif field_type == "uy":
            field = velocity[:, :, :, 1]
        elif field_type == "uz":
            field = velocity[:, :, :, 2]
        elif field_type == "Vorticity Magnitude":
            vort = compute_vorticity(velocity)
            field = np.sqrt(vort[:, :, :, 0]**2 + vort[:, :, :, 1]**2 + vort[:, :, :, 2]**2)
        elif field_type.startswith("ω"):
            vort = compute_vorticity(velocity)
            if field_type == "ωx":
                field = vort[:, :, :, 0]
            elif field_type == "ωy":
                field = vort[:, :, :, 1]
            else:  # ωz
                field = vort[:, :, :, 2]

        # Downsample
        field_ds = _downsample3d(field, downsample_step)
        velocity_ds = _downsample_vectors(velocity, downsample_step)
        nx_d, ny_d, nz_d = field_ds.shape
        xg, yg, zg = _make_grid(nx_d, ny_d, nz_d)

        vmin, vmax = _safe_minmax(field_ds)

        # Visualization modes
        st.sidebar.markdown("---")
        st.sidebar.subheader("👁️ Display Modes")
        show_volume = st.sidebar.checkbox("Volume rendering", value=True, key="show_vol")
        show_slices = st.sidebar.checkbox("Orthogonal slices", value=True, key="show_slices")
        show_iso = st.sidebar.checkbox("Isosurface", value=False, key="show_iso")
        show_vectors = st.sidebar.checkbox("Vector field (arrows)", value=False, key="show_vec")

        # Colormap
        cmap = st.sidebar.selectbox("Colormap", _colormap_options(), index=0, key="colormap")

        # Value range + opacity controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Rendering Controls")
        vrange = st.sidebar.slider(
            "Value range",
            min_value=float(vmin), max_value=float(vmax),
            value=(float(vmin), float(vmax)),
            step=(vmax - vmin) / 200 if vmax > vmin else 1.0,
            key="vrange"
        )

        if show_volume:
            vol_opacity = st.sidebar.slider(
                "Volume opacity", 0.01, 0.8, 0.12, 0.01,
                help="Higher = denser fog-like volume.",
                key="vol_opacity"
            )
            vol_surface_count = st.sidebar.slider(
                "Volume surfaces", 5, 40, 18, 1,
                help="More surfaces = richer volume but heavier.",
                key="vol_surfaces"
            )

        if show_iso:
            iso_value = st.sidebar.slider(
                "Isosurface value",
                min_value=float(vrange[0]), max_value=float(vrange[1]),
                value=float((vrange[0] + vrange[1]) / 2),
                step=(vrange[1] - vrange[0]) / 200 if vrange[1] > vrange[0] else 1.0,
                key="iso_value"
            )
            iso_opacity = st.sidebar.slider(
                "Isosurface opacity", 0.05, 1.0, 0.4, 0.05,
                key="iso_opacity"
            )

        # Slice controls
        if show_slices:
            st.sidebar.markdown("---")
            st.sidebar.subheader("✂️ Slice Planes")
            slice_x = st.sidebar.slider(
                "X slice", 0, nx_d-1, nx_d//2,
                key="slice_x"
            )
            slice_y = st.sidebar.slider(
                "Y slice", 0, ny_d-1, ny_d//2,
                key="slice_y"
            )
            slice_z = st.sidebar.slider(
                "Z slice", 0, nz_d-1, nz_d//2,
                key="slice_z"
            )
            slice_opacity = st.sidebar.slider(
                "Slice opacity", 0.05, 1.0, 0.9, 0.05,
                key="slice_opacity"
            )

        # Vector field controls
        if show_vectors:
            st.sidebar.markdown("---")
            st.sidebar.subheader("➡️ Vector Field")
            vector_scale = st.sidebar.slider(
                "Arrow scale", 0.1, 5.0, 1.0, 0.1,
                key="vec_scale"
            )
            vector_step = st.sidebar.slider(
                "Arrow spacing", 1, 10, 3, 1,
                help="Skip points for arrows (higher = fewer arrows)",
                key="vec_step"
            )
            vector_opacity = st.sidebar.slider(
                "Arrow opacity", 0.1, 1.0, 0.8, 0.1,
                key="vec_opacity"
            )

        # Clipping / cutting box
        st.sidebar.markdown("---")
        st.sidebar.subheader("✂️ Clipping Box")
        use_clip = st.sidebar.checkbox("Enable clipping", value=False, key="use_clip")
        if use_clip:
            cxmin, cxmax = st.sidebar.slider("Clip X", 0, nx_d-1, (0, nx_d-1), key="clip_x")
            cymin, cymax = st.sidebar.slider("Clip Y", 0, ny_d-1, (0, ny_d-1), key="clip_y")
            czmin, czmax = st.sidebar.slider("Clip Z", 0, nz_d-1, (0, nz_d-1), key="clip_z")
        else:
            cxmin, cxmax, cymin, cymax, czmin, czmax = 0, nx_d-1, 0, ny_d-1, 0, nz_d-1

        field_clip = _apply_clip(field_ds, cxmin, cxmax, cymin, cymax, czmin, czmax) if use_clip else field_ds

        # Camera controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("📷 Camera")
        camera_preset = st.sidebar.selectbox(
            "View preset",
            ["Isometric", "XY", "XZ", "YZ", "Custom"],
            key="camera_preset"
        )

        # Build Plotly 3D
        fig = go.Figure()

        # Volume rendering
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
                colorbar=dict(
                    title=dict(text=field_type, font=dict(size=14)),
                    len=0.75,
                    y=0.5,
                    thickness=20
                )
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
                name=f"Isosurface @ {iso_value:.3f}"
            ))

        # Orthogonal slice planes
        if show_slices:
            # XY plane at z = slice_z
            z_plane = np.full((nx_d, ny_d), slice_z)
            x_coords = np.arange(nx_d)[:, None] * np.ones((1, ny_d))
            y_coords = np.ones((nx_d, 1)) * np.arange(ny_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_coords, y_coords, z_plane,
                field_clip[:, :, slice_z],
                vrange[0], vrange[1], cmap, slice_opacity
            ))

            # XZ plane at y = slice_y
            y_plane = np.full((nx_d, nz_d), slice_y)
            x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
            z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_coords, y_plane, z_coords,
                field_clip[:, slice_y, :],
                vrange[0], vrange[1], cmap, slice_opacity
            ))

            # YZ plane at x = slice_x
            x_plane = np.full((ny_d, nz_d), slice_x)
            y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
            z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_plane, y_coords, z_coords,
                field_clip[slice_x, :, :],
                vrange[0], vrange[1], cmap, slice_opacity
            ))

        # Vector field (velocity arrows)
        if show_vectors:
            # Sample vectors
            step = vector_step
            x_vec = xg[::step, ::step, ::step].flatten()
            y_vec = yg[::step, ::step, ::step].flatten()
            z_vec = zg[::step, ::step, ::step].flatten()
            u_vec = velocity_ds[::step, ::step, ::step, 0].flatten()
            v_vec = velocity_ds[::step, ::step, ::step, 1].flatten()
            w_vec = velocity_ds[::step, ::step, ::step, 2].flatten()
            
            # Filter out NaN vectors
            valid = np.isfinite(u_vec) & np.isfinite(v_vec) & np.isfinite(w_vec)
            x_vec = x_vec[valid]
            y_vec = y_vec[valid]
            z_vec = z_vec[valid]
            u_vec = u_vec[valid] * vector_scale
            v_vec = v_vec[valid] * vector_scale
            w_vec = w_vec[valid] * vector_scale
            
            fig.add_trace(go.Cone(
                x=x_vec,
                y=y_vec,
                z=z_vec,
                u=u_vec,
                v=v_vec,
                w=w_vec,
                sizemode="absolute",
                sizeref=vector_scale * 2,
                anchor="tail",
                colorscale="Reds",
                showscale=False,
                opacity=vector_opacity,
                name="Velocity vectors"
            ))

        # Camera presets
        camera_dicts = {
            "Isometric": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
            "XY": dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
            "XZ": dict(eye=dict(x=0, y=2.5, z=0), up=dict(x=0, y=0, z=1)),
            "YZ": dict(eye=dict(x=2.5, y=0, z=0), up=dict(x=0, y=1, z=0)),
            "Custom": dict(eye=dict(x=1.4, y=1.4, z=1.2))
        }

        # Layout
        fig.update_layout(
            height=800,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=camera_dicts.get(camera_preset, camera_dicts["Isometric"]),
                bgcolor="white",
                xaxis=dict(backgroundcolor="white", gridcolor="lightgray", showbackground=True),
                yaxis=dict(backgroundcolor="white", gridcolor="lightgray", showbackground=True),
                zaxis=dict(backgroundcolor="white", gridcolor="lightgray", showbackground=True)
            ),
            legend=dict(
                itemsizing="constant",
                x=1.02,
                y=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white"
        )

        # Display plot
        st.plotly_chart(fig, use_container_width=True, config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{Path(selected_file).stem}_3d_view',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        })

        # Instructions
        with st.expander("ℹ️ Interactive Controls", expanded=False):
            st.markdown("""
            **Mouse Controls (like ParaView):**
            - **Rotate**: Left-click and drag
            - **Pan**: Right-click and drag (or middle-click)
            - **Zoom**: Scroll wheel (or pinch on trackpad)
            - **Reset view**: Double-click on plot
            
            **Keyboard Shortcuts:**
            - Use the camera preset dropdown for quick view changes
            - Adjust all parameters in the sidebar for real-time updates
            """)

        # Export options
        with st.expander("💾 Export Options", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download as HTML"):
                    html_str = fig.to_html(include_plotlyjs='cdn')
                    st.download_button(
                        label="Download HTML file",
                        data=html_str,
                        file_name=f"{Path(selected_file).stem}_3d.html",
                        mime="text/html"
                    )
            with col2:
                st.info("Use the camera icon (📷) in the plot toolbar to export as PNG/JPEG/SVG")

    except Exception as e:
        st.error(f"Error loading VTI file: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

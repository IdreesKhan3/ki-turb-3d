"""
3D Volume Viewer Page (Streamlit)
Interactive 3D volume visualization with ParaView-like features

Features:
- Reads *.vti velocity fields
- Field choices: |u|, ux, uy, uz, |ω|, individual vorticity components, Q_S^S, Q, R invariants
- Interactive Plotly 3D:
    * Volume rendering (opacity, colormap, value range)
    * Orthogonal slicing planes (x/y/z) with interactive sliders
    * Clipping (cutting) box: x/y/z min-max
    * Isosurface overlay
    * User rotates / zooms / pans in browser (like ParaView)
- Fast downsampling controls for large grids
- Time series animation support
- Export capabilities
"""

import streamlit as st
import numpy as np
import glob
import re
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css, apply_theme_to_plot_style
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import hex_to_rgb

from data_readers.vti_reader import read_vti_file, compute_velocity_magnitude, compute_vorticity
from data_readers.hdf5_reader import read_hdf5_file
from utils.file_detector import natural_sort_key
from utils.iso_surfaces import compute_qs_s, compute_q_invariant, compute_r_invariant
from utils.export_figs import export_panel
from utils.plot_style import default_plot_style, render_figure_size_ui, apply_figure_size, render_plot_title_ui
from utils.report_builder import capture_button
st.set_page_config(page_icon="⚫")


# -----------------------------
# Helpers
# -----------------------------
@st.cache_data(show_spinner=True)
def _cached_read_vti(filepath: str):
    """Cached VTI file reading for performance"""
    # Use absolute path for consistent caching
    abs_path = str(Path(filepath).resolve())
    return read_vti_file(abs_path)

@st.cache_data(show_spinner=True)
def _cached_read_hdf5(filepath: str, _cache_version: str = "v2"):
    """Cached HDF5 file reading for performance
    
    _cache_version: Internal parameter to invalidate cache when reader is updated
    """
    # Use absolute path for consistent caching
    abs_path = str(Path(filepath).resolve())
    return read_hdf5_file(abs_path)

def _load_velocity_file(filepath: str):
    """Load velocity data from either VTI or HDF5 file"""
    # Normalize to absolute path for consistent caching
    abs_filepath = str(Path(filepath).resolve())
    filepath_lower = abs_filepath.lower()
    if filepath_lower.endswith(('.h5', '.hdf5')):
        return _cached_read_hdf5(abs_filepath)
    elif filepath_lower.endswith('.vti'):
        return _cached_read_vti(abs_filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Expected .vti, .h5, or .hdf5")

def _safe_minmax(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(a.min()), float(a.max())
    if vmin == vmax:
        if vmin == 0.0:
            vmax = 1.0
        else:
            vmax = vmin * (1.0 + 1e-6) if vmin > 0 else vmin * (1.0 - 1e-6)
    return vmin, vmax

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
        "viridis", "cividis", "plasma", "magma", "inferno",
        "turbo", "rainbow", "jet", "portland", "rdbu",
        "spectral", "ice", "electric", "hot", "icefire",
        "greys", "ylorrd", "blues", "reds", "greens"
    ]

def _create_slice_surface(x_coords, y_coords, z_coords, field_slice, vmin, vmax, cmap, opacity):
    """Create a surface trace for a slice plane
    Matches ParaView's coordinate system: X horizontal, Y vertical, Z out-of-plane
    """
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
        hovertemplate="Value: %{surfacecolor:.4f}<extra></extra>",
        connectgaps=False
    )


# -----------------------------
# Main
# -----------------------------
def main():
    inject_theme_css()
    
    st.title("🔬 3D Volume Viewer")
    st.markdown("**Interactive 3D Volume Visualization with ParaView-like Controls**")

    # Get data directories
    data_dirs = st.session_state.get("data_directories", [])
    if not data_dirs and st.session_state.get("data_directory"):
        data_dirs = [st.session_state.data_directory]
    
    if not data_dirs:
        st.warning("Please select a data directory from the main page.")
        return
    
    # Process ALL directories independently - collect files from all
    all_vti_files = []
    all_hdf5_files = []
    
    for data_dir_path in data_dirs:
        # Resolve path to ensure it works regardless of how it was stored
        try:
            data_dir = Path(data_dir_path).resolve()
            if data_dir.exists() and data_dir.is_dir():
                # Collect files from THIS directory independently
                dir_vti = sorted(
                    glob.glob(str(data_dir / "*.vti")) + 
                    glob.glob(str(data_dir / "*.VTI")),
                    key=natural_sort_key
                )
                dir_hdf5 = sorted(
                    glob.glob(str(data_dir / "*.h5")) + 
                    glob.glob(str(data_dir / "*.H5")) +
                    glob.glob(str(data_dir / "*.hdf5")) + 
                    glob.glob(str(data_dir / "*.HDF5")),
                    key=natural_sort_key
                )
                all_vti_files.extend(dir_vti)
                all_hdf5_files.extend(dir_hdf5)
        except Exception:
            continue  # Skip invalid directories
    
    # Use first directory for metadata storage
    data_dir = Path(data_dirs[0]).resolve()
    
    # File type selector (like ParaView, VisIt, etc.)
    st.sidebar.header("📁 File Selection")
    
    # Determine available file types from ALL directories
    has_vti = len(all_vti_files) > 0
    has_hdf5 = len(all_hdf5_files) > 0
    
    if not has_vti and not has_hdf5:
        st.error("No 3D velocity files found in any of the selected directories.")
        st.info("Expected files: `*.vti`, `*.h5`, or `*.hdf5` (e.g., `velocity_50000.vti` or `velocity_50000.h5`)")
        return
    
    # Use combined file lists
    vti_files = all_vti_files
    hdf5_files = all_hdf5_files
    
    # File type selector
    file_type_options = []
    if has_vti:
        file_type_options.append(f"VTI ({len(vti_files)} files)")
    if has_hdf5:
        file_type_options.append(f"HDF5 ({len(hdf5_files)} files)")
    if has_vti and has_hdf5:
        file_type_options.append("Both (VTI + HDF5)")
    
    # Initialize file type selection in session state
    if 'file_type_selection' not in st.session_state:
        # Default to first available type, or "Both" if both are available
        if len(file_type_options) == 1:
            st.session_state.file_type_selection = file_type_options[0]
        elif "Both" in file_type_options:
            st.session_state.file_type_selection = "Both (VTI + HDF5)"
        else:
            st.session_state.file_type_selection = file_type_options[0]
    
    selected_file_type = st.sidebar.radio(
        "File Extension",
        options=file_type_options,
        index=file_type_options.index(st.session_state.file_type_selection) if st.session_state.file_type_selection in file_type_options else 0,
        key="file_type_radio"
    )
    st.session_state.file_type_selection = selected_file_type
    
    # Filter files based on selection
    if selected_file_type.startswith("VTI"):
        all_files = vti_files
    elif selected_file_type.startswith("HDF5"):
        all_files = hdf5_files
    else:  # Both
        all_files = vti_files + hdf5_files
    
    # Extract iteration numbers from selected files (general pattern matching)
    # Handles: Velocity_1000.vti, Velocity1000.vti, data_42.h5, etc.
    iterations = []
    for f in all_files:
        filename = Path(f).name
        # Try pattern with underscore: _NUMBER.ext
        match = re.search(r'_(\d+)\.(vti|h5|hdf5)', filename, re.IGNORECASE)
        if not match:
            # Try pattern without underscore: NUMBER.ext (before file extension)
            match = re.search(r'(\d+)\.(vti|h5|hdf5)', filename, re.IGNORECASE)
        if match:
            iterations.append(int(match.group(1)))
        else:
            # If no number found, use None (will show time step index instead)
            iterations.append(None)

    # Time Control
    st.sidebar.header("⏱️ Time Control")
    
    # Show file count for selected type
    st.sidebar.caption(f"Found {len(all_files)} files")
    
    # Initialize/reset file index when file type changes
    # Track previous file type to detect changes
    if 'prev_file_type' not in st.session_state:
        st.session_state.prev_file_type = selected_file_type
        st.session_state.file_index = 0
        st.session_state.initial_load = True  # Flag for first load
    elif st.session_state.prev_file_type != selected_file_type:
        # File type changed, reset to first file (index 0)
        st.session_state.file_index = 0
        st.session_state.prev_file_type = selected_file_type
        st.session_state.initial_load = True
    
    # Initialize file_index if not set (always start at 0 = first file)
    if 'file_index' not in st.session_state:
        st.session_state.file_index = 0
        st.session_state.initial_load = True
    
    # Get initial load flag
    initial_load = st.session_state.get('initial_load', False)
    
    # Ensure file_index is within valid bounds [0, len(all_files)-1]
    # Time step 0 = first file, time step 1 = second file, etc.
    st.session_state.file_index = max(0, min(st.session_state.file_index, len(all_files) - 1))
    
    # Previous / Next buttons - update file_index before slider reads it
    col_t1, col_t2, col_t3 = st.sidebar.columns([1, 2, 1])
    
    # Previous button: go to previous file (decrease index)
    if col_t1.button("◀", key="prev_file", help="Previous time step"):
        if st.session_state.file_index > 0:
            st.session_state.file_index -= 1
            # FIX: Force the slider key to match the new index
            st.session_state.slider_index = st.session_state.file_index
        # Note: Streamlit automatically reruns on button click, no need for explicit st.rerun()
    
    # Next button: go to next file (increase index)
    if col_t3.button("▶", key="next_file", help="Next time step"):
        if st.session_state.file_index < len(all_files) - 1:
            st.session_state.file_index += 1
            # FIX: Force the slider key to match the new index
            st.session_state.slider_index = st.session_state.file_index
        # Note: Streamlit automatically reruns on button click, no need for explicit st.rerun()
    
    # Time step slider: 0-indexed (0 = first file, 1 = second file, ..., 99 = 100th file)
    # The slider value comes from session state (updated by buttons or previous slider interaction)
    file_index = col_t2.slider(
        "Time Step",
        0, len(all_files) - 1,
        value=st.session_state.file_index,
        key="slider_index"
    )
    
    # Update session state from slider (when user drags slider)
    # This ensures slider and buttons stay in sync
    if file_index != st.session_state.file_index:
        st.session_state.file_index = file_index
    
    # Double-check bounds (should always be valid after slider)
    file_index = max(0, min(file_index, len(all_files) - 1))
    
    # Select file at this index: all_files[0] = first file, all_files[1] = second file, etc.
    selected_file = all_files[file_index]
    filename = Path(selected_file).name
    
    # Display iteration number if available, otherwise show time step
    iteration = iterations[file_index]
    st.sidebar.caption(f"File: {filename}")
    if iteration is not None:
        st.sidebar.caption(f"Iteration: {iteration}")
    else:
        st.sidebar.caption(f"Time Step: {file_index}")
    
    # HDF5-specific options
    is_hdf5 = selected_file.lower().endswith(('.h5', '.hdf5'))

    # Load velocity data (VTI or HDF5)
    try:
        file_ext = Path(selected_file).suffix.lower()
        file_type = "HDF5" if file_ext in ['.h5', '.hdf5'] else "VTI"
        
        # Use absolute path to ensure consistent file loading and caching
        abs_selected_file = str(Path(selected_file).resolve())
        
        # On initial load, clear cache to ensure fresh data
        if initial_load:
            _cached_read_vti.clear()
            _cached_read_hdf5.clear()
            st.session_state.initial_load = False
        
        with st.spinner(f"Loading {file_type} file {file_index + 1}/{len(all_files)} ({filename})..."):
            vti_data = _load_velocity_file(abs_selected_file)
        
        velocity = vti_data['velocity']
        
        # Ensure velocity has the correct shape
        if velocity is None or len(velocity.shape) != 4:
            raise ValueError(f"Invalid velocity data shape: {velocity.shape if velocity is not None else 'None'}")
        
        # Both VTI and HDF5 readers now apply transpose to fix x/y swap
        # No additional transpose needed here - data is already in correct orientation
        nx, ny, nz = velocity.shape[:3]
        
        # Verify data is valid (check for NaN/Inf)
        if np.any(np.isnan(velocity)) or np.any(np.isinf(velocity)):
            st.warning(f"⚠️ File {filename} contains NaN or Inf values. Visualization may be incorrect.")
        
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
            ["ux", "uy", "uz", "Velocity Magnitude",
             "Vorticity Magnitude", "ωx", "ωy", "ωz",
             "Q_S^S", "Q Invariant", "R Invariant"],
            index=0,  # Default to ux to match ParaView
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
        elif field_type == "Q_S^S":
            with st.spinner("Computing Q_S^S..."):
                field = compute_qs_s(velocity)
        elif field_type == "Q Invariant":
            with st.spinner("Computing Q invariant..."):
                field = compute_q_invariant(velocity)
        elif field_type == "R Invariant":
            with st.spinner("Computing R invariant..."):
                field = compute_r_invariant(velocity)

        # Downsample
        field_ds = _downsample3d(field, downsample_step)
        nx_d, ny_d, nz_d = field_ds.shape
        xg, yg, zg = _make_grid(nx_d, ny_d, nz_d)

        vmin, vmax = _safe_minmax(field_ds)

        # Visualization modes
        st.sidebar.markdown("---")
        st.sidebar.subheader("👁️ Display Modes")
        show_volume = st.sidebar.checkbox("Volume rendering", value=False, key="show_vol")
        show_slices = st.sidebar.checkbox("Orthogonal slices", value=True, key="show_slices")
        show_surface = st.sidebar.checkbox("Surface", value=False, key="show_surface")
        show_iso = st.sidebar.checkbox("Isosurface", value=False, key="show_iso")

        # Colormap (default to rdbu to match ParaView's typical velocity visualization)
        cmap_options = _colormap_options()
        rdbu_index = cmap_options.index("rdbu") if "rdbu" in cmap_options else 0
        cmap = st.sidebar.selectbox("Colormap", cmap_options, index=rdbu_index, key="colormap")

        # Value range + opacity controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎛️ Rendering Controls")
        if vmax <= vmin:
            vmax = vmin + 1.0 if vmin >= 0 else vmin - 1.0
        
        # Color max for contrast control (clips low values to reveal turbulence)
        cmax = st.sidebar.slider(
            "Color Max (Contrast)",
            float(vmin), float(vmax),
            float(vmax) * 0.6,
            help="Lower values reveal turbulent structures by clipping low-energy regions",
            key="color_max"
        )
        
        vrange = st.sidebar.slider(
            "Value range",
            min_value=float(vmin), max_value=float(vmax),
            value=(float(vmin), float(cmax)),
            step=(vmax - vmin) / 200 if vmax > vmin else 1.0,
            key="vrange"
        )

        if show_volume:
            vol_opacity = st.sidebar.slider(
                "Volume opacity", 0.01, 0.8, 0.15, 0.01,
                help="Higher = denser fog-like volume. Lower values (0.1-0.2) work better for turbulence.",
                key="vol_opacity"
            )
            vol_surface_count = st.sidebar.slider(
                "Volume surfaces", 5, 40, 20, 1,
                help="More surfaces = richer volume but heavier.",
                key="vol_surfaces"
            )

        if show_iso:
            # Special threshold slider for Q_S^S method (based on paper thresholds: 2.5, 3.5, 5.0, 6.5)
            if field_type == "Q_S^S":
                st.sidebar.markdown("**Q_S^S Threshold (Paper values: 2.5-6.5)**")
                qss_threshold = st.sidebar.slider(
                    "Q_S^S threshold",
                    min_value=0.0, max_value=10.0,
                    value=5.0,
                    step=0.1,
                    help="Paper thresholds: 32³=2.5, 64³=3.5, 128³=5.0, 256³=6.5",
                    key="qss_threshold"
                )
                iso_value = qss_threshold
            else:
                iso_min, iso_max = float(vrange[0]), float(vrange[1])
                if iso_max <= iso_min:
                    iso_max = iso_min + 1.0 if iso_min >= 0 else iso_min - 1.0
                iso_value = st.sidebar.slider(
                    "Isosurface value",
                    min_value=iso_min, max_value=iso_max,
                    value=float((iso_min + iso_max) / 2),
                    step=(iso_max - iso_min) / 200 if iso_max > iso_min else 1.0,
                    key="iso_value"
                )
            iso_opacity = st.sidebar.slider(
                "Isosurface opacity", 0.05, 1.0, 0.4, 0.05,
                key="iso_opacity"
            )

        if show_surface:
            surface_opacity = st.sidebar.slider(
                "Surface opacity", 0.05, 1.0, 0.8, 0.05,
                key="surface_opacity"
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

        # Plot Style (simplified for 3D)
        st.sidebar.markdown("---")
        with st.sidebar.expander("🎨 Plot Style (persistent)", expanded=False):
            # Initialize plot style
            if "plot_style_3d" not in st.session_state:
                st.session_state.plot_style_3d = default_plot_style()
            
            ps = dict(st.session_state.plot_style_3d)
            
            # Apply theme
            current_theme = st.session_state.get("theme", "Light Scientific")
            ps = apply_theme_to_plot_style(ps, current_theme)
            
            # Theme selector
            st.markdown("**Theme**")
            themes = ["Light Scientific", "Dark Scientific"]
            theme_idx = themes.index(current_theme) if current_theme in themes else 0
            selected_theme = st.selectbox("Theme", themes, index=theme_idx, key="3d_theme_selector")
            if selected_theme != current_theme:
                st.session_state.theme = selected_theme
                ps = apply_theme_to_plot_style(ps, selected_theme)
            
            # Background colors
            st.markdown("---")
            st.markdown("**Backgrounds**")
            ps["plot_bgcolor"] = st.color_picker("Scene background", ps.get("plot_bgcolor", "#FFFFFF"), key="3d_plot_bgcolor")
            ps["paper_bgcolor"] = st.color_picker("Paper background", ps.get("paper_bgcolor", "#FFFFFF"), key="3d_paper_bgcolor")
            
            # Grid color
            st.markdown("---")
            st.markdown("**Grid**")
            ps["grid_color"] = st.color_picker("Grid color", ps.get("grid_color", "#B0B0B0"), key="3d_grid_color")
            
            # Font sizes
            st.markdown("---")
            st.markdown("**Fonts**")
            ps["title_size"] = st.slider("Plot title size", 10, 32, int(ps.get("title_size", 16)), key="3d_title_size")
            ps["axis_title_size"] = st.slider("Axis title size", 8, 28, int(ps.get("axis_title_size", 14)), key="3d_axis_title_size")
            
            # Figure size
            st.markdown("---")
            render_figure_size_ui(ps, key_prefix="3d")
            
            # Plot title
            st.markdown("---")
            render_plot_title_ui(ps, key_prefix="3d")
            
            # Save to session state
            st.session_state.plot_style_3d = ps
        
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

        # Volume rendering (improved for turbulence visualization)
        if show_volume:
            # Cut off bottom 10% of values to make low-energy regions transparent
            isomin_val = vmin + (cmax - vmin) * 0.1 if cmax > vmin else vmin
            fig.add_trace(go.Volume(
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

        # Surface rendering (outer faces of domain)
        if show_surface:
            # Front face (z = 0)
            z_front = np.zeros((nx_d, ny_d))
            x_coords = np.arange(nx_d)[:, None] * np.ones((1, ny_d))
            y_coords = np.ones((nx_d, 1)) * np.arange(ny_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_coords, y_coords, z_front,
                field_clip[:, :, 0],
                vmin, cmax, cmap, surface_opacity
            ))
            
            # Back face (z = nz_d-1)
            z_back = np.full((nx_d, ny_d), nz_d - 1)
            fig.add_trace(_create_slice_surface(
                x_coords, y_coords, z_back,
                field_clip[:, :, nz_d - 1],
                vmin, cmax, cmap, surface_opacity
            ))
            
            # Left face (x = 0)
            x_left = np.zeros((ny_d, nz_d))
            y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
            z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_left, y_coords, z_coords,
                field_clip[0, :, :],
                vmin, cmax, cmap, surface_opacity
            ))
            
            # Right face (x = nx_d-1)
            x_right = np.full((ny_d, nz_d), nx_d - 1)
            fig.add_trace(_create_slice_surface(
                x_right, y_coords, z_coords,
                field_clip[nx_d - 1, :, :],
                vmin, cmax, cmap, surface_opacity
            ))
            
            # Bottom face (y = 0)
            y_bottom = np.zeros((nx_d, nz_d))
            x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
            z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_coords, y_bottom, z_coords,
                field_clip[:, 0, :],
                vmin, cmax, cmap, surface_opacity
            ))
            
            # Top face (y = ny_d-1)
            y_top = np.full((nx_d, nz_d), ny_d - 1)
            fig.add_trace(_create_slice_surface(
                x_coords, y_top, z_coords,
                field_clip[:, ny_d - 1, :],
                vmin, cmax, cmap, surface_opacity
            ))

        # Orthogonal slice planes (matching ParaView's coordinate system)
        if show_slices:
            # XY plane at z = slice_z (ParaView: Z-slice showing X-Y plane)
            # Data: field_clip[i, j, k] where i=x, j=y, k=z (Fortran order)
            z_plane = np.full((nx_d, ny_d), slice_z)
            x_coords = np.arange(nx_d)[:, None] * np.ones((1, ny_d))
            y_coords = np.ones((nx_d, 1)) * np.arange(ny_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_coords, y_coords, z_plane,
                field_clip[:, :, slice_z],
                vmin, cmax, cmap, slice_opacity
            ))

            # XZ plane at y = slice_y (ParaView: Y-slice showing X-Z plane)
            y_plane = np.full((nx_d, nz_d), slice_y)
            x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
            z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_coords, y_plane, z_coords,
                field_clip[:, slice_y, :],
                vmin, cmax, cmap, slice_opacity
            ))

            # YZ plane at x = slice_x (ParaView: X-slice showing Y-Z plane)
            x_plane = np.full((ny_d, nz_d), slice_x)
            y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
            z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
            fig.add_trace(_create_slice_surface(
                x_plane, y_coords, z_coords,
                field_clip[slice_x, :, :],
                vmin, cmax, cmap, slice_opacity
            ))

        # Camera presets
        camera_dicts = {
            "Isometric": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
            "XY": dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
            "XZ": dict(eye=dict(x=0, y=2.5, z=0), up=dict(x=0, y=0, z=1)),
            "YZ": dict(eye=dict(x=2.5, y=0, z=0), up=dict(x=0, y=1, z=0)),
            "Custom": dict(eye=dict(x=1.4, y=1.4, z=1.2))
        }

        # Get plot style
        ps = st.session_state.get("plot_style_3d", default_plot_style())
        current_theme = st.session_state.get("theme", "Light Scientific")
        ps = apply_theme_to_plot_style(ps, current_theme)
        
        # Apply figure size
        layout_kwargs = {}
        layout_kwargs = apply_figure_size(layout_kwargs, ps)
        default_height = layout_kwargs.get("height", 600)
        
        # Get colors from plot style
        scene_bgcolor = ps.get("plot_bgcolor", "#FFFFFF")
        paper_bgcolor = ps.get("paper_bgcolor", "#FFFFFF")
        grid_color = ps.get("grid_color", "#B0B0B0")
        axis_title_size = ps.get("axis_title_size", 14)
        font_color = ps.get("font_color", "#000000")
        title_size = ps.get("title_size", 16)
        
        # Build layout kwargs
        layout_kwargs_title = {}
        
        # Add title if enabled
        if ps.get("show_plot_title", False) and ps.get("plot_title"):
            layout_kwargs_title["title"] = dict(
                text=ps.get("plot_title"),
                font=dict(
                    family=ps.get("font_family", "Arial"),
                    size=title_size,
                    color=font_color
                )
            )
        
        # Layout
        fig.update_layout(
            height=default_height,
            **layout_kwargs_title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=camera_dicts.get(camera_preset, camera_dicts["Isometric"]),
                bgcolor=scene_bgcolor,
                xaxis=dict(
                    backgroundcolor=scene_bgcolor,
                    gridcolor=grid_color,
                    showbackground=True,
                    title_font=dict(size=axis_title_size, color=font_color),
                    tickfont=dict(color=font_color)
                ),
                yaxis=dict(
                    backgroundcolor=scene_bgcolor,
                    gridcolor=grid_color,
                    showbackground=True,
                    title_font=dict(size=axis_title_size, color=font_color),
                    tickfont=dict(color=font_color)
                ),
                zaxis=dict(
                    backgroundcolor=scene_bgcolor,
                    gridcolor=grid_color,
                    showbackground=True,
                    title_font=dict(size=axis_title_size, color=font_color),
                    tickfont=dict(color=font_color)
                )
            ),
            legend=dict(
                itemsizing="constant",
                x=1.02,
                y=1,
                bgcolor=f"rgba{tuple(list(hex_to_rgb(paper_bgcolor)) + [0.8])}",
                bordercolor=grid_color,
                borderwidth=1,
                font=dict(color=font_color)
            ),
            margin=dict(
                l=0, 
                r=0, 
                t=50 if (ps.get("show_plot_title", False) and ps.get("plot_title")) else 0, 
                b=0
            ),
            paper_bgcolor=paper_bgcolor
        )

        # Display plot
        st.plotly_chart(fig, width='stretch', config={
            'displayModeBar': True,
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'{Path(selected_file).stem}_3d_view',
                'height': 600,
                'width': 1200,
                'scale': 2
            }
        })

        # Capture button for report
        capture_title = f"3D Volume Viewer - {field_type}"
        if iteration is not None:
            capture_title += f" (Iteration {iteration})"
        else:
            capture_title += f" (Time Step {file_index})"
        capture_button(fig, title=capture_title, source_page="3D Volume Viewer")

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

        # Theory & Equations
        with st.expander("📚 Theory & Equations", expanded=False):
            st.markdown("### Velocity Fields")
            st.markdown("**Velocity magnitude:**")
            st.latex(r"|\mathbf{u}| = \sqrt{u_x^2 + u_y^2 + u_z^2}")
            
            st.markdown("### Vorticity")
            st.markdown("**Vorticity vector:**")
            st.latex(r"\boldsymbol{\omega} = \nabla \times \mathbf{u}")
            st.markdown("**Components:**")
            st.latex(r"\omega_x = \frac{\partial u_z}{\partial y} - \frac{\partial u_y}{\partial z}, \quad \omega_y = \frac{\partial u_x}{\partial z} - \frac{\partial u_z}{\partial x}, \quad \omega_z = \frac{\partial u_y}{\partial x} - \frac{\partial u_x}{\partial y}")
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
            st.latex(r"R = -\frac{1}{3}A_{ij}A_{jk}A_{ki} = -\frac{1}{3}\left(S_{ij}S_{jk}S_{ki} + \frac{3}{4}\omega_i\omega_j S_{ij}\right)")
            
            st.markdown("where $A_{ij} = \\partial u_i/\\partial x_j$ is the velocity gradient tensor.")

        # Export options - using shared export function
        export_panel(fig, data_dir, base_name=f"{Path(selected_file).stem}_3d_view")

    except Exception as e:
        file_type = "HDF5" if Path(selected_file).suffix.lower() in ['.h5', '.hdf5'] else "VTI"
        st.error(f"Error loading {file_type} file: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())


if __name__ == "__main__":
    main()

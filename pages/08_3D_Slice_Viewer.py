"""
3D Slice Viewer Page
Interactive 3D volume visualization for VTI files
"""

import streamlit as st
import numpy as np
import glob
import re
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data_readers.vti_reader import read_vti_file, compute_velocity_magnitude, compute_vorticity
from utils.file_detector import natural_sort_key

def main():
    st.title("🔬 3D Slice Viewer")
    
    if 'data_directory' not in st.session_state or not st.session_state.data_directory:
        st.warning("Please select a data directory from the main page.")
        return
    
    data_dir = Path(st.session_state.data_directory)
    
    # Find VTI files
    vti_files = sorted(glob.glob(str(data_dir / "*.vti")), key=natural_sort_key)
    
    if not vti_files:
        st.error("No VTI files found in the selected directory.")
        st.info("VTI files should be named like: `velocity_50000.vti` or `*_*.vti`")
        return
    
    st.sidebar.header("Options")
    
    # File selection
    selected_file = st.sidebar.selectbox("Select VTI file:", vti_files, format_func=lambda x: Path(x).name)
    
    # Field selection
    field_type = st.sidebar.selectbox("Field to visualize:", 
                                      ["Velocity Magnitude", "ux", "uy", "uz", "Vorticity"])
    
    # Load VTI file
    try:
        vti_data = read_vti_file(selected_file)
        nx, ny, nz = vti_data['dimensions']
        velocity = vti_data['velocity']
        
        st.success(f"Loaded: {Path(selected_file).name}")
        st.info(f"Dimensions: {nx} × {ny} × {nz}")
        
        # Compute field to visualize
        if field_type == "Velocity Magnitude":
            field = compute_velocity_magnitude(velocity)
        elif field_type == "ux":
            field = velocity[:, :, :, 0]
        elif field_type == "uy":
            field = velocity[:, :, :, 1]
        elif field_type == "uz":
            field = velocity[:, :, :, 2]
        elif field_type == "Vorticity":
            vorticity = compute_vorticity(velocity)
            field = np.sqrt(vorticity[:, :, :, 0]**2 + 
                           vorticity[:, :, :, 1]**2 + 
                           vorticity[:, :, :, 2]**2)
        
        # Slice selection
        st.sidebar.subheader("Slice Position")
        slice_x = st.sidebar.slider("X Slice", 0, nx-1, nx//2)
        slice_y = st.sidebar.slider("Y Slice", 0, ny-1, ny//2)
        slice_z = st.sidebar.slider("Z Slice", 0, nz-1, nz//2)
        
        # Display slices
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("XY Slice (Z = {})".format(slice_z))
            st.image(field[:, :, slice_z], use_container_width=True, clamp=True)
        
        with col2:
            st.subheader("XZ Slice (Y = {})".format(slice_y))
            st.image(field[:, slice_y, :], use_container_width=True, clamp=True)
        
        with col3:
            st.subheader("YZ Slice (X = {})".format(slice_x))
            st.image(field[slice_x, :, :], use_container_width=True, clamp=True)
        
        st.info("3D slice viewer - Full interactive visualization coming soon (requires pyvista or plotly volume rendering)")
        
    except Exception as e:
        st.error(f"Error loading VTI file: {e}")

if __name__ == "__main__":
    main()


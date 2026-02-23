"""
Theory and Equations Page
D3Q19 lattice visualization, MRT matrix generator, all mathematical equations
"""

import streamlit as st
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

from utils.theme_config import inject_theme_css
from utils.export_figs import export_panel
from utils.report_builder import capture_button
from utils.mrt_matrix import render_mrt_matrix_generator
from visualizations.d3q19_lattice import plot_d3q19_lattice, DEFAULT_LATTICE_COLORS
from content.theory_equations_content import (
    get_ns_equations_sections,
    get_ns_equations_footer,
    get_lbm_formulation_sections,
    get_lbm_formulation_footer,
)
st.set_page_config(page_icon="⚫", layout="wide")

def main():
    # Apply theme CSS (persists across pages)
    inject_theme_css()
    st.title("📚 Theory & Equations")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs(["NS-Equations", "LBM Formulation", "D3Q19 Lattice Visualization", "MRT Matrix Generator"])
    
    with tab1:
        st.header("From Navier-Stokes to LBM")
        for title, content, expanded in get_ns_equations_sections():
            with st.expander(title, expanded=expanded):
                st.markdown(content)
        st.divider()
        st.markdown(get_ns_equations_footer())
    
    with tab2:
        st.header("Lattice Boltzmann Method")
        st.info("**Primary focus:** MRT (Multiple Relaxation Time) | **Reference:** BGK/SRT (shown for app flexibility)")
        for title, content, expanded in get_lbm_formulation_sections():
            if title is None:
                st.markdown(content)
            else:
                with st.expander(title, expanded=expanded):
                    st.markdown(content)
        st.divider()
        st.markdown(get_lbm_formulation_footer())
    
    # D3Q19 Lattice Visualization Tab
    with tab3:
        st.header("D3Q19 Lattice Stencil Visualization")
        st.markdown("Interactive 3D visualization of the D3Q19 lattice stencil with full customization controls.")
        
        # Initialize session state for D3Q19 settings
        if 'd3q19_settings' not in st.session_state:
            st.session_state.d3q19_settings = _default_d3q19_settings()
        
        # Sidebar controls
        with st.sidebar:
            st.header("D3Q19 Visualization Controls")
            
            # Stencil Configuration
            with st.expander("Stencil Configuration", expanded=True):
                st.session_state.d3q19_settings['show_vectors'] = st.checkbox(
                    "Show Vectors", 
                    value=st.session_state.d3q19_settings.get('show_vectors', True),
                    key="d3q19_show_vectors"
                )
                st.session_state.d3q19_settings['vector_scale'] = st.slider(
                    "Vector Length Scale", 
                    0.1, 2.0, 
                    value=st.session_state.d3q19_settings.get('vector_scale', 1.0),
                    step=0.1,
                    key="d3q19_vector_scale"
                )
                st.session_state.d3q19_settings['vector_width'] = st.slider(
                    "Vector Width", 
                    1.0, 10.0, 
                    value=st.session_state.d3q19_settings.get('vector_width', 3.0),
                    step=0.5,
                    key="d3q19_vector_width"
                )
            
            # Node Appearance
            with st.expander("🔵 Node Appearance", expanded=False):
                node_style_options = ['circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x']
                current_node_style = st.session_state.d3q19_settings.get('node_style', 'circle')
                st.session_state.d3q19_settings['node_style'] = st.selectbox(
                    "Node Style",
                    node_style_options,
                    index=node_style_options.index(current_node_style) if current_node_style in node_style_options else 0,
                    key="d3q19_node_style"
                )
                st.session_state.d3q19_settings['node_size'] = st.slider(
                    "Node Size", 
                    5.0, 50.0, 
                    value=st.session_state.d3q19_settings.get('node_size', 10.0),
                    step=1.0,
                    key="d3q19_node_size"
                )
                st.session_state.d3q19_settings['node_opacity'] = st.slider(
                    "Node Opacity", 
                    0.0, 1.0, 
                    value=st.session_state.d3q19_settings.get('node_opacity', 0.8),
                    step=0.1,
                    key="d3q19_node_opacity"
                )
                st.session_state.d3q19_settings['node_edge_color'] = st.color_picker(
                    "Node Edge Color",
                    value=st.session_state.d3q19_settings.get('node_edge_color', '#000000'),
                    key="d3q19_node_edge_color"
                )
                st.session_state.d3q19_settings['node_edge_width'] = st.slider(
                    "Node Edge Width", 
                    0.0, 5.0, 
                    value=st.session_state.d3q19_settings.get('node_edge_width', 1.0),
                    step=0.1,
                    key="d3q19_node_edge_width"
                )
                st.divider()
                st.markdown("**Origin Marker**")
                origin_style_options = ['circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x']
                current_origin_style = st.session_state.d3q19_settings.get('origin_style', 'circle-open')
                st.session_state.d3q19_settings['origin_style'] = st.selectbox(
                    "Origin Style",
                    origin_style_options,
                    index=origin_style_options.index(current_origin_style) if current_origin_style in origin_style_options else 0,
                    key="d3q19_origin_style"
                )
                st.session_state.d3q19_settings['origin_size'] = st.slider(
                    "Origin Marker Size", 
                    5.0, 50.0, 
                    value=st.session_state.d3q19_settings.get('origin_size', 15.0),
                    step=1.0,
                    key="d3q19_origin_size"
                )
                st.session_state.d3q19_settings['origin_color'] = st.color_picker(
                    "Origin Color",
                    value=st.session_state.d3q19_settings.get('origin_color', '#052020'),
                    key="d3q19_origin_color"
                )
            
            # Vector Styling
            with st.expander("➡️ Vector Styling", expanded=False):
                st.session_state.d3q19_settings['vector_color'] = st.color_picker(
                    "Vector Color",
                    value=st.session_state.d3q19_settings.get('vector_color', '#FF0000'),
                    key="d3q19_vector_color"
                )
                st.session_state.d3q19_settings['vector_opacity'] = st.slider(
                    "Vector Opacity", 
                    0.0, 1.0, 
                    value=st.session_state.d3q19_settings.get('vector_opacity', 0.8),
                    step=0.1,
                    key="d3q19_vector_opacity"
                )
                st.session_state.d3q19_settings['vector_linestyle'] = st.selectbox(
                    "Vector Line Style",
                    ['solid', 'dash', 'dot', 'dashdot'],
                    index=['solid', 'dash', 'dot', 'dashdot'].index(
                        st.session_state.d3q19_settings.get('vector_linestyle', 'dashdot')
                    ),
                    key="d3q19_vector_linestyle"
                )
            
            # Labels
            with st.expander("🏷️ Labels", expanded=False):
                st.session_state.d3q19_settings['show_labels'] = st.checkbox(
                    "Show Labels", 
                    value=st.session_state.d3q19_settings.get('show_labels', True),
                    key="d3q19_show_labels"
                )
                st.session_state.d3q19_settings['label_prefix'] = st.text_input(
                    "Label Prefix (e.g., 'C' for C1, C2, ...)",
                    value=st.session_state.d3q19_settings.get('label_prefix', 'C'),
                    key="d3q19_label_prefix"
                )
                st.session_state.d3q19_settings['label_font_size'] = st.slider(
                    "Label Font Size", 
                    8, 24, 
                    value=st.session_state.d3q19_settings.get('label_font_size', 13),
                    step=1,
                    key="d3q19_label_font_size"
                )
                st.session_state.d3q19_settings['label_color'] = st.color_picker(
                    "Label Color",
                    value=st.session_state.d3q19_settings.get('label_color', '#000000'),
                    key="d3q19_label_color"
                )
            
            # Faces and Edges
            with st.expander("Faces & Edges", expanded=False):
                st.session_state.d3q19_settings['show_faces'] = st.checkbox(
                    "Show Colored Faces", 
                    value=st.session_state.d3q19_settings.get('show_faces', True),
                    key="d3q19_show_faces"
                )
                st.session_state.d3q19_settings['face_opacity'] = st.slider(
                    "Face Opacity", 
                    0.0, 1.0, 
                    value=st.session_state.d3q19_settings.get('face_opacity', 0.5),
                    step=0.1,
                    key="d3q19_face_opacity"
                )
                st.session_state.d3q19_settings['show_cube_edges'] = st.checkbox(
                    "Show Cube Edges", 
                    value=st.session_state.d3q19_settings.get('show_cube_edges', True),
                    key="d3q19_show_cube_edges"
                )
                st.session_state.d3q19_settings['cube_edge_color'] = st.color_picker(
                    "Cube Edge Color",
                    value=st.session_state.d3q19_settings.get('cube_edge_color', '#000000'),
                    key="d3q19_cube_edge_color"
                )
                st.session_state.d3q19_settings['cube_edge_width'] = st.slider(
                    "Cube Edge Width", 
                    0.5, 5.0, 
                    value=st.session_state.d3q19_settings.get('cube_edge_width', 1.0),
                    step=0.5,
                    key="d3q19_cube_edge_width"
                )
            
            # View Controls
            with st.expander("👁️ View Controls", expanded=False):
                # View presets - place BEFORE sliders so button updates take precedence
                st.markdown("**Quick View Presets:**")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Front View", key="d3q19_front", width='stretch'):
                        st.session_state.d3q19_settings['camera_elevation'] = 0.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 0.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                    if st.button("Side View", key="d3q19_side", width='stretch'):
                        st.session_state.d3q19_settings['camera_elevation'] = 0.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 90.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                with col2:
                    if st.button("Top View", key="d3q19_top", width='stretch'):
                        st.session_state.d3q19_settings['camera_elevation'] = 90.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 0.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                    if st.button("Isometric", key="d3q19_iso", width='stretch'):
                        st.session_state.d3q19_settings['camera_elevation'] = 35.0
                        st.session_state.d3q19_settings['camera_azimuth'] = 45.0
                        st.session_state.d3q19_settings['camera_zoom'] = 1.0
                        st.rerun()
                
                st.markdown("---")
                st.markdown("**Manual Camera Controls:**")
                st.session_state.d3q19_settings['camera_elevation'] = st.slider(
                    "Camera Elevation (degrees)", 
                    -90.0, 90.0, 
                    value=st.session_state.d3q19_settings.get('camera_elevation', 9.0),
                    step=1.0,
                    key="d3q19_camera_elevation"
                )
                st.session_state.d3q19_settings['camera_azimuth'] = st.slider(
                    "Camera Azimuth (degrees)", 
                    -180.0, 180.0, 
                    value=st.session_state.d3q19_settings.get('camera_azimuth', 16.0),
                    step=1.0,
                    key="d3q19_camera_azimuth"
                )
                st.session_state.d3q19_settings['camera_zoom'] = st.slider(
                    "Camera Zoom", 
                    0.5, 3.0, 
                    value=st.session_state.d3q19_settings.get('camera_zoom', 1.0),
                    step=0.1,
                    key="d3q19_camera_zoom"
                )
            
            # Advanced Options
            with st.expander("⚙️ Advanced Options", expanded=False):
                st.session_state.d3q19_settings['show_axes'] = st.checkbox(
                    "Show Coordinate Axes", 
                    value=st.session_state.d3q19_settings.get('show_axes', False),
                    key="d3q19_show_axes"
                )
                st.session_state.d3q19_settings['show_axis_labels'] = st.checkbox(
                    "Show Axis Labels", 
                    value=st.session_state.d3q19_settings.get('show_axis_labels', False),
                    key="d3q19_show_axis_labels"
                )
                st.session_state.d3q19_settings['show_origin_marker'] = st.checkbox(
                    "Show Origin Marker", 
                    value=st.session_state.d3q19_settings.get('show_origin_marker', True),
                    key="d3q19_show_origin_marker"
                )
                st.session_state.d3q19_settings['show_grid'] = st.checkbox(
                    "Show Grid", 
                    value=st.session_state.d3q19_settings.get('show_grid', False),
                    key="d3q19_show_grid"
                )
                st.session_state.d3q19_settings['background_color'] = st.color_picker(
                    "Background Color",
                    value=st.session_state.d3q19_settings.get('background_color', '#FFFFFF'),
                    key="d3q19_background_color"
                )
            
            # Reset button
            st.markdown("---")
            if st.button("♻️ Reset to Defaults", key="d3q19_reset"):
                st.session_state.d3q19_settings = _default_d3q19_settings()
                
                # Clear widget state so widgets re-read from defaults
                widget_keys = [
                    "d3q19_show_vectors", "d3q19_vector_scale", "d3q19_vector_width",
                    "d3q19_node_style", "d3q19_node_size", "d3q19_node_opacity",
                    "d3q19_node_edge_color", "d3q19_node_edge_width",
                    "d3q19_origin_style", "d3q19_origin_size", "d3q19_origin_color",
                    "d3q19_vector_color", "d3q19_vector_opacity", "d3q19_vector_linestyle",
                    "d3q19_show_labels", "d3q19_label_prefix", "d3q19_label_font_size",
                    "d3q19_label_color", "d3q19_show_faces", "d3q19_face_opacity",
                    "d3q19_show_cube_edges", "d3q19_cube_edge_color", "d3q19_cube_edge_width",
                    "d3q19_camera_elevation", "d3q19_camera_azimuth", "d3q19_camera_zoom",
                    "d3q19_show_axes", "d3q19_show_axis_labels", "d3q19_show_origin_marker",
                    "d3q19_show_grid", "d3q19_background_color"
                ]
                for widget_key in widget_keys:
                    if widget_key in st.session_state:
                        del st.session_state[widget_key]
                
                st.toast("Reset.", icon="♻️")
                st.rerun()
        
        # Generate and display visualization with theme-aware colors
        current_theme = st.session_state.get("theme", "Light Scientific")
        is_dark = "Dark" in current_theme
        
        # Override colors for dark theme
        plot_settings = st.session_state.d3q19_settings.copy()
        if is_dark:
            # Use dark background
            if plot_settings.get('background_color', '#FFFFFF') == '#FFFFFF':
                plot_settings['background_color'] = '#1e1e1e'
            # Use light colors for labels and text
            if plot_settings.get('label_color', '#000000') == '#000000':
                plot_settings['label_color'] = '#d4d4d4'
            # Use light colors for cube edges if black (make borders visible)
            if plot_settings.get('cube_edge_color', '#000000') == '#000000':
                plot_settings['cube_edge_color'] = '#808080'  # Brighter gray for better visibility
            # Ensure cube edges are visible (increase width if too thin)
            if plot_settings.get('cube_edge_width', 1.0) < 1.5:
                plot_settings['cube_edge_width'] = 2.0
            # Use light grid color
            if 'grid_color' not in plot_settings or plot_settings.get('grid_color', '#000000') == '#000000':
                plot_settings['grid_color'] = '#3e3e42'
            # Use light color for node edges if black (for better visibility)
            if plot_settings.get('node_edge_color', '#000000') == '#000000':
                plot_settings['node_edge_color'] = '#d4d4d4'
        
        fig = plot_d3q19_lattice(**plot_settings)
        
        # Update title and axis label colors for dark theme
        if is_dark:
            fig.update_layout(
                title_font=dict(color='#ffffff'),
                scene=dict(
                    xaxis=dict(
                        title_font=dict(color='#d4d4d4'),
                        tickfont=dict(color='#d4d4d4')
                    ),
                    yaxis=dict(
                        title_font=dict(color='#d4d4d4'),
                        tickfont=dict(color='#d4d4d4')
                    ),
                    zaxis=dict(
                        title_font=dict(color='#d4d4d4'),
                        tickfont=dict(color='#d4d4d4')
                    )
                )
            )
        
        st.plotly_chart(
            fig, 
            width='stretch',
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'd3q19_lattice',
                    'height': 800,
                    'width': 800,
                    'scale': 2
                }
            }
        )
        
        # Capture to report
        capture_button(fig, title="D3Q19 Lattice Stencil Visualization", source_page="Theory & Equations")
        
        # Export options - using comprehensive export panel like other pages
        export_panel(fig, project_root, "d3q19_lattice")
    
    # MRT Matrix Generator Tab
    with tab4:
        render_mrt_matrix_generator()
    


def _default_d3q19_settings():
    """Return default settings for D3Q19 visualization"""
    return {
        'show_vectors': True,
        'vector_scale': 1.0,
        'vector_width': 3.0,
        'node_size': 10.0,
        'node_colors': DEFAULT_LATTICE_COLORS.copy(),
        'node_opacity': 0.8,
        'node_style': 'circle',
        'node_edge_color': '#000000',
        'node_edge_width': 1.0,
        'origin_size': 15.0,
        'origin_color': '#052020',
        'origin_style': 'circle-open',
        'vector_color': '#FF0000',
        'vector_opacity': 0.8,
        'vector_linestyle': 'dashdot',
        'show_vector_arrows': False,
        'arrow_head_size': 0.1,
        'show_labels': True,
        'label_prefix': 'C',
        'label_font_size': 13,
        'label_color': '#000000',
        'label_offset': 1.19,
        'show_faces': False,
        'face_opacity': 0.5,
        'show_cube_edges': True,
        'cube_edge_color': '#000000',
        'cube_edge_width': 2.0,
        'cube_edge_style': 'solid',
        'show_grid': False,
        'grid_color': '#808080',
        'grid_opacity': 0.3,
        'background_color': '#FFFFFF',
        'show_axes': False,
        'show_axis_labels': False,
        'show_origin_marker': True,
        'camera_elevation': 9.0,
        'camera_azimuth': 16.0,
        'camera_zoom': 1.0,
        'width': 800,
        'height': 800,
        'title': 'D3Q19 Lattice Stencil'
    }

if __name__ == "__main__":
    main()



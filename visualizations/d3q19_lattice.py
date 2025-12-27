"""
D3Q19 Lattice Visualization
Interactive 3D visualization of D3Q19 lattice stencil with full customization
"""

import numpy as np
import plotly.graph_objects as go
from typing import Dict, Optional, Tuple, List


# # D3Q19 lattice directions 
D3Q19_DIRX = np.array([1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0])
D3Q19_DIRY = np.array([0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0])
D3Q19_DIRZ = np.array([0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0])


# Default colors for lattice points (matching original)
DEFAULT_LATTICE_COLORS = [
    'red', 'red', 'green', 'green', 'yellow', 'yellow', 'cyan', 'cyan',
    'magenta', 'magenta', 'teal', 'teal', 'olive', 'olive', 'gray', 'gray',
    'navy', 'navy', 'chocolate'
]


def _create_dashed_segments(start, end, dash_length=0.08, gap_length=0.04):
    """Create multiple short line segments to simulate a tight dash pattern
    Args:
        start: [x, y, z] start point
        end: [x, y, z] end point
        dash_length: length of each dash segment
        gap_length: length of gap between dashes
    Returns:
        List of (start, end) tuples for each dash segment
    """
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    total_length = np.linalg.norm(direction)
    
    if total_length < 1e-10:
        return [(start, end)]
    
    unit_dir = direction / total_length
    segments = []
    current_pos = start.copy()
    
    while True:
        # Calculate remaining distance
        remaining = np.linalg.norm(end - current_pos)
        
        if remaining <= dash_length:
            # Last segment - draw to end
            segments.append((current_pos.copy(), end.copy()))
            break
        
        # Create a dash segment
        dash_end = current_pos + unit_dir * dash_length
        segments.append((current_pos.copy(), dash_end.copy()))
        
        # Move past the gap
        current_pos = dash_end + unit_dir * gap_length
        
        # Check if we've passed the end
        if np.linalg.norm(end - current_pos) < 0:
            break
    
    return segments


def _create_cube_edges():
    """Create all cube edge lines matching the original matplotlib code EXACTLY
    Returns: list of tuples (start, end, style) where style is 'solid' or 'dash'
    
    Matches original matplotlib ax.plot() calls exactly from lines 176-221
    Each line matches the original ax.plot([x1, x2], [y1, y2], [z1, z2]) format
    """
    edges = []
    
    # Front face (solid) - lines 178-180
    edges.append(([-1, -1, 1], [-1, 1, 1], 'solid'))    # line 178: x=[-1,-1], y=[-1,1], z=[1,1]
    edges.append(([1, -1, 1], [1, 1, 1], 'solid'))      # line 179: x=[1,1], y=[-1,1], z=[1,1]
    edges.append(([1, -1, -1], [1, 1, -1], 'solid'))    # line 180: x=[1,1], y=[-1,1], z=[-1,-1]
    
    # Back face (dashed) - lines 183-185
    edges.append(([-1, -1, 1], [-1, 1, 1], 'solid'))    # line 183: x=[-1,-1], y=[-1,1], z=[1,1] (duplicate)
    edges.append(([-1, -1, -1], [1, -1, -1], 'dash'))   # line 184: x=[-1,1], y=[-1,-1], z=[-1,-1]
    edges.append(([-1, -1, -1], [-1, 1, -1], 'dash'))   # line 185: x=[-1,-1], y=[-1,1], z=[-1,-1]
    
    # Right face (solid) - lines 188-191
    edges.append(([-1, 1, 1], [1, 1, 1], 'solid'))      # line 188: x=[-1,1], y=[1,1], z=[1,1]
    edges.append(([-1, 1, -1], [1, 1, -1], 'dash'))     # line 189: x=[-1,1], y=[1,1], z=[-1,-1]
    edges.append(([-1, 1, -1], [-1, 1, 1], 'solid'))    # line 190: x=[-1,-1], y=[1,1], z=[-1,1]
    edges.append(([1, 1, -1], [1, 1, 1], 'solid'))      # line 191: x=[1,1], y=[1,1], z=[-1,1]
    
    # Left face (dashed) - lines 194-196
    edges.append(([1, -1, -1], [1, -1, 1], 'solid'))    # line 194: x=[1,1], y=[-1,-1], z=[-1,1]
    edges.append(([-1, -1, -1], [-1, -1, 1], 'dash'))   # line 195: x=[-1,-1], y=[-1,-1], z=[-1,1]
    edges.append(([-1, -1, 1], [1, -1, 1], 'dash'))     # line 196: x=[-1,1], y=[-1,-1], z=[1,1]
    
    # Mid lines for the cube edges - lines 199-221
    # Top - lines 200-201
    edges.append(([0, -1, 1], [0, 1, 1], 'solid'))      # line 200: x=[0,0], y=[-1,1], z=[1,1]
    edges.append(([1, 0, 1], [-1, 0, 1], 'solid'))      # line 201: x=[1,-1], y=[0,0], z=[1,1]
    
    # Bottom - lines 204-205
    edges.append(([1, 0, -1], [-1, 0, -1], 'dash'))      # line 204: x=[1,-1], y=[0,0], z=[-1,-1]
    edges.append(([0, -1, -1], [0, 1, -1], 'dash'))      # line 205: x=[0,0], y=[-1,1], z=[-1,-1]
    
    # Front - lines 208-209
    edges.append(([1, 0, -1], [1, 0, 1], 'solid'))      # line 208: x=[1,1], y=[0,0], z=[-1,1]
    edges.append(([1, -1, 0], [1, 1, 0], 'solid'))      # line 209: x=[1,1], y=[-1,1], z=[0,0]
    
    # Back - lines 212-213
    edges.append(([-1, 0, 1], [-1, 0, -1], 'dash'))      # line 212: x=[-1,-1], y=[0,0], z=[1,-1]
    edges.append(([-1, -1, 0], [-1, 1, 0], 'dash'))      # line 213: x=[-1,-1], y=[-1,1], z=[0,0]
    
    # Right - lines 216-217
    edges.append(([1, 1, 0], [-1, 1, 0], 'solid'))      # line 216: x=[1,-1], y=[1,1], z=[0,0]
    edges.append(([0, 1, 1], [0, 1, -1], 'solid'))      # line 217: x=[0,0], y=[1,1], z=[1,-1]
    
    # Left - lines 220-221
    edges.append(([0, -1, 1], [0, -1, -1], 'dash'))      # line 220: x=[0,0], y=[-1,-1], z=[1,-1]
    edges.append(([1, -1, 0], [-1, -1, 0], 'dash'))      # line 221: x=[1,-1], y=[-1,-1], z=[0,0]
    
    return edges


def _create_faces():
    """Create the three colored faces from the original visualization"""
    # Face 1: (15,5,18,2,16,6,17,1) - indices are 0-based, so subtract 1
    face1_indices = [14, 4, 17, 1, 15, 5, 16, 0]
    face1_vertices = [[D3Q19_DIRX[i], D3Q19_DIRY[i], D3Q19_DIRZ[i]] for i in face1_indices]
    
    # Face 2: (14,4,12,6,13,3,11,5)
    face2_indices = [13, 3, 11, 5, 12, 2, 10, 4]
    face2_vertices = [[D3Q19_DIRX[i], D3Q19_DIRY[i], D3Q19_DIRZ[i]] for i in face2_indices]
    
    # Face 3: (9,1,7,3,10,2,8,4)
    face3_indices = [8, 0, 6, 2, 9, 1, 7, 3]
    face3_vertices = [[D3Q19_DIRX[i], D3Q19_DIRY[i], D3Q19_DIRZ[i]] for i in face3_indices]
    
    return [
        (face1_vertices, 'lightgreen'),
        (face2_vertices, 'lightgoldenrodyellow'),
        (face3_vertices, 'lightblue')
    ]


def plot_d3q19_lattice(
    # Stencil configuration
    show_vectors: bool = True,
    vector_scale: float = 1.0,
    vector_width: float = 3.0,
    
    # Node appearance
    node_size: float = 10.0,
    node_colors: Optional[List[str]] = None,
    node_opacity: float = 0.8,
    node_style: str = 'circle',  # 'circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x'
    node_edge_color: str = 'black',
    node_edge_width: float = 1.0,
    origin_size: float = 15.0,
    origin_color: str = '#052020',
    origin_style: str = 'circle-open',  # 'circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x'
    
    # Vector styling
    vector_color: str = 'red',
    vector_opacity: float = 0.8,
    vector_linestyle: str = 'dashdot',  # 'solid', 'dash', 'dot', 'dashdot'
    show_vector_arrows: bool = False,
    arrow_head_size: float = 0.1,
    
    # Labels
    show_labels: bool = True,
    label_prefix: str = 'C',  # 'C' for C1, C2, etc. or '' for numbers
    label_font_size: int = 13,
    label_color: str = 'black',
    label_offset: float = 1.19,
    
    # Faces and edges
    show_faces: bool = False,
    face_opacity: float = 0.5,
    show_cube_edges: bool = True,
    cube_edge_color: str = 'black',
    cube_edge_width: float = 2.0,
    cube_edge_style: str = 'solid',  # 'solid' or 'dash'
    
    # Grid and background
    show_grid: bool = False,
    grid_color: str = 'gray',
    grid_opacity: float = 0.3,
    background_color: str = 'white',
    
    # Coordinate system
    show_axes: bool = False,
    show_axis_labels: bool = False,
    show_origin_marker: bool = True,
    
    # View controls
    camera_elevation: float = 9.0,
    camera_azimuth: float = 16.0,
    camera_zoom: float = 1.0,
    
    # Layout
    width: int = 800,
    height: int = 800,
    title: str = "D3Q19 Lattice Stencil",
    
) -> go.Figure:
    """Create interactive D3Q19 lattice visualization"""
    if node_colors is None:
        node_colors = DEFAULT_LATTICE_COLORS.copy()
    
    # Validate symbol values (Plotly Scatter3d only supports specific symbols)
    valid_symbols = ['circle', 'circle-open', 'square', 'square-open', 'diamond', 'diamond-open', 'cross', 'x']
    if node_style not in valid_symbols:
        node_style = 'circle'
    if origin_style not in valid_symbols:
        origin_style = 'circle'
    
    # Create figure
    fig = go.Figure()
    
    # Plot origin point
    if show_origin_marker:
        fig.add_trace(go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            mode='markers',
            marker=dict(
                size=origin_size,
                color=origin_color,
                symbol=origin_style,
                line=dict(color='black', width=1),
                opacity=1.0
            ),
            name='Origin',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Plot lattice points
    for i in range(19):
        fig.add_trace(go.Scatter3d(
            x=[D3Q19_DIRX[i]],
            y=[D3Q19_DIRY[i]],
            z=[D3Q19_DIRZ[i]],
            mode='markers',
            marker=dict(
                size=node_size,
                color=node_colors[i] if i < len(node_colors) else 'blue',
                symbol=node_style,
                line=dict(color=node_edge_color, width=node_edge_width),
                opacity=node_opacity
            ),
            name=f'C{i+1}',
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add labels
        if show_labels:
            label_text = f'{label_prefix}{i+1}' if label_prefix else str(i+1)
            fig.add_trace(go.Scatter3d(
                x=[D3Q19_DIRX[i] * label_offset],
                y=[D3Q19_DIRY[i] * label_offset],
                z=[D3Q19_DIRZ[i] * label_offset],
                mode='text',
                text=[label_text],
                textfont=dict(size=label_font_size, color=label_color),
                name=f'Label {i+1}',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add vectors from origin to lattice points
    if show_vectors:
        for i in range(19):
            # Scale the direction vector
            end_x = D3Q19_DIRX[i] * vector_scale
            end_y = D3Q19_DIRY[i] * vector_scale
            end_z = D3Q19_DIRZ[i] * vector_scale
            
            # Convert linestyle
            dash_map = {
                'solid': None,
                'dash': 'dash',
                'dot': 'dot',
                'dashdot': 'dashdot'
            }
            dash = dash_map.get(vector_linestyle, 'dashdot')
            
            fig.add_trace(go.Scatter3d(
                x=[0, end_x],
                y=[0, end_y],
                z=[0, end_z],
                mode='lines',
                line=dict(
                    color=vector_color,
                    width=vector_width,
                    dash=dash
                ),
                opacity=vector_opacity,
                name=f'Vector {i+1}',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Add colored faces - using proper triangulation
    if show_faces:
        faces = _create_faces()
        for face_vertices, face_color in faces:
            if len(face_vertices) >= 3:
                # Extract coordinates
                x_coords = [v[0] for v in face_vertices]
                y_coords = [v[1] for v in face_vertices]
                z_coords = [v[2] for v in face_vertices]
                
                # Create proper triangulation using fan method from first vertex
                n_vertices = len(face_vertices)
                i_indices = []
                j_indices = []
                k_indices = []
                
                # Fan triangulation: connect first vertex to all other pairs
                for tri_idx in range(1, n_vertices - 1):
                    i_indices.append(0)
                    j_indices.append(tri_idx)
                    k_indices.append(tri_idx + 1)
                
                # Add the closing triangle if needed
                if n_vertices > 3:
                    i_indices.append(0)
                    j_indices.append(n_vertices - 1)
                    k_indices.append(1)
                
                fig.add_trace(go.Mesh3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    i=i_indices,
                    j=j_indices,
                    k=k_indices,
                    color=face_color,
                    opacity=face_opacity,
                    flatshading=True,
                    showlegend=False,
                    hoverinfo='skip',
                    lighting=dict(
                        ambient=1.0,
                        diffuse=0.0,
                        specular=0.0,
                        roughness=1.0,
                        fresnel=0.0
                    ),
                    lightposition=dict(x=0, y=0, z=0)
                ))
    
    # Add cube edges - separate traces for solid and dashed lines to match original
    if show_cube_edges:
        edges = _create_cube_edges()
        
        # Separate solid and dashed edges
        x_solid = []
        y_solid = []
        z_solid = []
        x_dash = []
        y_dash = []
        z_dash = []
        
        for edge in edges:
            # Each edge is a tuple: (start, end, style)
            start, end, style = edge
            
            if style == 'solid':
                x_solid.extend([start[0], end[0], None])
                y_solid.extend([start[1], end[1], None])
                z_solid.extend([start[2], end[2], None])
            else:  # dash - create multiple short segments for tighter dash pattern
                dash_segments = _create_dashed_segments(start, end, dash_length=0.08, gap_length=0.04)
                for seg_start, seg_end in dash_segments:
                    x_dash.extend([seg_start[0], seg_end[0], None])
                    y_dash.extend([seg_start[1], seg_end[1], None])
                    z_dash.extend([seg_start[2], seg_end[2], None])
        
        # Add solid edges
        if x_solid:
            fig.add_trace(go.Scatter3d(
                x=x_solid,
                y=y_solid,
                z=z_solid,
                mode='lines',
                line=dict(
                    color=cube_edge_color,
                    width=cube_edge_width,
                    dash=None  # solid
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Cube Edges Solid'
            ))
        
        # Add dashed edges (as multiple solid segments to create tight dash pattern)
        if x_dash:
            fig.add_trace(go.Scatter3d(
                x=x_dash,
                y=y_dash,
                z=z_dash,
                mode='lines',
                line=dict(
                    color=cube_edge_color,
                    width=cube_edge_width,
                    dash=None  # solid segments to create dash pattern
                ),
                showlegend=False,
                hoverinfo='skip',
                name='Cube Edges Dashed'
            ))
    
    # Add coordinate axes if requested
    if show_axes:
        axis_length = 1.2
        # X axis (red)
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=5, color='red'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Y axis (green)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines+markers',
            line=dict(color='green', width=3),
            marker=dict(size=5, color='green'),
            showlegend=False,
            hoverinfo='skip'
        ))
        # Z axis (blue)
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=5, color='blue'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        if show_axis_labels:
            fig.add_trace(go.Scatter3d(
                x=[axis_length * 1.1], y=[0], z=[0],
                mode='text',
                text=['x'],
                textfont=dict(size=14, color='red'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[axis_length * 1.1], z=[0],
                mode='text',
                text=['y'],
                textfont=dict(size=14, color='green'),
                showlegend=False,
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[axis_length * 1.1],
                mode='text',
                text=['z'],
                textfont=dict(size=14, color='blue'),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout - disable all interactive features that cause visual artifacts
    fig.update_layout(
        title=title,
            scene=dict(
                xaxis=dict(
                    range=[-1.5, 1.5],
                    showgrid=show_grid,
                    gridcolor=grid_color,
                    gridwidth=1,
                    backgroundcolor=background_color,
                    showbackground=True,
                    showticklabels=show_axis_labels,
                    title='X' if show_axis_labels else '',
                    showspikes=False,
                    spikecolor='rgba(0,0,0,0)',
                    spikethickness=0,
                    visible=True,
                    zeroline=False
                ),
                yaxis=dict(
                    range=[-1.5, 1.5],
                    showgrid=show_grid,
                    gridcolor=grid_color,
                    gridwidth=1,
                    backgroundcolor=background_color,
                    showbackground=True,
                    showticklabels=show_axis_labels,
                    title='Y' if show_axis_labels else '',
                    showspikes=False,
                    spikecolor='rgba(0,0,0,0)',
                    spikethickness=0,
                    visible=True,
                    zeroline=False
                ),
                zaxis=dict(
                    range=[-1.5, 1.5],
                    showgrid=show_grid,
                    gridcolor=grid_color,
                    gridwidth=1,
                    backgroundcolor=background_color,
                    showbackground=True,
                    showticklabels=show_axis_labels,
                    title='Z' if show_axis_labels else '',
                    showspikes=False,
                    spikecolor='rgba(0,0,0,0)',
                    spikethickness=0,
                    visible=True,
                    zeroline=False
                ),
                aspectmode='cube',
                camera=dict(
                    eye=dict(
                        x=2.0 * camera_zoom * np.cos(np.radians(camera_azimuth)) * np.cos(np.radians(camera_elevation)),
                        y=2.0 * camera_zoom * np.sin(np.radians(camera_azimuth)) * np.cos(np.radians(camera_elevation)),
                        z=2.0 * camera_zoom * np.sin(np.radians(camera_elevation))
                    ),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1),
                    projection=dict(type='perspective')
                ),
                dragmode='orbit',
                hovermode=False,
                bgcolor=background_color
            ),
        width=width,
        height=height,
        paper_bgcolor=background_color,
        plot_bgcolor=background_color,
        margin=dict(l=0, r=0, t=50, b=0),
        hovermode=False,
        # Disable all interactive features
        clickmode='none',
        # Make sure nothing is selectable
        uirevision='static'  # Prevent UI state changes
    )
    
    # Force all traces to be non-interactive
    for trace in fig.data:
        if hasattr(trace, 'hoverinfo'):
            trace.hoverinfo = 'skip'
        if hasattr(trace, 'hovertext'):
            trace.hovertext = None
    
    return fig

"""
Shared 3D Volume Viewer visualization — single source of truth for volume/slice/isosurface plots.

Used by:
  1. Manual page (pages/VolumeViewer3D/views.py) — can optionally import build_3d_volume_figure
  2. AI agents (plot_volume_3d tool)

Pure Python plotting logic — no Streamlit dependency.
Supports: volume rendering, orthogonal slices, isosurface, surface (6 faces), axes, camera, style.
"""

from typing import Optional

import numpy as np
import plotly.graph_objects as go

CAMERA_PRESETS = {
    "Isometric": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
    "XY": dict(eye=dict(x=0, y=0, z=2.5), up=dict(x=0, y=1, z=0)),
    "XZ": dict(eye=dict(x=0, y=2.5, z=0), up=dict(x=0, y=0, z=1)),
    "YZ": dict(eye=dict(x=2.5, y=0, z=0), up=dict(x=0, y=1, z=0)),
    "Custom": dict(eye=dict(x=1.4, y=1.4, z=1.2)),
}


def create_slice_surface(x_coords, y_coords, z_coords, field_slice, vmin, vmax, cmap, opacity):
    """Create a surface trace for a slice plane. Matches ParaView coordinate system."""
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
        connectgaps=False,
    )


def build_3d_volume_figure(
    xg: np.ndarray,
    yg: np.ndarray,
    zg: np.ndarray,
    field: np.ndarray,
    vmin: float,
    vmax: float,
    cmax: float,
    cmap: str,
    field_type: str,
    *,
    show_volume: bool = False,
    vol_opacity: float = 0.15,
    vol_surface_count: int = 20,
    show_iso: bool = False,
    iso_value: float = 0.0,
    iso_opacity: float = 0.4,
    show_slices: bool = True,
    slice_x: int = 0,
    slice_y: int = 0,
    slice_z: int = 0,
    slice_opacity: float = 0.9,
    show_surface: bool = False,
    surface_opacity: float = 0.8,
    show_axes: bool = False,
    show_axis_labels: bool = False,
    camera_preset: str = "Isometric",
    style_updates: Optional[dict] = None,
) -> go.Figure:
    """
    Build 3D figure with volume, isosurface, surface (6 faces), and/or orthogonal slices.

    xg, yg, zg: Coordinate grids (from np.mgrid)
    field: 3D scalar field (downsampled)
    vmin, vmax, cmax: Value range for colormap
    cmap: Plotly colorscale name
    field_type: Label for colorbar
    style_updates: plot_bgcolor, paper_bgcolor, font_family, height, figure_width, figure_height, etc.
    """
    nx_d, ny_d, nz_d = field.shape
    ps = style_updates or {}
    scene_bgcolor = ps.get("plot_bgcolor", "#FFFFFF")
    paper_bgcolor = ps.get("paper_bgcolor", "#FFFFFF")
    grid_color = ps.get("grid_color", "#B0B0B0")
    axis_title_size = ps.get("axis_title_size", 14)
    font_color = ps.get("font_color") or ("#d4d4d4" if "dark" in str(ps.get("template", "")).lower() else "#000000")
    height = ps.get("height") or ps.get("figure_height") or 600
    camera = CAMERA_PRESETS.get(camera_preset, CAMERA_PRESETS["Isometric"])

    fig = go.Figure()

    if show_volume:
        isomin_val = vmin + (cmax - vmin) * 0.1 if cmax > vmin else vmin
        fig.add_trace(
            go.Volume(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=field.flatten(),
                isomin=isomin_val,
                isomax=cmax,
                opacity=vol_opacity,
                surface_count=vol_surface_count,
                colorscale=cmap,
                caps=dict(x_show=False, y_show=False, z_show=False),
                name="Volume",
                showscale=True,
                colorbar=dict(title=dict(text=field_type, font=dict(size=14)), len=0.75, y=0.5, thickness=20),
            )
        )

    if show_iso:
        fig.add_trace(
            go.Isosurface(
                x=xg.flatten(),
                y=yg.flatten(),
                z=zg.flatten(),
                value=field.flatten(),
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
        fig.add_trace(
            create_slice_surface(
                x_coords, y_coords, np.zeros((nx_d, ny_d)),
                field[:, :, 0], vmin, cmax, cmap, surface_opacity
            )
        )
        fig.add_trace(
            create_slice_surface(
                x_coords, y_coords, np.full((nx_d, ny_d), nz_d - 1),
                field[:, :, nz_d - 1], vmin, cmax, cmap, surface_opacity
            )
        )
        y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                np.zeros((ny_d, nz_d)), y_coords, z_coords,
                field[0, :, :], vmin, cmax, cmap, surface_opacity
            )
        )
        fig.add_trace(
            create_slice_surface(
                np.full((ny_d, nz_d), nx_d - 1), y_coords, z_coords,
                field[nx_d - 1, :, :], vmin, cmax, cmap, surface_opacity
            )
        )
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_coords, np.zeros((nx_d, nz_d)), z_coords,
                field[:, 0, :], vmin, cmax, cmap, surface_opacity
            )
        )
        fig.add_trace(
            create_slice_surface(
                x_coords, np.full((nx_d, nz_d), ny_d - 1), z_coords,
                field[:, ny_d - 1, :], vmin, cmax, cmap, surface_opacity
            )
        )

    if show_slices:
        z_plane = np.full((nx_d, ny_d), slice_z)
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, ny_d))
        y_coords = np.ones((nx_d, 1)) * np.arange(ny_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_coords, y_coords, z_plane,
                field[:, :, slice_z], vmin, cmax, cmap, slice_opacity
            )
        )
        y_plane = np.full((nx_d, nz_d), slice_y)
        x_coords = np.arange(nx_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((nx_d, 1)) * np.arange(nz_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_coords, y_plane, z_coords,
                field[:, slice_y, :], vmin, cmax, cmap, slice_opacity
            )
        )
        x_plane = np.full((ny_d, nz_d), slice_x)
        y_coords = np.arange(ny_d)[:, None] * np.ones((1, nz_d))
        z_coords = np.ones((ny_d, 1)) * np.arange(nz_d)[None, :]
        fig.add_trace(
            create_slice_surface(
                x_plane, y_coords, z_coords,
                field[slice_x, :, :], vmin, cmax, cmap, slice_opacity
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
                    mode="text", text=["x"], textfont=dict(size=14, color="red"),
                    showlegend=False, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[axis_length * 1.1], z=[0],
                    mode="text", text=["y"], textfont=dict(size=14, color="green"),
                    showlegend=False, hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[0], y=[0], z=[axis_length * 1.1],
                    mode="text", text=["z"], textfont=dict(size=14, color="blue"),
                    showlegend=False, hoverinfo="skip",
                )
            )

    scene_dict = dict(
        xaxis_title="X" if show_axis_labels else "",
        yaxis_title="Y" if show_axis_labels else "",
        zaxis_title="Z" if show_axis_labels else "",
        aspectmode="data",
        camera=camera,
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
    )

    layout_kw = dict(
        scene=scene_dict,
        height=height,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor=paper_bgcolor,
    )
    if ps.get("show_plot_title") and ps.get("plot_title"):
        layout_kw["title"] = dict(
            text=ps["plot_title"],
            font=dict(family=ps.get("font_family", "Arial"), size=ps.get("title_size", 16), color=font_color),
        )
        layout_kw["margin"] = dict(l=0, r=0, t=50, b=0)
    fig.update_layout(**layout_kw)
    return fig

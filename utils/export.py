"""
Export utilities for plots and data
"""

import plotly.graph_objects as go
from pathlib import Path
from typing import Optional


def export_plot(fig: go.Figure, filepath: str, format: str = 'png', width: int = 800, height: int = 600, dpi: int = 300):
    """
    Export Plotly figure to file
    
    Args:
        fig: Plotly figure object
        filepath: Output file path
        format: Export format ('png', 'pdf', 'svg', 'html')
        width: Image width in pixels
        height: Image height in pixels
        dpi: Resolution for raster formats
    """
    try:
        if format == 'png':
            fig.write_image(filepath, width=width, height=height, scale=dpi/100)
        elif format == 'pdf':
            fig.write_image(filepath, width=width, height=height)
        elif format == 'svg':
            fig.write_image(filepath, width=width, height=height)
        elif format == 'html':
            fig.write_html(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise ValueError(f"Error exporting plot: {e}")


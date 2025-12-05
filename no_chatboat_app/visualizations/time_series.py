"""
Time series visualization
"""

import numpy as np
import plotly.graph_objects as go
import pandas as pd
from typing import List, Optional


def plot_time_series(
    time: np.ndarray,
    values: np.ndarray,
    label: str = "Value",
    color: str = "blue",
    ylabel: str = "Value"
) -> go.Figure:
    """
    Plot time series data
    
    Args:
        time: Time values
        values: Data values
        label: Plot label
        color: Line color
        ylabel: Y-axis label
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=values,
        mode='lines',
        name=label,
        line=dict(color=color, width=2),
        hovertemplate='Time: %{x:.2e}<br>Value: %{y:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis=dict(title='Time', showgrid=True),
        yaxis=dict(title=ylabel, showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=400
    )
    
    return fig


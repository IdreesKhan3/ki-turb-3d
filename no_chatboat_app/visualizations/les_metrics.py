"""
LES metrics visualization
C_S statistics and effective viscosity
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional


def plot_les_metrics(
    time: np.ndarray,
    C_S: Optional[np.ndarray] = None,
    nu_t: Optional[np.ndarray] = None,
    label: str = "LES Metrics"
) -> go.Figure:
    """
    Plot LES metrics: C_S and effective viscosity
    
    Args:
        time: Time values
        C_S: Smagorinsky constant values (optional)
        nu_t: Effective viscosity values (optional)
        label: Plot label
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if C_S is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=C_S,
            mode='lines',
            name='C_S',
            line=dict(color='blue', width=2),
            yaxis='y',
            hovertemplate='Time: %{x:.2e}<br>C_S: %{y:.4f}<extra></extra>'
        ))
    
    if nu_t is not None:
        fig.add_trace(go.Scatter(
            x=time,
            y=nu_t,
            mode='lines',
            name='ν_t',
            line=dict(color='green', width=2),
            yaxis='y2' if C_S is not None else 'y',
            hovertemplate='Time: %{x:.2e}<br>ν_t: %{y:.4f}<extra></extra>'
        ))
    
    layout = dict(
        title=label,
        xaxis=dict(title='Time', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=500
    )
    
    if C_S is not None and nu_t is not None:
        layout['yaxis'] = dict(title='C_S', showgrid=True, side='left')
        layout['yaxis2'] = dict(title='ν_t', showgrid=True, side='right', overlaying='y')
    elif C_S is not None:
        layout['yaxis'] = dict(title='C_S', showgrid=True)
    elif nu_t is not None:
        layout['yaxis'] = dict(title='ν_t', showgrid=True)
    
    fig.update_layout(**layout)
    
    return fig


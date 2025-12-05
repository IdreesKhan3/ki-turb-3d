"""
Isotropy visualization
Spectral and real-space isotropy plots
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple, Optional


def plot_spectral_isotropy(
    k: np.ndarray,
    IC: np.ndarray,
    IC_deriv: Optional[np.ndarray] = None,
    label: str = "IC(k)"
) -> go.Figure:
    """
    Plot spectral isotropy coefficient IC(k) = E₂₂(k) / E₁₁(k)
    
    Args:
        k: Wavenumber values
        IC: Isotropy coefficient values
        IC_deriv: Derivative-based isotropy coefficient (optional)
        label: Plot label
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=k,
        y=IC,
        mode='lines',
        name=label,
        line=dict(color='blue', width=2),
        hovertemplate='k: %{x:.2e}<br>IC: %{y:.4f}<extra></extra>'
    ))
    
    if IC_deriv is not None:
        fig.add_trace(go.Scatter(
            x=k,
            y=IC_deriv,
            mode='lines',
            name='IC_deriv',
            line=dict(color='green', width=2, dash='dash'),
            hovertemplate='k: %{x:.2e}<br>IC_deriv: %{y:.4f}<extra></extra>'
        ))
    
    # Reference line at 1.0
    fig.add_hline(y=1.0, line_dash="dot", line_color="red", annotation_text="Isotropic (IC=1.0)")
    
    fig.update_layout(
        title="Spectral Isotropy Coefficient",
        xaxis=dict(title='k', showgrid=True),
        yaxis=dict(title='IC(k)', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=500
    )
    
    return fig


def plot_real_space_isotropy(
    time: np.ndarray,
    frac_x: np.ndarray,
    frac_y: np.ndarray,
    frac_z: np.ndarray
) -> go.Figure:
    """
    Plot real-space isotropy (fractional energies)
    
    Args:
        time: Normalized time values (t/t_0)
        frac_x: Fractional energy in x direction
        frac_y: Fractional energy in y direction
        frac_z: Fractional energy in z direction
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time,
        y=frac_x,
        mode='lines+markers',
        name='E_x/E_tot',
        line=dict(color='blue', width=2),
        marker=dict(size=2, opacity=0.4)
    ))
    
    fig.add_trace(go.Scatter(
        x=time,
        y=frac_y,
        mode='lines+markers',
        name='E_y/E_tot',
        line=dict(color='green', width=2),
        marker=dict(size=2, opacity=0.4)
    ))
    
    fig.add_trace(go.Scatter(
        x=time,
        y=frac_z,
        mode='lines+markers',
        name='E_z/E_tot',
        line=dict(color='orange', width=2),
        marker=dict(size=2, opacity=0.4)
    ))
    
    # Reference line at 1/3
    fig.add_hline(y=1/3, line_dash="dot", line_color="red", annotation_text="Isotropic (1/3)")
    
    fig.update_layout(
        title="Real-Space Isotropy (Fractional Energies)",
        xaxis=dict(title='t/t_0', showgrid=True),
        yaxis=dict(title='Energy Fraction', showgrid=True, range=[0.25, 0.45]),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=500
    )
    
    return fig


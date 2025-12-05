"""
Energy spectra visualization
Replicates plot_spectra.ipynb functionality with Plotly
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
from utils.data_processor import compute_energy_variance


def plot_energy_spectrum(
    k_vals: np.ndarray,
    E_avg: np.ndarray,
    E_std: Optional[np.ndarray] = None,
    label: str = "E(k)",
    color: str = "blue",
    linestyle: str = "-",
    show_error_bars: bool = True
) -> go.Figure:
    """
    Plot time-averaged energy spectrum with error bars
    
    Replicates plot_spectra.ipynb lines 96-98
    
    Args:
        k_vals: Wavenumber values
        E_avg: Averaged energy spectrum
        E_std: Standard deviation (optional, for error bars)
        label: Plot label
        color: Line color
        linestyle: Line style
        show_error_bars: Whether to show error bars
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Main curve
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=E_avg,
        mode='lines',
        name=label,
        line=dict(color=color, width=2),
        hovertemplate='k: %{x:.2e}<br>E(k): %{y:.2e}<extra></extra>'
    ))
    
    # Error bars (fill_between equivalent)
    if E_std is not None and show_error_bars:
        fig.add_trace(go.Scatter(
            x=k_vals,
            y=E_avg - E_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=k_vals,
            y=E_avg + E_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba({_color_to_rgb(color)}, 0.25)',
            name=f'{label} ± σ',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Kolmogorov reference line (optional)
    # k_start, k_end = 4, 20
    # k_fit = k_vals[(k_vals >= k_start) & (k_vals <= k_end)]
    # if len(k_fit) > 1:
    #     ref = k_fit**(-5/3)
    #     ref *= E_avg[2] / ref[1] if len(E_avg) > 2 else 1.0
    #     fig.add_trace(go.Scatter(
    #         x=k_fit,
    #         y=ref,
    #         mode='lines',
    #         name=r'$k^{-5/3}$',
    #         line=dict(color='gray', dash='dot', width=2)
    #     ))
    
    fig.update_layout(
        title=dict(text="Energy Spectrum", font=dict(size=16)),
        xaxis=dict(title='k', type='log', showgrid=True),
        yaxis=dict(title='E(k)', type='log', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig


def plot_time_evolution_spectrum(
    spectrum_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    selected_iteration: Optional[int] = None,
    show_all: bool = True,
    every_nth: int = 1
) -> go.Figure:
    """
    Plot time evolution of energy spectra (all iterations)
    
    Args:
        spectrum_data: Dictionary mapping iteration -> (k, E)
        selected_iteration: Iteration to highlight (optional)
        show_all: Show all iterations (with transparency)
        every_nth: Show every Nth iteration (to reduce clutter)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    iterations = sorted(spectrum_data.keys())
    n_iterations = len(iterations)
    
    # Color scale for time progression
    colors = ['blue' if i < n_iterations/3 else 'green' if i < 2*n_iterations/3 else 'red' 
              for i in range(n_iterations)]
    
    for idx, iter_num in enumerate(iterations[::every_nth]):
        if iter_num == selected_iteration:
            # Highlight selected iteration
            k, E = spectrum_data[iter_num]
            fig.add_trace(go.Scatter(
                x=k,
                y=E,
                mode='lines',
                name=f'Iter {iter_num}',
                line=dict(color='red', width=3),
                opacity=1.0
            ))
        elif show_all:
            k, E = spectrum_data[iter_num]
            fig.add_trace(go.Scatter(
                x=k,
                y=E,
                mode='lines',
                name=f'Iter {iter_num}',
                line=dict(color=colors[idx], width=1),
                opacity=0.3,
                showlegend=False
            ))
    
    fig.update_layout(
        title="Time Evolution of Energy Spectrum",
        xaxis=dict(title='k', type='log', showgrid=True),
        yaxis=dict(title='E(k)', type='log', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig


def plot_normalized_spectrum_with_pope(
    k_eta: np.ndarray,
    E_norm: np.ndarray,
    E_pope_norm: Optional[np.ndarray] = None,
    E_std: Optional[np.ndarray] = None,
    label: str = "E(k)",
    color: str = "blue",
    show_error_bars: bool = True
) -> go.Figure:
    """
    Plot normalized energy spectrum with Pope model overlay
    
    Uses normalized axes: k*η (wavenumber × Kolmogorov scale) vs E(k)/(ε^(2/3)*k^(-5/3))
    This is the compensated spectrum format for comparing with Pope model.
    
    Args:
        k_eta: Normalized wavenumber (k * η)
        E_norm: Normalized energy spectrum
        E_pope_norm: Pope model normalized spectrum (optional overlay)
        E_std: Standard deviation for error bars (optional)
        label: Plot label
        color: Line color
        show_error_bars: Whether to show error bars
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    # Main normalized spectrum curve
    fig.add_trace(go.Scatter(
        x=k_eta,
        y=E_norm,
        mode='lines',
        name=label,
        line=dict(color=color, width=2),
        hovertemplate='k·η: %{x:.2e}<br>E/(ε²ᐟ³k⁻⁵ᐟ³): %{y:.2e}<extra></extra>'
    ))
    
    # Pope model overlay (if provided)
    if E_pope_norm is not None:
        fig.add_trace(go.Scatter(
            x=k_eta,
            y=E_pope_norm,
            mode='lines',
            name='Pope Model',
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='k·η: %{x:.2e}<br>Pope: %{y:.2e}<extra></extra>'
        ))
    
    # Error bars (fill_between equivalent)
    if E_std is not None and show_error_bars:
        fig.add_trace(go.Scatter(
            x=k_eta,
            y=E_norm - E_std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=k_eta,
            y=E_norm + E_std,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=f'rgba({_color_to_rgb(color)}, 0.25)',
            name=f'{label} ± σ',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title=dict(text="Normalized Energy Spectrum with Pope Model", font=dict(size=16)),
        xaxis=dict(title='k·η', type='log', showgrid=True),
        yaxis=dict(title='E(k)/(ε²ᐟ³k⁻⁵ᐟ³)', type='log', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig


def _color_to_rgb(color: str) -> str:
    """Convert color name to RGB string for rgba"""
    color_map = {
        'blue': '31, 119, 180',
        'green': '44, 160, 44',
        'red': '214, 39, 40',
        'purple': '148, 103, 189',
        'orange': '255, 127, 14',
    }
    return color_map.get(color, '128, 128, 128')


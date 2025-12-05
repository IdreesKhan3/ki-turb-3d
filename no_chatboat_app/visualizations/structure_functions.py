"""
Structure functions visualization
ESS analysis and plotting
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


def plot_structure_functions(
    r: np.ndarray,
    S_p: Dict[int, np.ndarray],
    orders: List[int],
    labels: Optional[Dict[int, str]] = None
) -> go.Figure:
    """
    Plot structure functions S_p(r) for multiple orders
    
    Args:
        r: Separation distance values
        S_p: Dictionary mapping order p -> S_p(r) values
        orders: List of orders to plot
        labels: Optional dictionary of labels for each order
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'gray']
    
    for idx, p in enumerate(orders):
        if p in S_p:
            label = labels.get(p, f'S_{p}(r)') if labels else f'S_{p}(r)'
            fig.add_trace(go.Scatter(
                x=r,
                y=S_p[p],
                mode='lines',
                name=label,
                line=dict(color=colors[idx % len(colors)], width=2),
                hovertemplate=f'p={p}<br>r: %{{x:.2e}}<br>S_p: %{{y:.2e}}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Structure Functions",
        xaxis=dict(title='r', type='log', showgrid=True),
        yaxis=dict(title='S_p(r)', type='log', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig


def plot_ess(
    S_3: np.ndarray,
    S_p_dict: Dict[int, np.ndarray],
    orders: List[int],
    simulation_labels: Optional[List[str]] = None
) -> go.Figure:
    """
    Plot Extended Self-Similarity (ESS): S_p(r) vs S_3(r)
    
    Replicates ESS_multi_simu.py functionality
    
    Args:
        S_3: S_3(r) values (reference)
        S_p_dict: Dictionary mapping order p -> S_p(r) values
        orders: List of orders to plot
        simulation_labels: Optional labels for different simulations
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    colors = ['blue', 'green', 'purple', 'orange', 'brown', 'gray']
    markers = ['circle', 'square', 'triangle-up', 'diamond', 'cross', 'x']
    
    for idx, p in enumerate(orders):
        if p in S_p_dict:
            fig.add_trace(go.Scatter(
                x=S_3,
                y=S_p_dict[p],
                mode='lines+markers',
                name=f'p={p}',
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(symbol=markers[idx % len(markers)], size=4),
                hovertemplate=f'p={p}<br>S_3: %{{x:.2e}}<br>S_p: %{{y:.2e}}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Extended Self-Similarity (ESS)",
        xaxis=dict(title='S_3(r)', type='log', showgrid=True),
        yaxis=dict(title='S_p(r)', type='log', showgrid=True),
        hovermode='closest',
        template='plotly_white',
        width=800,
        height=600
    )
    
    return fig


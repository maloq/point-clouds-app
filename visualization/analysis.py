"""
Analysis visualization functions.
"""

import plotly.graph_objects as go
import numpy as np
import torch
from typing import List, Dict, Callable, Tuple
import sys
sys.path.append('..')
from utils.rotation import interpolate_rotations, geodesic_distance
from metrics.chamfer import chamfer_distance

def plot_geodesic_path(
    cloud: torch.Tensor,
    target: torch.Tensor,
    rot_start: torch.Tensor,
    rot_end: torch.Tensor,
    steps: int = 20,
    metrics: Dict[str, Callable] = None
) -> go.Figure:
    """
    Plot metrics along geodesic path between two rotations.
    """
    if metrics is None:
        metrics = {'Chamfer': chamfer_distance}
        
    # Interpolate rotations
    rotations = interpolate_rotations(rot_start, rot_end, steps)
    
    # Calculate metrics
    results = {name: [] for name in metrics}
    angles = []
    
    # Target is assumed to be cloud rotated by rot_end (or just a target cloud)
    # If target is fixed, we rotate cloud by interpolated rotations and compare
    
    for i in range(steps):
        R = rotations[i]
        # Angle from start
        angle = geodesic_distance(rot_start, R).item()
        angles.append(angle)
        
        # Rotate cloud
        rotated_cloud = cloud @ R.T
        
        # Compute metrics
        for name, func in metrics.items():
            val = func(rotated_cloud, target)
            if isinstance(val, tuple): val = val[0]
            results[name].append(val.item())
            
    # Plot
    fig = go.Figure()
    
    for name, values in results.items():
        fig.add_trace(go.Scatter(
            x=angles,
            y=values,
            mode='lines+markers',
            name=name
        ))
        
    fig.update_layout(
        title="Metrics along Geodesic Path",
        xaxis_title="Rotation Angle (rad)",
        yaxis_title="Metric Value",
        hovermode="x unified"
    )
    
    return fig

def plot_loss_landscape(
    center_loss: float,
    grid_losses: np.ndarray,
    extent: float = 1.0,
    title: str = "Loss Landscape"
) -> go.Figure:
    """
    Plot 2D loss landscape.
    """
    # grid_losses is [N, N]
    N = grid_losses.shape[0]
    x = np.linspace(-extent, extent, N)
    y = np.linspace(-extent, extent, N)
    
    fig = go.Figure(data=[go.Surface(
        z=grid_losses,
        x=x,
        y=y,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Direction 1",
            yaxis_title="Direction 2",
            zaxis_title="Loss"
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

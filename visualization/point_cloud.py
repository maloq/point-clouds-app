"""
Plotly visualization functions for point clouds.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
from typing import Optional, Union, List, Tuple

def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def render_cloud(
    cloud: Union[torch.Tensor, np.ndarray],
    color: str = 'blue',
    title: str = 'Point Cloud',
    size: int = 2,
    opacity: float = 0.8
) -> go.Figure:
    """
    Render a single point cloud.
    """
    cloud = to_numpy(cloud)
    if cloud.ndim == 3:
        cloud = cloud[0]  # Take first in batch
        
    fig = go.Figure(data=[go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=size * 2.5,
            color=color,
            opacity=opacity,
            line=dict(color='black', width=1)
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def render_overlay(
    cloud1: Union[torch.Tensor, np.ndarray],
    cloud2: Union[torch.Tensor, np.ndarray],
    color1: str = 'blue',
    color2: str = 'red',
    title: str = 'Overlay',
    size: int = 2,
    opacity: float = 0.6,
    lines: Optional[Union[torch.Tensor, np.ndarray]] = None,
    label1: str = 'Source (Ref)',
    label2: str = 'Target (Deformed)',
    axis_range: Optional[Tuple[float, float]] = None,
    height: int = 800
) -> go.Figure:
    """
    Render two point clouds overlaid.
    Optionally draw lines between matched points.
    """
    cloud1 = to_numpy(cloud1)
    cloud2 = to_numpy(cloud2)
    
    if cloud1.ndim == 3: cloud1 = cloud1[0]
    if cloud2.ndim == 3: cloud2 = cloud2[0]
    
    data = []
    
    # Cloud 1 (Reference) - Larger, more transparent
    data.append(go.Scatter3d(
        x=cloud1[:, 0],
        y=cloud1[:, 1],
        z=cloud1[:, 2],
        mode='markers',
        name=label1,
        marker=dict(
            size=size * 3.5, 
            color=color1, 
            opacity=0.3,
            line=dict(color='black', width=1)
        )
    ))
    
    # Cloud 2 (Deformed) - Smaller, more solid
    data.append(go.Scatter3d(
        x=cloud2[:, 0],
        y=cloud2[:, 1],
        z=cloud2[:, 2],
        mode='markers',
        name=label2,
        marker=dict(
            size=size * 2.5, 
            color=color2, 
            opacity=0.9,
            line=dict(color='black', width=1)
        )
    ))
    
    # Lines
    if lines is not None:
        # lines is expected to be [N, 2] indices or similar?
        # Or just assume 1-to-1 correspondence if lines=True?
        # Let's assume lines is a list of pairs or just True for 1-to-1
        
        # If lines is True (or truthy but not array), draw 1-to-1 lines
        # But drawing N lines is heavy for Plotly.
        # Let's draw a subset or use a single trace with None separators
        
        x_lines = []
        y_lines = []
        z_lines = []
        
        # Limit to 100 lines for performance
        n_lines = min(len(cloud1), 100)
        indices = np.random.choice(len(cloud1), n_lines, replace=False)
        
        for i in indices:
            x_lines.extend([cloud1[i, 0], cloud2[i, 0], None])
            y_lines.extend([cloud1[i, 1], cloud2[i, 1], None])
            z_lines.extend([cloud1[i, 2], cloud2[i, 2], None])
            
        data.append(go.Scatter3d(
            x=x_lines,
            y=y_lines,
            z=z_lines,
            mode='lines',
            name='Correspondence',
            line=dict(color='gray', width=1, dash='solid')
        ))
        
    fig = go.Figure(data=data)
    
    layout_dict = dict(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=height
    )
    
    if axis_range is not None:
        r_min, r_max = axis_range
        layout_dict['scene']['xaxis']['range'] = [r_min, r_max]
        layout_dict['scene']['yaxis']['range'] = [r_min, r_max]
        layout_dict['scene']['zaxis']['range'] = [r_min, r_max]
        # Force manual aspect mode to respect ranges
        layout_dict['scene']['aspectmode'] = 'manual'
        layout_dict['scene']['aspectratio'] = dict(x=1, y=1, z=1)
        
    fig.update_layout(**layout_dict)
    
    return fig

def render_heatmap(
    cloud: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    title: str = 'Heatmap',
    size: int = 2,
    colorscale: str = 'Viridis'
) -> go.Figure:
    """
    Render point cloud with color based on values.
    """
    cloud = to_numpy(cloud)
    values = to_numpy(values)
    
    if cloud.ndim == 3: cloud = cloud[0]
    if values.ndim == 2: values = values[0]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=cloud[:, 0],
        y=cloud[:, 1],
        z=cloud[:, 2],
        mode='markers',
        marker=dict(
            size=size,
            color=values,
            colorscale=colorscale,
            colorbar=dict(title="Value"),
            opacity=0.8
        )
    )])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    return fig

def render_side_by_side(
    cloud1: Union[torch.Tensor, np.ndarray],
    cloud2: Union[torch.Tensor, np.ndarray],
    color1: str = 'blue',
    color2: str = 'red',
    title: str = 'Side-by-Side View',
    size: int = 2,
    opacity: float = 0.8,
    axis_range: Optional[Tuple[float, float]] = None,
    height: int = 800
) -> go.Figure:
    """
    Render two point clouds in side-by-side subplots.
    """
    cloud1 = to_numpy(cloud1)
    cloud2 = to_numpy(cloud2)
    
    if cloud1.ndim == 3: cloud1 = cloud1[0]
    if cloud2.ndim == 3: cloud2 = cloud2[0]
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Source (Reference)", "Target (Deformed)")
    )
    
    # Cloud 1
    fig.add_trace(go.Scatter3d(
        x=cloud1[:, 0], y=cloud1[:, 1], z=cloud1[:, 2],
        mode='markers',
        name='Cloud 1',
        marker=dict(
            size=size * 2.5, 
            color=color1, 
            opacity=opacity,
            line=dict(color='black', width=1)
        )
    ), row=1, col=1)
    
    # Cloud 2
    fig.add_trace(go.Scatter3d(
        x=cloud2[:, 0], y=cloud2[:, 1], z=cloud2[:, 2],
        mode='markers',
        name='Cloud 2',
        marker=dict(
            size=size * 2.5, 
            color=color2, 
            opacity=opacity,
            line=dict(color='black', width=1)
        )
    ), row=1, col=2)
    
    layout_dict = dict(
        title=title,
        margin=dict(l=0, r=0, b=0, t=30),
        height=height
    )
    
    if axis_range is not None:
        r_min, r_max = axis_range
        scene_dict = dict(
            xaxis=dict(range=[r_min, r_max], visible=False),
            yaxis=dict(range=[r_min, r_max], visible=False),
            zaxis=dict(range=[r_min, r_max], visible=False),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        )
        layout_dict['scene'] = scene_dict
        layout_dict['scene2'] = scene_dict
    else:
        scene_dict = dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data'
        )
        layout_dict['scene'] = scene_dict
        layout_dict['scene2'] = scene_dict
        
    fig.update_layout(**layout_dict)
    
    return fig

"""
Hausdorff distance implementations.
"""

import torch
from typing import Tuple


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between two point sets.
    
    Args:
        x: Point cloud [N, D]
        y: Point cloud [M, D]
    
    Returns:
        Distance matrix [N, M]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    yy = (y ** 2).sum(dim=-1, keepdim=True)
    xy = x @ y.T
    
    distances = xx + yy.T - 2 * xy
    return torch.clamp(distances, min=0)


def hausdorff_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    directed: bool = False
) -> torch.Tensor:
    """
    Compute Hausdorff distance between two point clouds.
    H(X, Y) = max(max_x min_y d(x, y), max_y min_x d(y, x))
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        directed: If True, compute only directed distance from x to y
    
    Returns:
        Hausdorff distance
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # Directed from x to y: max over x of min distance to y
    dist_x_to_y = dist_matrix.min(dim=1)[0].max()
    
    if directed:
        return dist_x_to_y
    
    # Directed from y to x
    dist_y_to_x = dist_matrix.min(dim=0)[0].max()
    
    # Symmetric Hausdorff distance
    return torch.max(dist_x_to_y, dist_y_to_x)


def hausdorff_distance_percentile(
    x: torch.Tensor,
    y: torch.Tensor,
    percentile: float = 95.0
) -> torch.Tensor:
    """
    Compute percentile Hausdorff distance (more robust to outliers).
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        percentile: Percentile to use instead of max
    
    Returns:
        Percentile Hausdorff distance
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # Min distances for each point
    min_dist_x = dist_matrix.min(dim=1)[0]
    min_dist_y = dist_matrix.min(dim=0)[0]
    
    # Percentile instead of max
    k_x = int(len(min_dist_x) * percentile / 100)
    k_y = int(len(min_dist_y) * percentile / 100)
    
    dist_x_to_y = min_dist_x.sort()[0][min(k_x, len(min_dist_x) - 1)]
    dist_y_to_x = min_dist_y.sort()[0][min(k_y, len(min_dist_y) - 1)]
    
    return torch.max(dist_x_to_y, dist_y_to_x)


def soft_hausdorff_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    temperature: float = 1.0,
    bidirectional: bool = True
) -> torch.Tensor:
    """
    Compute soft (differentiable) Hausdorff distance using softmax.
    Uses log-sum-exp as smooth approximation to max.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        temperature: Temperature parameter (lower = closer to true max)
        bidirectional: If True, compute symmetric distance
    
    Returns:
        Soft Hausdorff distance
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # Soft min using negative logsumexp
    # soft_min(x) ≈ -temperature * logsumexp(-x / temperature)
    soft_min_x_to_y = -temperature * torch.logsumexp(-dist_matrix / temperature, dim=1)
    
    # Soft max using logsumexp
    soft_max_x = temperature * torch.logsumexp(soft_min_x_to_y / temperature, dim=0)
    
    if not bidirectional:
        return soft_max_x
    
    soft_min_y_to_x = -temperature * torch.logsumexp(-dist_matrix / temperature, dim=0)
    soft_max_y = temperature * torch.logsumexp(soft_min_y_to_x / temperature, dim=0)
    
    return torch.max(soft_max_x, soft_max_y)


def average_hausdorff_distance(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Compute average Hausdorff distance.
    More robust variant using mean instead of max.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
    
    Returns:
        Average Hausdorff distance
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # Mean of min distances (this is essentially Chamfer distance with sqrt)
    avg_x_to_y = dist_matrix.min(dim=1)[0].mean()
    avg_y_to_x = dist_matrix.min(dim=0)[0].mean()
    
    return torch.max(avg_x_to_y, avg_y_to_x)


def modified_hausdorff_distance(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Compute Modified Hausdorff Distance (MHD).
    Uses mean instead of max for robustness.
    
    MHD(X, Y) = max(mean_x min_y d(x, y), mean_y min_x d(y, x))
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
    
    Returns:
        Modified Hausdorff distance
    """
    return average_hausdorff_distance(x, y)


def hausdorff_distance_with_info(
    x: torch.Tensor,
    y: torch.Tensor
) -> Tuple[torch.Tensor, dict]:
    """
    Compute Hausdorff distance with additional information.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
    
    Returns:
        Hausdorff distance and info dictionary
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # Min distances
    min_dist_x, min_idx_x = dist_matrix.min(dim=1)
    min_dist_y, min_idx_y = dist_matrix.min(dim=0)
    
    # Max of mins
    max_x, argmax_x = min_dist_x.max(dim=0)
    max_y, argmax_y = min_dist_y.max(dim=0)
    
    hausdorff = torch.max(max_x, max_y)
    
    info = {
        'hausdorff': hausdorff,
        'directed_x_to_y': max_x,
        'directed_y_to_x': max_y,
        'worst_point_x': argmax_x.item(),  # Point in x with max min-distance
        'worst_point_y': argmax_y.item(),  # Point in y with max min-distance
        'nearest_in_y': min_idx_x,  # For each point in x, nearest point in y
        'nearest_in_x': min_idx_y,  # For each point in y, nearest point in x
        'min_distances_x': min_dist_x,  # Min distance for each point in x
        'min_distances_y': min_dist_y,  # Min distance for each point in y
    }
    
    return hausdorff, info

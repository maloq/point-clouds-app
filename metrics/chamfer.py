"""
Chamfer distance implementations.
"""

import torch
from typing import Tuple, Optional


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between two point sets.
    
    Args:
        x: Point cloud [N, D]
        y: Point cloud [M, D]
    
    Returns:
        Distance matrix [N, M]
    """
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x.y
    xx = (x ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
    yy = (y ** 2).sum(dim=-1, keepdim=True)  # [M, 1]
    xy = x @ y.T  # [N, M]
    
    distances = xx + yy.T - 2 * xy
    return torch.clamp(distances, min=0)  # Numerical stability


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    bidirectional: bool = True,
    return_indices: bool = False,
    return_per_point: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Compute Chamfer distance between two point clouds.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        bidirectional: If True, compute both directions
        return_indices: If True, return nearest neighbor indices
        return_per_point: If True, return per-point distances
    
    Returns:
        Chamfer distance (and optionally indices and per-point distances)
    """
    # Compute pairwise distances
    dist_matrix = pairwise_distances(x, y)  # [N, M]
    
    # x -> y: for each point in x, find nearest in y
    dist_x_to_y, idx_x_to_y = dist_matrix.min(dim=1)  # [N]
    
    # y -> x: for each point in y, find nearest in x
    dist_y_to_x, idx_y_to_x = dist_matrix.min(dim=0)  # [M]
    
    # Chamfer distance
    if bidirectional:
        chamfer = dist_x_to_y.mean() + dist_y_to_x.mean()
    else:
        chamfer = dist_x_to_y.mean()
    
    results = [chamfer]
    
    if return_indices:
        results.append((idx_x_to_y, idx_y_to_x))
    
    if return_per_point:
        results.append((dist_x_to_y, dist_y_to_x))
    
    if len(results) == 1:
        return results[0]
    return tuple(results)


def chamfer_distance_sqrt(
    x: torch.Tensor,
    y: torch.Tensor,
    bidirectional: bool = True
) -> torch.Tensor:
    """
    Compute Chamfer distance with sqrt (actual distances, not squared).
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        bidirectional: If True, compute both directions
    
    Returns:
        Chamfer distance
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    dist_x_to_y = dist_matrix.min(dim=1)[0]
    dist_y_to_x = dist_matrix.min(dim=0)[0]
    
    if bidirectional:
        return dist_x_to_y.mean() + dist_y_to_x.mean()
    return dist_x_to_y.mean()


def chamfer_distance_weighted(
    x: torch.Tensor,
    y: torch.Tensor,
    weights_x: Optional[torch.Tensor] = None,
    weights_y: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute weighted Chamfer distance.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        weights_x: Weights for x points [N]
        weights_y: Weights for y points [M]
    
    Returns:
        Weighted Chamfer distance
    """
    if weights_x is None:
        weights_x = torch.ones(x.shape[0], device=x.device)
    if weights_y is None:
        weights_y = torch.ones(y.shape[0], device=y.device)
    
    # Normalize weights
    weights_x = weights_x / weights_x.sum()
    weights_y = weights_y / weights_y.sum()
    
    dist_matrix = pairwise_distances(x, y)
    
    dist_x_to_y = dist_matrix.min(dim=1)[0]
    dist_y_to_x = dist_matrix.min(dim=0)[0]
    
    chamfer = (weights_x * dist_x_to_y).sum() + (weights_y * dist_y_to_x).sum()
    
    return chamfer


def chamfer_distance_batched(
    x: torch.Tensor,
    y: torch.Tensor,
    bidirectional: bool = True
) -> torch.Tensor:
    """
    Compute Chamfer distance for batched point clouds.
    
    Args:
        x: Point clouds [B, N, 3]
        y: Point clouds [B, M, 3]
        bidirectional: If True, compute both directions
    
    Returns:
        Chamfer distances [B]
    """
    B, N, D = x.shape
    M = y.shape[1]
    
    # Compute pairwise distances
    xx = (x ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    yy = (y ** 2).sum(dim=-1, keepdim=True)  # [B, M, 1]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    
    dist_matrix = xx + yy.transpose(1, 2) - 2 * xy
    dist_matrix = torch.clamp(dist_matrix, min=0)
    
    dist_x_to_y = dist_matrix.min(dim=2)[0].mean(dim=1)  # [B]
    dist_y_to_x = dist_matrix.min(dim=1)[0].mean(dim=1)  # [B]
    
    if bidirectional:
        return dist_x_to_y + dist_y_to_x
    return dist_x_to_y


def get_correspondences(
    x: torch.Tensor,
    y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get nearest neighbor correspondences between point clouds.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
    
    Returns:
        idx_x_to_y: For each x, index of nearest y [N]
        idx_y_to_x: For each y, index of nearest x [M]
        dist_x_to_y: Distance to nearest y for each x [N]
        dist_y_to_x: Distance to nearest x for each y [M]
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix_sqrt = torch.sqrt(dist_matrix + 1e-8)
    
    dist_x_to_y, idx_x_to_y = dist_matrix_sqrt.min(dim=1)
    dist_y_to_x, idx_y_to_x = dist_matrix_sqrt.min(dim=0)
    
    return idx_x_to_y, idx_y_to_x, dist_x_to_y, dist_y_to_x

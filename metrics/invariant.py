"""
Rotation-invariant metrics based on pairwise distance distributions.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """
    Compute all pairwise distances within a point cloud.
    
    Args:
        x: Point cloud [N, 3]
    
    Returns:
        Distance matrix [N, N]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    dist = xx + xx.T - 2 * x @ x.T
    dist = torch.clamp(dist, min=0)
    return torch.sqrt(dist + 1e-8)


def get_upper_triangular_distances(dist_matrix: torch.Tensor) -> torch.Tensor:
    """
    Extract upper triangular (unique) distances from distance matrix.
    
    Args:
        dist_matrix: Distance matrix [N, N]
    
    Returns:
        Unique distances [N*(N-1)/2]
    """
    N = dist_matrix.shape[0]
    indices = torch.triu_indices(N, N, offset=1, device=dist_matrix.device)
    return dist_matrix[indices[0], indices[1]]


def pairwise_distance_histogram(
    x: torch.Tensor,
    n_bins: int = 50,
    range_min: float = 0.0,
    range_max: Optional[float] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute histogram of pairwise distances.
    
    Args:
        x: Point cloud [N, 3]
        n_bins: Number of histogram bins
        range_min: Minimum distance for histogram
        range_max: Maximum distance (auto if None)
    
    Returns:
        histogram: Normalized histogram counts [n_bins]
        bin_edges: Bin edges [n_bins + 1]
    """
    dist_matrix = compute_pairwise_distances(x)
    distances = get_upper_triangular_distances(dist_matrix)
    
    if range_max is None:
        range_max = distances.max().item() * 1.1
    
    # Compute histogram
    bin_edges = torch.linspace(range_min, range_max, n_bins + 1, device=x.device)
    histogram = torch.histc(distances, bins=n_bins, min=range_min, max=range_max)
    
    # Normalize
    histogram = histogram / (histogram.sum() + 1e-8)
    
    return histogram, bin_edges


def pairwise_distance_distribution_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 50,
    distance_type: str = 'l2'
) -> torch.Tensor:
    """
    Compute distance between pairwise distance distributions.
    This is rotation-invariant.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        n_bins: Number of histogram bins
        distance_type: 'l1', 'l2', 'chi2', 'kl', 'js', or 'emd'
    
    Returns:
        Distribution distance
    """
    # Get distance ranges
    dist_x = compute_pairwise_distances(x)
    dist_y = compute_pairwise_distances(y)
    
    unique_x = get_upper_triangular_distances(dist_x)
    unique_y = get_upper_triangular_distances(dist_y)
    
    range_max = max(unique_x.max().item(), unique_y.max().item()) * 1.1
    
    # Compute histograms with same range
    hist_x, _ = pairwise_distance_histogram(x, n_bins, 0, range_max)
    hist_y, _ = pairwise_distance_histogram(y, n_bins, 0, range_max)
    
    # Compute distance between histograms
    if distance_type == 'l1':
        return (hist_x - hist_y).abs().sum()
    
    elif distance_type == 'l2':
        return torch.sqrt(((hist_x - hist_y) ** 2).sum())
    
    elif distance_type == 'chi2':
        # Chi-squared distance
        denom = hist_x + hist_y + 1e-8
        return ((hist_x - hist_y) ** 2 / denom).sum() / 2
    
    elif distance_type == 'kl':
        # KL divergence (asymmetric)
        hist_x = hist_x + 1e-8
        hist_y = hist_y + 1e-8
        return (hist_x * torch.log(hist_x / hist_y)).sum()
    
    elif distance_type == 'js':
        # Jensen-Shannon divergence
        hist_x = hist_x + 1e-8
        hist_y = hist_y + 1e-8
        m = (hist_x + hist_y) / 2
        kl_xm = (hist_x * torch.log(hist_x / m)).sum()
        kl_ym = (hist_y * torch.log(hist_y / m)).sum()
        return (kl_xm + kl_ym) / 2
    
    elif distance_type == 'emd':
        # Earth mover's distance between histograms
        # Use cumulative distribution difference
        cdf_x = hist_x.cumsum(dim=0)
        cdf_y = hist_y.cumsum(dim=0)
        return (cdf_x - cdf_y).abs().sum() / n_bins
    
    else:
        raise ValueError(f"Unknown distance type: {distance_type}")


def d2_shape_distribution(
    x: torch.Tensor,
    n_samples: int = 10000,
    n_bins: int = 50
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute D2 shape distribution (Osada et al.).
    Distribution of distances between random point pairs.
    
    Args:
        x: Point cloud [N, 3]
        n_samples: Number of random pairs to sample
        n_bins: Number of histogram bins
    
    Returns:
        histogram: Normalized histogram [n_bins]
        bin_edges: Bin edges [n_bins + 1]
    """
    N = x.shape[0]
    
    # Sample random pairs
    idx1 = torch.randint(0, N, (n_samples,), device=x.device)
    idx2 = torch.randint(0, N, (n_samples,), device=x.device)
    
    # Compute distances
    distances = (x[idx1] - x[idx2]).norm(dim=-1)
    
    # Remove self-distances (when idx1 == idx2)
    valid = idx1 != idx2
    distances = distances[valid]
    
    # Compute histogram
    range_max = distances.max().item() * 1.1
    bin_edges = torch.linspace(0, range_max, n_bins + 1, device=x.device)
    histogram = torch.histc(distances, bins=n_bins, min=0, max=range_max)
    histogram = histogram / (histogram.sum() + 1e-8)
    
    return histogram, bin_edges


def d2_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 10000,
    n_bins: int = 50
) -> torch.Tensor:
    """
    Compute distance based on D2 shape distributions.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        n_samples: Number of random pairs
        n_bins: Number of histogram bins
    
    Returns:
        D2 distance
    """
    # Get distributions
    hist_x, edges_x = d2_shape_distribution(x, n_samples, n_bins)
    hist_y, edges_y = d2_shape_distribution(y, n_samples, n_bins)
    
    # Use common range
    range_max = max(edges_x[-1].item(), edges_y[-1].item())
    
    # Recompute with common range
    N_x, N_y = x.shape[0], y.shape[0]
    
    idx1_x = torch.randint(0, N_x, (n_samples,), device=x.device)
    idx2_x = torch.randint(0, N_x, (n_samples,), device=x.device)
    dist_x = (x[idx1_x] - x[idx2_x]).norm(dim=-1)
    dist_x = dist_x[idx1_x != idx2_x]
    
    idx1_y = torch.randint(0, N_y, (n_samples,), device=y.device)
    idx2_y = torch.randint(0, N_y, (n_samples,), device=y.device)
    dist_y = (y[idx1_y] - y[idx2_y]).norm(dim=-1)
    dist_y = dist_y[idx1_y != idx2_y]
    
    hist_x = torch.histc(dist_x, bins=n_bins, min=0, max=range_max)
    hist_y = torch.histc(dist_y, bins=n_bins, min=0, max=range_max)
    
    hist_x = hist_x / (hist_x.sum() + 1e-8)
    hist_y = hist_y / (hist_y.sum() + 1e-8)
    
    # L2 distance between histograms
    return torch.sqrt(((hist_x - hist_y) ** 2).sum())


def moment_invariants(
    x: torch.Tensor,
    max_order: int = 4
) -> torch.Tensor:
    """
    Compute moment invariants of pairwise distance distribution.
    
    Args:
        x: Point cloud [N, 3]
        max_order: Maximum moment order
    
    Returns:
        Moment values [max_order]
    """
    dist_matrix = compute_pairwise_distances(x)
    distances = get_upper_triangular_distances(dist_matrix)
    
    # Normalize distances
    mean = distances.mean()
    std = distances.std() + 1e-8
    normalized = (distances - mean) / std
    
    # Compute central moments
    moments = torch.zeros(max_order, device=x.device)
    for k in range(max_order):
        moments[k] = (normalized ** (k + 1)).mean()
    
    return moments


def moment_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    max_order: int = 4
) -> torch.Tensor:
    """
    Compute distance based on moment invariants.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        max_order: Maximum moment order
    
    Returns:
        Moment-based distance
    """
    moments_x = moment_invariants(x, max_order)
    moments_y = moment_invariants(y, max_order)
    
    return torch.sqrt(((moments_x - moments_y) ** 2).sum())


def radius_distribution_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 50
) -> torch.Tensor:
    """
    Compute distance based on distribution of point distances from centroid.
    This is rotation-invariant.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        n_bins: Number of histogram bins
    
    Returns:
        Radius distribution distance
    """
    # Compute distances from centroid
    centroid_x = x.mean(dim=0)
    centroid_y = y.mean(dim=0)
    
    radii_x = (x - centroid_x).norm(dim=-1)
    radii_y = (y - centroid_y).norm(dim=-1)
    
    # Common range
    range_max = max(radii_x.max().item(), radii_y.max().item()) * 1.1
    
    # Compute histograms
    hist_x = torch.histc(radii_x, bins=n_bins, min=0, max=range_max)
    hist_y = torch.histc(radii_y, bins=n_bins, min=0, max=range_max)
    
    hist_x = hist_x / (hist_x.sum() + 1e-8)
    hist_y = hist_y / (hist_y.sum() + 1e-8)
    
    # L2 distance
    return torch.sqrt(((hist_x - hist_y) ** 2).sum())


def intrinsic_volume_ratio(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compare intrinsic "volume" ratios using local density estimates.
    Rotation-invariant measure of point distribution.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        k: Number of neighbors for density estimation
    
    Returns:
        Volume ratio distance
    """
    def local_density(points, k):
        N = points.shape[0]
        dist = compute_pairwise_distances(points)
        # Set diagonal to large value
        dist = dist + torch.eye(N, device=points.device) * 1e10
        # k-th nearest neighbor distance
        knn_dist, _ = dist.topk(k, dim=1, largest=False)
        radius = knn_dist[:, -1]
        # Volume ~ radius^3
        volume = radius ** 3
        return volume.mean()
    
    vol_x = local_density(x, k)
    vol_y = local_density(y, k)
    
    # Log ratio of volumes
    return torch.abs(torch.log(vol_x / (vol_y + 1e-8)))

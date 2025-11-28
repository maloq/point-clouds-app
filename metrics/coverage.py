"""
Coverage and Precision metrics for point cloud comparison.
Common in generative model evaluation.
"""

import torch
from typing import Tuple


def pairwise_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared distances between two point sets.
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    yy = (y ** 2).sum(dim=-1, keepdim=True)
    xy = x @ y.T
    
    distances = xx + yy.T - 2 * xy
    return torch.clamp(distances, min=0)


def coverage(
    reference: torch.Tensor,
    generated: torch.Tensor,
    threshold: float = None,
    k: int = 1
) -> torch.Tensor:
    """
    Compute coverage: fraction of reference points that have a nearby generated point.
    
    Args:
        reference: Reference point cloud [N, 3]
        generated: Generated point cloud [M, 3]
        threshold: Distance threshold (if None, uses adaptive threshold)
        k: Number of nearest neighbors to consider
    
    Returns:
        Coverage score in [0, 1]
    """
    dist_matrix = pairwise_distances(reference, generated)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # For each reference point, find distance to k-th nearest generated point
    min_dists, _ = dist_matrix.topk(k, dim=1, largest=False)
    min_dists = min_dists[:, -1]  # k-th nearest neighbor distance
    
    if threshold is None:
        # Adaptive threshold: median of nearest neighbor distances within reference
        ref_dist = pairwise_distances(reference, reference)
        ref_dist = torch.sqrt(ref_dist + 1e-8)
        ref_dist = ref_dist + torch.eye(reference.shape[0], device=reference.device) * 1e10
        ref_nn_dists = ref_dist.min(dim=1)[0]
        threshold = ref_nn_dists.median() * 2
    
    # Count reference points that are covered
    covered = (min_dists <= threshold).float()
    
    return covered.mean()


def precision(
    reference: torch.Tensor,
    generated: torch.Tensor,
    threshold: float = None,
    k: int = 1
) -> torch.Tensor:
    """
    Compute precision: fraction of generated points that are near a reference point.
    
    Args:
        reference: Reference point cloud [N, 3]
        generated: Generated point cloud [M, 3]
        threshold: Distance threshold (if None, uses adaptive threshold)
        k: Number of nearest neighbors to consider
    
    Returns:
        Precision score in [0, 1]
    """
    dist_matrix = pairwise_distances(generated, reference)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # For each generated point, find distance to k-th nearest reference point
    min_dists, _ = dist_matrix.topk(k, dim=1, largest=False)
    min_dists = min_dists[:, -1]
    
    if threshold is None:
        # Adaptive threshold: median of nearest neighbor distances within reference
        ref_dist = pairwise_distances(reference, reference)
        ref_dist = torch.sqrt(ref_dist + 1e-8)
        ref_dist = ref_dist + torch.eye(reference.shape[0], device=reference.device) * 1e10
        ref_nn_dists = ref_dist.min(dim=1)[0]
        threshold = ref_nn_dists.median() * 2
    
    # Count generated points that are precise
    precise = (min_dists <= threshold).float()
    
    return precise.mean()


def coverage_precision(
    reference: torch.Tensor,
    generated: torch.Tensor,
    threshold: float = None,
    k: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both coverage and precision.
    
    Args:
        reference: Reference point cloud [N, 3]
        generated: Generated point cloud [M, 3]
        threshold: Distance threshold
        k: Number of nearest neighbors
    
    Returns:
        (coverage, precision) tuple
    """
    # Compute adaptive threshold if needed
    if threshold is None:
        ref_dist = pairwise_distances(reference, reference)
        ref_dist = torch.sqrt(ref_dist + 1e-8)
        ref_dist = ref_dist + torch.eye(reference.shape[0], device=reference.device) * 1e10
        ref_nn_dists = ref_dist.min(dim=1)[0]
        threshold = ref_nn_dists.median() * 2
    
    cov = coverage(reference, generated, threshold, k)
    prec = precision(reference, generated, threshold, k)
    
    return cov, prec


def f1_score(
    reference: torch.Tensor,
    generated: torch.Tensor,
    threshold: float = None,
    k: int = 1
) -> torch.Tensor:
    """
    Compute F1 score (harmonic mean of coverage and precision).
    
    Args:
        reference: Reference point cloud [N, 3]
        generated: Generated point cloud [M, 3]
        threshold: Distance threshold
        k: Number of nearest neighbors
    
    Returns:
        F1 score in [0, 1]
    """
    cov, prec = coverage_precision(reference, generated, threshold, k)
    
    if cov + prec == 0:
        return torch.tensor(0.0, device=reference.device)
    
    return 2 * cov * prec / (cov + prec)


def density(
    reference: torch.Tensor,
    generated: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compute density: average number of generated points around each reference point.
    
    Args:
        reference: Reference point cloud [N, 3]
        generated: Generated point cloud [M, 3]
        k: Number of nearest neighbors for radius estimation
    
    Returns:
        Density score
    """
    # Estimate local radius from reference point cloud
    ref_dist = pairwise_distances(reference, reference)
    ref_dist = torch.sqrt(ref_dist + 1e-8)
    ref_dist = ref_dist + torch.eye(reference.shape[0], device=reference.device) * 1e10
    
    # k-th nearest neighbor distance as radius
    radii, _ = ref_dist.topk(k, dim=1, largest=False)
    radii = radii[:, -1]  # [N]
    
    # Count generated points within radius of each reference point
    dist_ref_gen = pairwise_distances(reference, generated)
    dist_ref_gen = torch.sqrt(dist_ref_gen + 1e-8)
    
    # Count points within radius
    within_radius = (dist_ref_gen <= radii.unsqueeze(1)).float()
    counts = within_radius.sum(dim=1)  # [N]
    
    return counts.mean()


def recall_at_k(
    reference: torch.Tensor,
    generated: torch.Tensor,
    k: int = 5
) -> torch.Tensor:
    """
    Compute recall at k: fraction of reference points with at least one
    generated point among their k-nearest neighbors.
    
    Args:
        reference: Reference point cloud [N, 3]
        generated: Generated point cloud [M, 3]
        k: Number of nearest neighbors
    
    Returns:
        Recall score
    """
    N = reference.shape[0]
    
    # For each reference point, check if any generated point is among k-nearest in full set
    # This is a simplified version - we check if nearest generated point is close enough
    
    dist_matrix = pairwise_distances(reference, generated)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    # k-th nearest reference-reference distance
    ref_dist = pairwise_distances(reference, reference)
    ref_dist = torch.sqrt(ref_dist + 1e-8)
    ref_dist = ref_dist + torch.eye(N, device=reference.device) * 1e10
    
    ref_knn, _ = ref_dist.topk(k, dim=1, largest=False)
    threshold = ref_knn[:, -1]  # k-th NN distance for each reference point
    
    # Check if nearest generated point is within threshold
    min_gen_dist = dist_matrix.min(dim=1)[0]
    recalled = (min_gen_dist <= threshold).float()
    
    return recalled.mean()


def minimum_matching_distance(
    x: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    """
    Compute Minimum Matching Distance (MMD).
    Average distance from each point in x to nearest point in y.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
    
    Returns:
        MMD value
    """
    dist_matrix = pairwise_distances(x, y)
    dist_matrix = torch.sqrt(dist_matrix + 1e-8)
    
    min_dists = dist_matrix.min(dim=1)[0]
    
    return min_dists.mean()

"""
Spectral distance metrics based on Laplacian eigenvalues.
These metrics are rotation-invariant as they depend only on intrinsic geometry.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_knn_graph(
    x: torch.Tensor,
    k: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k-nearest neighbors graph.
    
    Args:
        x: Point cloud [N, 3]
        k: Number of nearest neighbors
    
    Returns:
        indices: KNN indices [N, k]
        distances: KNN distances [N, k]
    """
    N = x.shape[0]
    
    # Pairwise distances
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    dist = xx + xx.T - 2 * x @ x.T
    dist = torch.clamp(dist, min=0)
    dist = torch.sqrt(dist + 1e-8)
    
    # Set self-distance to large value
    dist = dist + torch.eye(N, device=x.device) * 1e10
    
    # Get k nearest neighbors
    distances, indices = dist.topk(k, dim=1, largest=False)
    
    return indices, distances


def compute_adjacency_matrix(
    x: torch.Tensor,
    k: int = 10,
    sigma: Optional[float] = None
) -> torch.Tensor:
    """
    Compute adjacency matrix using Gaussian kernel on KNN graph.
    
    Args:
        x: Point cloud [N, 3]
        k: Number of nearest neighbors
        sigma: Kernel bandwidth (auto if None)
    
    Returns:
        Adjacency matrix [N, N]
    """
    N = x.shape[0]
    indices, distances = compute_knn_graph(x, k)
    
    if sigma is None:
        # Use median distance as bandwidth
        sigma = distances.median().item()
        if sigma < 1e-8:
            sigma = 1.0
    
    # Create sparse adjacency matrix with Gaussian weights
    A = torch.zeros(N, N, device=x.device)
    
    for i in range(N):
        A[i, indices[i]] = torch.exp(-distances[i] ** 2 / (2 * sigma ** 2))
    
    # Make symmetric
    A = (A + A.T) / 2
    
    return A


def compute_laplacian(
    A: torch.Tensor,
    normalized: bool = True
) -> torch.Tensor:
    """
    Compute graph Laplacian from adjacency matrix.
    
    Args:
        A: Adjacency matrix [N, N]
        normalized: If True, compute normalized Laplacian
    
    Returns:
        Laplacian matrix [N, N]
    """
    D = A.sum(dim=1)
    
    if normalized:
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        D_inv_sqrt = torch.diag(1.0 / (torch.sqrt(D) + 1e-8))
        L = torch.eye(A.shape[0], device=A.device) - D_inv_sqrt @ A @ D_inv_sqrt
    else:
        # Unnormalized Laplacian: L = D - A
        L = torch.diag(D) - A
    
    return L


def compute_laplacian_eigenvalues(
    x: torch.Tensor,
    k_neighbors: int = 10,
    n_eigenvalues: int = 20,
    normalized: bool = True
) -> torch.Tensor:
    """
    Compute Laplacian eigenvalues for a point cloud.
    
    Args:
        x: Point cloud [N, 3]
        k_neighbors: Number of nearest neighbors for graph
        n_eigenvalues: Number of eigenvalues to compute
        normalized: If True, use normalized Laplacian
    
    Returns:
        Eigenvalues [n_eigenvalues]
    """
    A = compute_adjacency_matrix(x, k_neighbors)
    L = compute_laplacian(A, normalized)
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvalsh(L)
    
    # Sort and take smallest (including zero for connected graph)
    eigenvalues = eigenvalues.sort()[0]
    
    # Pad or truncate
    n = min(n_eigenvalues, len(eigenvalues))
    result = torch.zeros(n_eigenvalues, device=x.device)
    result[:n] = eigenvalues[:n]
    
    return result


def spectral_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    k_neighbors: int = 10,
    n_eigenvalues: int = 20,
    normalized: bool = True,
    p: int = 2
) -> torch.Tensor:
    """
    Compute spectral distance between two point clouds.
    Based on difference in Laplacian eigenvalues.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        k_neighbors: Number of nearest neighbors for graph
        n_eigenvalues: Number of eigenvalues to compare
        normalized: If True, use normalized Laplacian
        p: Norm to use for comparison (1 or 2)
    
    Returns:
        Spectral distance
    """
    eig_x = compute_laplacian_eigenvalues(x, k_neighbors, n_eigenvalues, normalized)
    eig_y = compute_laplacian_eigenvalues(y, k_neighbors, n_eigenvalues, normalized)
    
    if p == 2:
        return torch.sqrt(((eig_x - eig_y) ** 2).sum())
    elif p == 1:
        return (eig_x - eig_y).abs().sum()
    else:
        return ((eig_x - eig_y).abs() ** p).sum() ** (1/p)


def heat_kernel_signature(
    x: torch.Tensor,
    t: torch.Tensor,
    k_neighbors: int = 10,
    n_eigenvalues: int = 50
) -> torch.Tensor:
    """
    Compute Heat Kernel Signature for a point cloud.
    HKS is a rotation-invariant local descriptor.
    
    Args:
        x: Point cloud [N, 3]
        t: Time parameters [T]
        k_neighbors: Number of nearest neighbors
        n_eigenvalues: Number of eigenvalues to use
    
    Returns:
        HKS values [N, T]
    """
    N = x.shape[0]
    T = len(t)
    
    A = compute_adjacency_matrix(x, k_neighbors)
    L = compute_laplacian(A, normalized=True)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Use only first n_eigenvalues
    n = min(n_eigenvalues, N)
    eigenvalues = eigenvalues[:n]
    eigenvectors = eigenvectors[:, :n]
    
    # Compute HKS
    # HKS(x, t) = sum_k exp(-eigenvalue_k * t) * eigenvector_k(x)^2
    hks = torch.zeros(N, T, device=x.device)
    
    for i, time in enumerate(t):
        weights = torch.exp(-eigenvalues * time)
        hks[:, i] = (eigenvectors ** 2 @ weights)
    
    return hks


def hks_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    t: Optional[torch.Tensor] = None,
    k_neighbors: int = 10,
    n_eigenvalues: int = 50
) -> torch.Tensor:
    """
    Compute distance based on Heat Kernel Signature histograms.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        t: Time parameters (auto if None)
        k_neighbors: Number of nearest neighbors
        n_eigenvalues: Number of eigenvalues to use
    
    Returns:
        HKS-based distance
    """
    if t is None:
        t = torch.logspace(-2, 2, 10, device=x.device)
    
    hks_x = heat_kernel_signature(x, t, k_neighbors, n_eigenvalues)
    hks_y = heat_kernel_signature(y, t, k_neighbors, n_eigenvalues)
    
    # Compare distributions of HKS values using histogram comparison
    # Use mean and std as summary statistics
    mean_x = hks_x.mean(dim=0)
    mean_y = hks_y.mean(dim=0)
    std_x = hks_x.std(dim=0)
    std_y = hks_y.std(dim=0)
    
    # Combine mean and std differences
    mean_diff = ((mean_x - mean_y) ** 2).sum()
    std_diff = ((std_x - std_y) ** 2).sum()
    
    return torch.sqrt(mean_diff + std_diff)


def wave_kernel_signature(
    x: torch.Tensor,
    energies: torch.Tensor,
    sigma: float = 0.1,
    k_neighbors: int = 10,
    n_eigenvalues: int = 50
) -> torch.Tensor:
    """
    Compute Wave Kernel Signature for a point cloud.
    WKS is another rotation-invariant descriptor.
    
    Args:
        x: Point cloud [N, 3]
        energies: Energy levels [E]
        sigma: Energy scale variance
        k_neighbors: Number of nearest neighbors
        n_eigenvalues: Number of eigenvalues to use
    
    Returns:
        WKS values [N, E]
    """
    N = x.shape[0]
    E = len(energies)
    
    A = compute_adjacency_matrix(x, k_neighbors)
    L = compute_laplacian(A, normalized=True)
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L)
    
    # Use only first n_eigenvalues
    n = min(n_eigenvalues, N)
    eigenvalues = eigenvalues[:n]
    eigenvectors = eigenvectors[:, :n]
    
    # log of eigenvalues (skip first zero eigenvalue)
    log_eigenvalues = torch.log(eigenvalues[1:] + 1e-8)
    eigenvectors = eigenvectors[:, 1:]
    
    # Compute WKS
    wks = torch.zeros(N, E, device=x.device)
    
    for i, energy in enumerate(energies):
        weights = torch.exp(-(log_eigenvalues - energy) ** 2 / (2 * sigma ** 2))
        weights = weights / (weights.sum() + 1e-8)  # Normalize
        wks[:, i] = (eigenvectors ** 2 @ weights)
    
    return wks


def shape_dna_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    k_neighbors: int = 10,
    n_eigenvalues: int = 20
) -> torch.Tensor:
    """
    Compute Shape DNA distance based on Laplacian spectrum.
    Shape DNA is the sequence of Laplacian eigenvalues.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        k_neighbors: Number of nearest neighbors
        n_eigenvalues: Number of eigenvalues to use
    
    Returns:
        Shape DNA distance
    """
    # This is essentially the same as spectral_distance with p=2
    return spectral_distance(x, y, k_neighbors, n_eigenvalues, normalized=True, p=2)

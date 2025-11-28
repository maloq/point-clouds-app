"""
Earth Mover's Distance (EMD) and Optimal Transport metrics.
"""

import torch
import numpy as np
from typing import Tuple, Optional

# Try to import optional dependencies
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False

try:
    from geomloss import SamplesLoss
    HAS_GEOMLOSS = True
except ImportError:
    HAS_GEOMLOSS = False


def compute_cost_matrix(x: torch.Tensor, y: torch.Tensor, p: int = 2) -> torch.Tensor:
    """
    Compute cost matrix between two point sets.
    
    Args:
        x: Point cloud [N, D]
        y: Point cloud [M, D]
        p: Power for distance (1 or 2)
    
    Returns:
        Cost matrix [N, M]
    """
    diff = x.unsqueeze(1) - y.unsqueeze(0)  # [N, M, D]
    
    if p == 2:
        return (diff ** 2).sum(dim=-1)
    elif p == 1:
        return diff.abs().sum(dim=-1)
    else:
        return (diff.abs() ** p).sum(dim=-1)


def emd_exact(
    x: torch.Tensor,
    y: torch.Tensor,
    return_transport: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Compute exact Earth Mover's Distance using linear programming.
    WARNING: O(n^3) complexity, only suitable for small point clouds (<500 points).
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        return_transport: If True, return transport plan
    
    Returns:
        EMD value (and optionally transport plan)
    """
    if not HAS_POT:
        raise ImportError("POT library required for exact EMD. Install with: pip install pot")
    
    N, M = x.shape[0], y.shape[0]
    
    if N > 500 or M > 500:
        raise ValueError(f"Exact EMD is too slow for {N}x{M} points. Use approximate methods.")
    
    # Compute cost matrix
    cost = compute_cost_matrix(x, y, p=2)
    cost_np = cost.detach().cpu().numpy()
    
    # Uniform weights
    a = np.ones(N) / N
    b = np.ones(M) / M
    
    # Compute EMD
    transport = ot.emd(a, b, cost_np)
    emd = np.sum(transport * cost_np)
    
    result = torch.tensor(emd, dtype=x.dtype, device=x.device)
    
    if return_transport:
        transport_tensor = torch.tensor(transport, dtype=x.dtype, device=x.device)
        return result, transport_tensor
    
    return result


def sinkhorn_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
    threshold: float = 1e-9,
    return_transport: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Compute Sinkhorn distance (entropy-regularized optimal transport).
    Differentiable approximation to EMD.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        epsilon: Regularization parameter (smaller = closer to EMD, but slower)
        max_iter: Maximum iterations
        threshold: Convergence threshold
        return_transport: If True, return transport plan
    
    Returns:
        Sinkhorn distance (and optionally transport plan)
    """
    N, M = x.shape[0], y.shape[0]
    
    # Cost matrix
    C = compute_cost_matrix(x, y, p=2)
    
    # Gibbs kernel
    K = torch.exp(-C / epsilon)
    
    # Uniform marginals
    a = torch.ones(N, device=x.device) / N
    b = torch.ones(M, device=y.device) / M
    
    # Sinkhorn iterations
    u = torch.ones(N, device=x.device)
    v = torch.ones(M, device=y.device)
    
    for _ in range(max_iter):
        u_prev = u
        
        u = a / (K @ v + 1e-8)
        v = b / (K.T @ u + 1e-8)
        
        # Check convergence
        if (u - u_prev).abs().max() < threshold:
            break
    
    # Transport plan
    transport = u.unsqueeze(1) * K * v.unsqueeze(0)
    
    # Sinkhorn distance
    distance = (transport * C).sum()
    
    if return_transport:
        return distance, transport
    
    return distance


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Compute Sinkhorn divergence (debiased Sinkhorn distance).
    S(x, y) = W_eps(x, y) - 0.5 * W_eps(x, x) - 0.5 * W_eps(y, y)
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        epsilon: Regularization parameter
        max_iter: Maximum iterations
    
    Returns:
        Sinkhorn divergence
    """
    w_xy = sinkhorn_distance(x, y, epsilon, max_iter)
    w_xx = sinkhorn_distance(x, x, epsilon, max_iter)
    w_yy = sinkhorn_distance(y, y, epsilon, max_iter)
    
    return w_xy - 0.5 * w_xx - 0.5 * w_yy


def sliced_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    n_projections: int = 100,
    p: int = 2
) -> torch.Tensor:
    """
    Compute Sliced Wasserstein Distance.
    Projects point clouds onto random 1D directions and computes 1D Wasserstein.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        n_projections: Number of random projections
        p: Power for Wasserstein distance
    
    Returns:
        Sliced Wasserstein distance
    """
    D = x.shape[1]
    N, M = x.shape[0], y.shape[0]
    
    # Generate random directions on unit sphere
    directions = torch.randn(n_projections, D, device=x.device)
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-8)
    
    # Project points
    proj_x = x @ directions.T  # [N, n_projections]
    proj_y = y @ directions.T  # [M, n_projections]
    
    # Sort projections
    proj_x_sorted, _ = proj_x.sort(dim=0)
    proj_y_sorted, _ = proj_y.sort(dim=0)
    
    # Interpolate to same size if different
    if N != M:
        # Use linear interpolation
        t_x = torch.linspace(0, 1, N, device=x.device)
        t_y = torch.linspace(0, 1, M, device=y.device)
        t_common = torch.linspace(0, 1, max(N, M), device=x.device)
        
        # Interpolate both to common size
        proj_x_interp = torch.zeros(len(t_common), n_projections, device=x.device)
        proj_y_interp = torch.zeros(len(t_common), n_projections, device=y.device)
        
        for i in range(n_projections):
            proj_x_interp[:, i] = torch.from_numpy(
                np.interp(t_common.cpu().numpy(), t_x.cpu().numpy(), proj_x_sorted[:, i].cpu().numpy())
            ).to(x.device)
            proj_y_interp[:, i] = torch.from_numpy(
                np.interp(t_common.cpu().numpy(), t_y.cpu().numpy(), proj_y_sorted[:, i].cpu().numpy())
            ).to(y.device)
        
        proj_x_sorted = proj_x_interp
        proj_y_sorted = proj_y_interp
    
    # Compute 1D Wasserstein distances
    if p == 2:
        wasserstein_1d = ((proj_x_sorted - proj_y_sorted) ** 2).mean(dim=0)
        swd = wasserstein_1d.mean().sqrt()
    else:
        wasserstein_1d = (proj_x_sorted - proj_y_sorted).abs().pow(p).mean(dim=0)
        swd = wasserstein_1d.mean().pow(1/p)
    
    return swd


def geomloss_sinkhorn(
    x: torch.Tensor,
    y: torch.Tensor,
    blur: float = 0.05,
    reach: Optional[float] = None,
    scaling: float = 0.5
) -> torch.Tensor:
    """
    Compute Sinkhorn distance using GeomLoss library.
    More efficient implementation with GPU support.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        blur: Regularization parameter
        reach: Reach parameter for unbalanced OT (None for balanced)
        scaling: Multiscale parameter
    
    Returns:
        Sinkhorn distance
    """
    if not HAS_GEOMLOSS:
        raise ImportError("GeomLoss library required. Install with: pip install geomloss")
    
    loss = SamplesLoss(
        loss="sinkhorn",
        blur=blur,
        reach=reach,
        scaling=scaling,
        debias=True  # Use Sinkhorn divergence
    )
    
    return loss(x, y)


def emd_approximate(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = "sinkhorn",
    **kwargs
) -> torch.Tensor:
    """
    Compute approximate EMD using specified method.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        method: One of "sinkhorn", "sliced", "geomloss"
        **kwargs: Additional arguments for specific method
    
    Returns:
        Approximate EMD
    """
    if method == "sinkhorn":
        return sinkhorn_distance(x, y, **kwargs)
    elif method == "sliced":
        return sliced_wasserstein_distance(x, y, **kwargs)
    elif method == "geomloss":
        return geomloss_sinkhorn(x, y, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def get_optimal_transport_plan(
    x: torch.Tensor,
    y: torch.Tensor,
    method: str = "sinkhorn",
    epsilon: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get optimal transport plan between point clouds.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        method: "exact" or "sinkhorn"
        epsilon: Regularization for Sinkhorn
    
    Returns:
        transport: Transport plan [N, M]
        distance: EMD/Sinkhorn distance
    """
    if method == "exact":
        if not HAS_POT:
            raise ImportError("POT library required")
        return emd_exact(x, y, return_transport=True)
    else:
        return sinkhorn_distance(x, y, epsilon=epsilon, return_transport=True)


def wasserstein_barycenter(
    point_clouds: list,
    weights: Optional[torch.Tensor] = None,
    n_points: int = 100,
    max_iter: int = 100,
    epsilon: float = 0.1
) -> torch.Tensor:
    """
    Compute Wasserstein barycenter of multiple point clouds.
    
    Args:
        point_clouds: List of point cloud tensors
        weights: Weights for each point cloud
        n_points: Number of points in barycenter
        max_iter: Maximum iterations
        epsilon: Regularization parameter
    
    Returns:
        Barycenter point cloud [n_points, 3]
    """
    K = len(point_clouds)
    D = point_clouds[0].shape[1]
    device = point_clouds[0].device
    
    if weights is None:
        weights = torch.ones(K, device=device) / K
    
    # Initialize barycenter as mean of centroids
    barycenter = torch.zeros(n_points, D, device=device)
    for i, pc in enumerate(point_clouds):
        # Sample or pad to n_points
        if pc.shape[0] >= n_points:
            indices = torch.randperm(pc.shape[0], device=device)[:n_points]
            barycenter += weights[i] * pc[indices]
        else:
            indices = torch.randint(0, pc.shape[0], (n_points,), device=device)
            barycenter += weights[i] * pc[indices]
    
    # Iteratively update barycenter
    for _ in range(max_iter):
        barycenter_new = torch.zeros_like(barycenter)
        
        for i, pc in enumerate(point_clouds):
            _, transport = sinkhorn_distance(
                barycenter, pc, epsilon=epsilon, return_transport=True
            )
            
            # Update barycenter based on transport
            transport_sum = transport.sum(dim=1, keepdim=True) + 1e-8
            barycenter_new += weights[i] * (transport @ pc) / transport_sum
        
        barycenter = barycenter_new
    
    return barycenter

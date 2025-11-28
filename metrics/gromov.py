"""
Gromov-Wasserstein distance implementation.
A rotation-invariant metric that compares intrinsic geometry.
"""

import torch
from typing import Tuple, Optional

try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False


def compute_distance_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise distance matrix within a point cloud.
    
    Args:
        x: Point cloud [N, 3]
    
    Returns:
        Distance matrix [N, N]
    """
    xx = (x ** 2).sum(dim=-1, keepdim=True)
    dist = xx + xx.T - 2 * x @ x.T
    dist = torch.clamp(dist, min=0)
    return torch.sqrt(dist + 1e-8)


def gromov_wasserstein_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fun: str = 'square_loss',
    max_iter: int = 100,
    tol: float = 1e-9,
    return_transport: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Compute Gromov-Wasserstein distance between two point clouds.
    This metric is rotation-invariant as it compares internal distances.
    
    GW(X, Y) = min_T sum_{i,j,k,l} |d_X(i,j) - d_Y(k,l)|^2 * T_ik * T_jl
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        loss_fun: Loss function ('square_loss' or 'kl_loss')
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        return_transport: If True, return transport plan
    
    Returns:
        GW distance (and optionally transport plan)
    """
    if not HAS_POT:
        raise ImportError("POT library required. Install with: pip install pot")
    
    N, M = x.shape[0], y.shape[0]
    
    # Compute internal distance matrices
    C1 = compute_distance_matrix(x)
    C2 = compute_distance_matrix(y)
    
    # Convert to numpy
    C1_np = C1.detach().cpu().numpy()
    C2_np = C2.detach().cpu().numpy()
    
    # Uniform distributions
    p = torch.ones(N) / N
    q = torch.ones(M) / M
    p_np = p.numpy()
    q_np = q.numpy()
    
    # Compute GW distance
    T, log = ot.gromov.gromov_wasserstein(
        C1_np, C2_np, p_np, q_np,
        loss_fun=loss_fun,
        max_iter=max_iter,
        tol=tol,
        log=True
    )
    
    gw_dist = torch.tensor(log['gw_dist'], dtype=x.dtype, device=x.device)
    
    if return_transport:
        T_tensor = torch.tensor(T, dtype=x.dtype, device=x.device)
        return gw_dist, T_tensor
    
    return gw_dist


def entropic_gromov_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 100,
    tol: float = 1e-9,
    return_transport: bool = False
) -> Tuple[torch.Tensor, ...]:
    """
    Compute entropic Gromov-Wasserstein distance.
    Regularized version that is more stable and faster.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        epsilon: Regularization parameter
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        return_transport: If True, return transport plan
    
    Returns:
        Entropic GW distance (and optionally transport plan)
    """
    if not HAS_POT:
        raise ImportError("POT library required. Install with: pip install pot")
    
    N, M = x.shape[0], y.shape[0]
    
    # Compute internal distance matrices
    C1 = compute_distance_matrix(x)
    C2 = compute_distance_matrix(y)
    
    # Convert to numpy
    C1_np = C1.detach().cpu().numpy()
    C2_np = C2.detach().cpu().numpy()
    
    # Uniform distributions
    p = torch.ones(N) / N
    q = torch.ones(M) / M
    p_np = p.numpy()
    q_np = q.numpy()
    
    # Compute entropic GW
    T, log = ot.gromov.entropic_gromov_wasserstein(
        C1_np, C2_np, p_np, q_np,
        loss_fun='square_loss',
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        log=True
    )
    
    gw_dist = torch.tensor(log['gw_dist'], dtype=x.dtype, device=x.device)
    
    if return_transport:
        T_tensor = torch.tensor(T, dtype=x.dtype, device=x.device)
        return gw_dist, T_tensor
    
    return gw_dist


def gromov_wasserstein_differentiable(
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.1,
    max_iter: int = 50
) -> torch.Tensor:
    """
    Differentiable approximation to Gromov-Wasserstein distance.
    Uses Sinkhorn-like iterations that maintain gradient flow.
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        epsilon: Regularization parameter
        max_iter: Maximum iterations
    
    Returns:
        Approximate GW distance (differentiable)
    """
    N, M = x.shape[0], y.shape[0]
    
    # Compute internal distance matrices
    C1 = compute_distance_matrix(x)  # [N, N]
    C2 = compute_distance_matrix(y)  # [M, M]
    
    # Initialize transport plan
    T = torch.ones(N, M, device=x.device) / (N * M)
    
    # Uniform marginals
    p = torch.ones(N, device=x.device) / N
    q = torch.ones(M, device=y.device) / M
    
    for _ in range(max_iter):
        # Compute gradient of GW objective w.r.t. T
        # gradient[i,k] = sum_{j,l} |C1[i,j] - C2[k,l]|^2 * T[j,l]
        
        # Efficient computation using matrix operations
        # |C1 - C2|^2 = C1^2 + C2^2 - 2*C1*C2
        # The gradient becomes: C1^2 @ T @ 1 + 1 @ T @ C2^2 - 2 * C1 @ T @ C2
        
        ones_N = torch.ones(N, 1, device=x.device)
        ones_M = torch.ones(M, 1, device=y.device)
        
        C1_sq = C1 ** 2
        C2_sq = C2 ** 2
        
        gradient = (C1_sq @ T @ ones_M @ ones_M.T +
                   ones_N @ ones_N.T @ T @ C2_sq -
                   2 * C1 @ T @ C2.T)
        
        # Sinkhorn step
        K = torch.exp(-gradient / epsilon)
        
        # Row normalization
        T = K * p.unsqueeze(1)
        T = T / (T.sum(dim=1, keepdim=True) + 1e-8)
        
        # Column normalization
        T = T * q.unsqueeze(0)
        T = T / (T.sum(dim=0, keepdim=True) + 1e-8)
    
    # Compute final GW distance
    # GW = sum_{ijkl} |C1[i,j] - C2[k,l]|^2 * T[i,k] * T[j,l]
    C1_expanded = C1.unsqueeze(2).unsqueeze(3)  # [N, N, 1, 1]
    C2_expanded = C2.unsqueeze(0).unsqueeze(1)  # [1, 1, M, M]
    T_expanded = T.unsqueeze(1).unsqueeze(3)    # [N, 1, M, 1]
    T_expanded2 = T.unsqueeze(0).unsqueeze(2)   # [1, N, 1, M]
    
    # This is memory intensive, use efficient formulation instead
    term1 = (C1 ** 2 * (T.sum(dim=1, keepdim=True) @ T.sum(dim=1, keepdim=True).T)).sum()
    term2 = (C2 ** 2 * (T.sum(dim=0, keepdim=True).T @ T.sum(dim=0, keepdim=True))).sum()
    term3 = 2 * ((C1 @ T @ C2.T) * T).sum()
    
    gw_dist = term1 + term2 - term3
    
    return gw_dist


def fused_gromov_wasserstein(
    x: torch.Tensor,
    y: torch.Tensor,
    features_x: Optional[torch.Tensor] = None,
    features_y: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    max_iter: int = 100
) -> torch.Tensor:
    """
    Compute Fused Gromov-Wasserstein distance.
    Combines structure (GW) and feature (W) information.
    
    FGW = alpha * GW(C1, C2) + (1-alpha) * W(features)
    
    Args:
        x: Point cloud [N, 3]
        y: Point cloud [M, 3]
        features_x: Features for x [N, D] (uses positions if None)
        features_y: Features for y [M, D] (uses positions if None)
        alpha: Balance between structure and features
        epsilon: Regularization parameter
        max_iter: Maximum iterations
    
    Returns:
        FGW distance
    """
    if not HAS_POT:
        raise ImportError("POT library required. Install with: pip install pot")
    
    N, M = x.shape[0], y.shape[0]
    
    # Use positions as features if not provided
    if features_x is None:
        features_x = x
    if features_y is None:
        features_y = y
    
    # Compute internal distance matrices (structure)
    C1 = compute_distance_matrix(x)
    C2 = compute_distance_matrix(y)
    
    # Compute feature distance matrix
    fx = features_x.unsqueeze(1)
    fy = features_y.unsqueeze(0)
    M_features = ((fx - fy) ** 2).sum(dim=-1)
    
    # Convert to numpy
    C1_np = C1.detach().cpu().numpy()
    C2_np = C2.detach().cpu().numpy()
    M_np = M_features.detach().cpu().numpy()
    
    # Uniform distributions
    p = torch.ones(N) / N
    q = torch.ones(M) / M
    
    # Compute FGW
    T, log = ot.gromov.fused_gromov_wasserstein(
        M_np, C1_np, C2_np, p.numpy(), q.numpy(),
        loss_fun='square_loss',
        alpha=alpha,
        log=True
    )
    
    fgw_dist = torch.tensor(log['fgw_dist'], dtype=x.dtype, device=x.device)
    
    return fgw_dist

"""
Rotation utilities: quaternions, rotation matrices, SO(3) operations.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Union


def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions (w, x, y, z format).
    
    Args:
        q1: Quaternion tensor [..., 4]
        q2: Quaternion tensor [..., 4]
    
    Returns:
        Product quaternion [..., 4]
    """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of quaternion (w, x, y, z format)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: Quaternion tensor [..., 4] in (w, x, y, z) format
    
    Returns:
        Rotation matrix [..., 3, 3]
    """
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    w, x, y, z = q.unbind(-1)
    
    # Rotation matrix from quaternion
    R = torch.stack([
        torch.stack([1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y], dim=-1),
        torch.stack([2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x], dim=-1),
        torch.stack([2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y], dim=-1),
    ], dim=-2)
    
    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to quaternion.
    
    Args:
        R: Rotation matrix [..., 3, 3]
    
    Returns:
        Quaternion [..., 4] in (w, x, y, z) format
    """
    batch_shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    
    q = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)
    
    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1
    
    # Case 2: R[0,0] > R[1,1] and R[0,0] > R[2,2]
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2
    
    # Case 3: R[1,1] > R[2,2]
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3
    
    # Case 4: else
    mask4 = ~mask1 & ~mask2 & ~mask3
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4
    
    q = q.reshape(*batch_shape, 4)
    return q


def euler_to_matrix(euler: torch.Tensor, order: str = 'xyz') -> torch.Tensor:
    """
    Convert Euler angles to rotation matrix.
    
    Args:
        euler: Euler angles [..., 3] in radians
        order: Rotation order (e.g., 'xyz', 'zyx')
    
    Returns:
        Rotation matrix [..., 3, 3]
    """
    angles = euler.unbind(-1)
    
    def Rx(theta):
        c, s = torch.cos(theta), torch.sin(theta)
        o, z = torch.ones_like(theta), torch.zeros_like(theta)
        return torch.stack([
            torch.stack([o, z, z], dim=-1),
            torch.stack([z, c, -s], dim=-1),
            torch.stack([z, s, c], dim=-1),
        ], dim=-2)
    
    def Ry(theta):
        c, s = torch.cos(theta), torch.sin(theta)
        o, z = torch.ones_like(theta), torch.zeros_like(theta)
        return torch.stack([
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ], dim=-2)
    
    def Rz(theta):
        c, s = torch.cos(theta), torch.sin(theta)
        o, z = torch.ones_like(theta), torch.zeros_like(theta)
        return torch.stack([
            torch.stack([c, -s, z], dim=-1),
            torch.stack([s, c, z], dim=-1),
            torch.stack([z, z, o], dim=-1),
        ], dim=-2)
    
    rot_funcs = {'x': Rx, 'y': Ry, 'z': Rz}
    
    R = torch.eye(3, device=euler.device, dtype=euler.dtype)
    R = R.expand(*euler.shape[:-1], 3, 3).clone()
    
    for i, axis in enumerate(order):
        R = R @ rot_funcs[axis](angles[i])
    
    return R


def matrix_to_euler(R: torch.Tensor, order: str = 'xyz') -> torch.Tensor:
    """
    Convert rotation matrix to Euler angles.
    
    Args:
        R: Rotation matrix [..., 3, 3]
        order: Rotation order (e.g., 'xyz')
    
    Returns:
        Euler angles [..., 3] in radians
    """
    if order == 'xyz':
        sy = torch.sqrt(R[..., 0, 0]**2 + R[..., 1, 0]**2)
        singular = sy < 1e-6
        
        x = torch.where(singular, torch.atan2(-R[..., 1, 2], R[..., 1, 1]),
                       torch.atan2(R[..., 2, 1], R[..., 2, 2]))
        y = torch.where(singular, torch.atan2(-R[..., 2, 0], sy),
                       torch.atan2(-R[..., 2, 0], sy))
        z = torch.where(singular, torch.zeros_like(sy),
                       torch.atan2(R[..., 1, 0], R[..., 0, 0]))
        
        return torch.stack([x, y, z], dim=-1)
    else:
        raise NotImplementedError(f"Order {order} not implemented")


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle representation to rotation matrix using Rodrigues formula.
    
    Args:
        axis: Rotation axis [..., 3] (will be normalized)
        angle: Rotation angle [...] in radians
    
    Returns:
        Rotation matrix [..., 3, 3]
    """
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    
    K = torch.zeros(*axis.shape[:-1], 3, 3, device=axis.device, dtype=axis.dtype)
    K[..., 0, 1] = -axis[..., 2]
    K[..., 0, 2] = axis[..., 1]
    K[..., 1, 0] = axis[..., 2]
    K[..., 1, 2] = -axis[..., 0]
    K[..., 2, 0] = -axis[..., 1]
    K[..., 2, 1] = axis[..., 0]
    
    angle = angle.unsqueeze(-1).unsqueeze(-1)
    I = torch.eye(3, device=axis.device, dtype=axis.dtype)
    
    R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
    
    return R


def matrix_to_axis_angle(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert rotation matrix to axis-angle representation.
    
    Args:
        R: Rotation matrix [..., 3, 3]
    
    Returns:
        axis: Rotation axis [..., 3]
        angle: Rotation angle [...] in radians
    """
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Axis from skew-symmetric part
    axis = torch.stack([
        R[..., 2, 1] - R[..., 1, 2],
        R[..., 0, 2] - R[..., 2, 0],
        R[..., 1, 0] - R[..., 0, 1],
    ], dim=-1)
    
    axis = axis / (2 * torch.sin(angle).unsqueeze(-1) + 1e-8)
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    
    return axis, angle


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to rotation matrix.
    From "On the Continuity of Rotation Representations in Neural Networks"
    (Zhou et al., CVPR 2019)
    
    Args:
        d6: 6D rotation representation [..., 6]
    
    Returns:
        Rotation matrix [..., 3, 3]
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    
    b1 = a1 / (a1.norm(dim=-1, keepdim=True) + 1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2 / (b2.norm(dim=-1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D representation.
    
    Args:
        R: Rotation matrix [..., 3, 3]
    
    Returns:
        6D rotation representation [..., 6]
    """
    return R[..., :2, :].reshape(*R.shape[:-2], 6)


def quaternion_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Spherical linear interpolation between quaternions.
    
    Args:
        q0: Start quaternion [..., 4]
        q1: End quaternion [..., 4]
        t: Interpolation parameter [...] in [0, 1]
    
    Returns:
        Interpolated quaternion [..., 4]
    """
    q0 = q0 / (q0.norm(dim=-1, keepdim=True) + 1e-8)
    q1 = q1 / (q1.norm(dim=-1, keepdim=True) + 1e-8)
    
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    
    # If dot < 0, negate one quaternion to take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)
    
    # Clamp dot product
    dot = torch.clamp(dot, -1, 1)
    
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    
    # Handle small angles
    small_angle = sin_theta.abs() < 1e-6
    
    t = t.unsqueeze(-1) if t.dim() < q0.dim() else t
    
    s0 = torch.where(small_angle, 1 - t, torch.sin((1 - t) * theta) / sin_theta)
    s1 = torch.where(small_angle, t, torch.sin(t * theta) / sin_theta)
    
    return s0 * q0 + s1 * q1


def random_quaternion(shape: Tuple[int, ...], device: torch.device = None) -> torch.Tensor:
    """
    Generate random unit quaternions uniformly distributed on SO(3).
    
    Args:
        shape: Output shape (excluding the quaternion dimension)
        device: Torch device
    
    Returns:
        Random quaternions [..., 4]
    """
    # Use the subgroup algorithm for uniform distribution
    u = torch.rand(*shape, 3, device=device)
    
    q = torch.stack([
        torch.sqrt(1 - u[..., 0]) * torch.sin(2 * np.pi * u[..., 1]),
        torch.sqrt(1 - u[..., 0]) * torch.cos(2 * np.pi * u[..., 1]),
        torch.sqrt(u[..., 0]) * torch.sin(2 * np.pi * u[..., 2]),
        torch.sqrt(u[..., 0]) * torch.cos(2 * np.pi * u[..., 2]),
    ], dim=-1)
    
    return q


def random_rotation_matrix(shape: Tuple[int, ...], device: torch.device = None) -> torch.Tensor:
    """
    Generate random rotation matrices uniformly distributed on SO(3).
    
    Args:
        shape: Output shape (excluding the matrix dimensions)
        device: Torch device
    
    Returns:
        Random rotation matrices [..., 3, 3]
    """
    q = random_quaternion(shape, device)
    return quaternion_to_matrix(q)


def geodesic_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance between rotation matrices on SO(3).
    
    Args:
        R1: Rotation matrix [..., 3, 3]
        R2: Rotation matrix [..., 3, 3]
    
    Returns:
        Geodesic distance [...] in radians
    """
    R_diff = R1.transpose(-1, -2) @ R2
    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]
    # Clamp for numerical stability
    cos_angle = torch.clamp((trace - 1) / 2, -1, 1)
    return torch.acos(cos_angle)


def interpolate_rotations(R_start: torch.Tensor, R_end: torch.Tensor, 
                          n_steps: int) -> torch.Tensor:
    """
    Interpolate between two rotations along geodesic path.
    
    Args:
        R_start: Start rotation matrix [3, 3]
        R_end: End rotation matrix [3, 3]
        n_steps: Number of interpolation steps
    
    Returns:
        Interpolated rotations [n_steps, 3, 3]
    """
    q_start = matrix_to_quaternion(R_start.unsqueeze(0))
    q_end = matrix_to_quaternion(R_end.unsqueeze(0))
    
    t = torch.linspace(0, 1, n_steps, device=R_start.device)
    
    q_interp = quaternion_slerp(
        q_start.expand(n_steps, -1),
        q_end.expand(n_steps, -1),
        t
    )
    
    return quaternion_to_matrix(q_interp)


def sample_rotation_heatmap(n_samples_per_axis: int = 50, 
                            device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample rotations for 2D heatmap visualization.
    Uses two Euler angles (keeping third fixed).
    
    Args:
        n_samples_per_axis: Number of samples per axis
        device: Torch device
    
    Returns:
        angles1: First angle values [n_samples_per_axis]
        angles2: Second angle values [n_samples_per_axis]
        rotations: Rotation matrices [n_samples_per_axis, n_samples_per_axis, 3, 3]
    """
    angles = torch.linspace(-np.pi, np.pi, n_samples_per_axis, device=device)
    
    angles1, angles2 = torch.meshgrid(angles, angles, indexing='ij')
    
    euler = torch.stack([
        angles1,
        angles2,
        torch.zeros_like(angles1)
    ], dim=-1)
    
    rotations = euler_to_matrix(euler)
    
    return angles, angles, rotations

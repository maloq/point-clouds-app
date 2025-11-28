"""
Point cloud structure generators.
"""

import torch
import numpy as np
from typing import Optional, Tuple
from enum import Enum


class StructureType(Enum):
    """Available point cloud structure types."""
    SPHERE_SURFACE = "sphere_surface"
    BALL_UNIFORM = "ball_uniform"
    FLAT_PLANE = "flat_plane"
    CUBE_SURFACE = "cube_surface"
    CUBE_FILLED = "cube_filled"
    LATTICE_SIMPLE_CUBIC = "lattice_simple_cubic"
    LATTICE_BCC = "lattice_bcc"
    LATTICE_FCC = "lattice_fcc"
    TORUS = "torus"
    HELIX = "helix"
    NORMAL_DISTRIBUTION = "normal_distribution"


def generate_sphere_surface(
    n_points: int,
    radius: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points uniformly distributed on sphere surface.
    Uses Fibonacci sphere algorithm for better distribution.
    
    Args:
        n_points: Number of points
        radius: Sphere radius
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    indices = torch.arange(n_points, dtype=torch.float32, device=device)
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Fibonacci sphere
    y = 1 - (2 * indices + 1) / n_points
    r = torch.sqrt(1 - y * y)
    theta = 2 * np.pi * indices / phi
    
    x = r * torch.cos(theta)
    z = r * torch.sin(theta)
    
    points = torch.stack([x, y, z], dim=-1) * radius
    
    return points


def generate_ball_uniform(
    n_points: int,
    radius: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points uniformly distributed inside a ball.
    
    Args:
        n_points: Number of points
        radius: Ball radius
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    # Generate random directions
    directions = torch.randn(n_points, 3, device=device)
    directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Generate random radii with cubic root for uniform distribution
    u = torch.rand(n_points, 1, device=device)
    r = radius * torch.pow(u, 1/3)
    
    points = directions * r
    
    return points


def generate_flat_plane(
    n_points: int,
    width: float = 2.0,
    height: float = 2.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points uniformly distributed on a flat plane (z=0).
    
    Args:
        n_points: Number of points
        width: Plane width
        height: Plane height
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    x = (torch.rand(n_points, device=device) - 0.5) * width
    y = (torch.rand(n_points, device=device) - 0.5) * height
    z = torch.zeros(n_points, device=device)
    
    return torch.stack([x, y, z], dim=-1)


def generate_cube_surface(
    n_points: int,
    size: float = 2.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points uniformly distributed on cube surface.
    
    Args:
        n_points: Number of points
        size: Cube side length
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    half = size / 2
    
    # Distribute points proportionally to face areas (all same)
    points_per_face = n_points // 6
    remainder = n_points % 6
    
    points_list = []
    
    for face_idx in range(6):
        n = points_per_face + (1 if face_idx < remainder else 0)
        
        # Generate random 2D coordinates
        u = (torch.rand(n, device=device) - 0.5) * size
        v = (torch.rand(n, device=device) - 0.5) * size
        
        if face_idx == 0:  # +X face
            pts = torch.stack([torch.full((n,), half, device=device), u, v], dim=-1)
        elif face_idx == 1:  # -X face
            pts = torch.stack([torch.full((n,), -half, device=device), u, v], dim=-1)
        elif face_idx == 2:  # +Y face
            pts = torch.stack([u, torch.full((n,), half, device=device), v], dim=-1)
        elif face_idx == 3:  # -Y face
            pts = torch.stack([u, torch.full((n,), -half, device=device), v], dim=-1)
        elif face_idx == 4:  # +Z face
            pts = torch.stack([u, v, torch.full((n,), half, device=device)], dim=-1)
        else:  # -Z face
            pts = torch.stack([u, v, torch.full((n,), -half, device=device)], dim=-1)
        
        points_list.append(pts)
    
    return torch.cat(points_list, dim=0)


def generate_cube_filled(
    n_points: int,
    size: float = 2.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points uniformly distributed inside a cube.
    
    Args:
        n_points: Number of points
        size: Cube side length
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    points = (torch.rand(n_points, 3, device=device) - 0.5) * size
    return points


def generate_lattice_simple_cubic(
    n_points: int,
    spacing: float = 0.5,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate simple cubic lattice points.
    
    Args:
        n_points: Approximate number of points
        spacing: Lattice spacing
        device: Torch device
    
    Returns:
        Point cloud tensor [actual_n_points, 3]
    """
    # Calculate grid size to get approximately n_points
    grid_size = int(np.ceil(n_points ** (1/3)))
    
    x = torch.arange(grid_size, device=device, dtype=torch.float32) * spacing
    x = x - x.mean()
    
    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
    points = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    # Subsample if too many points
    if len(points) > n_points:
        indices = torch.randperm(len(points), device=device)[:n_points]
        points = points[indices]
    
    return points


def generate_lattice_bcc(
    n_points: int,
    spacing: float = 0.5,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate body-centered cubic lattice points.
    
    Args:
        n_points: Approximate number of points
        spacing: Lattice spacing
        device: Torch device
    
    Returns:
        Point cloud tensor [actual_n_points, 3]
    """
    # BCC has 2 atoms per unit cell
    grid_size = int(np.ceil((n_points / 2) ** (1/3)))
    
    x = torch.arange(grid_size, device=device, dtype=torch.float32) * spacing
    x = x - x.mean()
    
    # Corner atoms
    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
    corners = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    # Center atoms (offset by half spacing)
    x_center = x + spacing / 2
    x_center = x_center[:-1]  # Remove last to keep within bounds
    xx, yy, zz = torch.meshgrid(x_center, x_center, x_center, indexing='ij')
    centers = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    points = torch.cat([corners, centers], dim=0)
    
    # Subsample if needed
    if len(points) > n_points:
        indices = torch.randperm(len(points), device=device)[:n_points]
        points = points[indices]
    
    return points


def generate_lattice_fcc(
    n_points: int,
    spacing: float = 0.5,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate face-centered cubic lattice points.
    
    Args:
        n_points: Approximate number of points
        spacing: Lattice spacing
        device: Torch device
    
    Returns:
        Point cloud tensor [actual_n_points, 3]
    """
    # FCC has 4 atoms per unit cell
    grid_size = int(np.ceil((n_points / 4) ** (1/3)))
    
    x = torch.arange(grid_size, device=device, dtype=torch.float32) * spacing
    x = x - x.mean()
    
    # Corner atoms
    xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
    corners = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
    
    # Face centers
    half = spacing / 2
    face_atoms = []
    
    for i in range(grid_size):
        for j in range(grid_size):
            for k in range(grid_size):
                base = torch.tensor([x[i], x[j], x[k]], device=device)
                # XY face center
                face_atoms.append(base + torch.tensor([half, half, 0], device=device))
                # XZ face center
                face_atoms.append(base + torch.tensor([half, 0, half], device=device))
                # YZ face center
                face_atoms.append(base + torch.tensor([0, half, half], device=device))
    
    if face_atoms:
        faces = torch.stack(face_atoms, dim=0)
        points = torch.cat([corners, faces], dim=0)
    else:
        points = corners
    
    # Subsample if needed
    if len(points) > n_points:
        indices = torch.randperm(len(points), device=device)[:n_points]
        points = points[indices]
    
    return points


def generate_torus(
    n_points: int,
    major_radius: float = 1.0,
    minor_radius: float = 0.3,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points uniformly distributed on torus surface.
    
    Args:
        n_points: Number of points
        major_radius: Distance from center of tube to center of torus
        minor_radius: Radius of the tube
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    # Use rejection sampling for uniform distribution
    # Account for the fact that outer part of torus has more area
    
    points = []
    while len(points) < n_points:
        # Generate candidate points
        n_candidates = n_points * 2
        
        theta = torch.rand(n_candidates, device=device) * 2 * np.pi  # Angle around torus
        phi = torch.rand(n_candidates, device=device) * 2 * np.pi    # Angle around tube
        
        # Rejection sampling based on radius from axis
        # Points on outer edge should be accepted more often
        r = major_radius + minor_radius * torch.cos(phi)
        accept_prob = r / (major_radius + minor_radius)
        accept_mask = torch.rand(n_candidates, device=device) < accept_prob
        
        theta = theta[accept_mask]
        phi = phi[accept_mask]
        
        x = (major_radius + minor_radius * torch.cos(phi)) * torch.cos(theta)
        y = (major_radius + minor_radius * torch.cos(phi)) * torch.sin(theta)
        z = minor_radius * torch.sin(phi)
        
        pts = torch.stack([x, y, z], dim=-1)
        points.append(pts)
    
    points = torch.cat(points, dim=0)[:n_points]
    
    return points


def generate_helix(
    n_points: int,
    radius: float = 1.0,
    pitch: float = 0.5,
    turns: float = 3.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points along a helix.
    
    Args:
        n_points: Number of points
        radius: Helix radius
        pitch: Vertical distance per turn
        turns: Number of turns
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    t = torch.linspace(0, turns * 2 * np.pi, n_points, device=device)
    
    x = radius * torch.cos(t)
    y = radius * torch.sin(t)
    z = pitch * t / (2 * np.pi)
    
    # Center vertically
    z = z - z.mean()
    
    return torch.stack([x, y, z], dim=-1)


def generate_normal_distribution(
    n_points: int,
    std: float = 1.0,
    device: torch.device = None
) -> torch.Tensor:
    """
    Generate points from 3D normal distribution.
    
    Args:
        n_points: Number of points
        std: Standard deviation
        device: Torch device
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    return torch.randn(n_points, 3, device=device) * std


def generate_point_cloud(
    structure_type: StructureType,
    n_points: int,
    device: torch.device = None,
    **kwargs
) -> torch.Tensor:
    """
    Generate point cloud of specified type.
    
    Args:
        structure_type: Type of structure to generate
        n_points: Number of points
        device: Torch device
        **kwargs: Additional parameters for specific structure types
    
    Returns:
        Point cloud tensor [n_points, 3]
    """
    generators = {
        StructureType.SPHERE_SURFACE: generate_sphere_surface,
        StructureType.BALL_UNIFORM: generate_ball_uniform,
        StructureType.FLAT_PLANE: generate_flat_plane,
        StructureType.CUBE_SURFACE: generate_cube_surface,
        StructureType.CUBE_FILLED: generate_cube_filled,
        StructureType.LATTICE_SIMPLE_CUBIC: generate_lattice_simple_cubic,
        StructureType.LATTICE_BCC: generate_lattice_bcc,
        StructureType.LATTICE_FCC: generate_lattice_fcc,
        StructureType.TORUS: generate_torus,
        StructureType.HELIX: generate_helix,
        StructureType.NORMAL_DISTRIBUTION: generate_normal_distribution,
    }
    
    if structure_type not in generators:
        raise ValueError(f"Unknown structure type: {structure_type}")
    
    return generators[structure_type](n_points, device=device, **kwargs)


def center_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """
    Center point cloud at origin.
    
    Args:
        points: Point cloud [N, 3]
    
    Returns:
        Centered point cloud [N, 3]
    """
    return points - points.mean(dim=0, keepdim=True)


def normalize_point_cloud(points: torch.Tensor) -> torch.Tensor:
    """
    Normalize point cloud to fit in unit sphere.
    
    Args:
        points: Point cloud [N, 3]
    
    Returns:
        Normalized point cloud [N, 3]
    """
    points = center_point_cloud(points)
    max_dist = points.norm(dim=-1).max()
    return points / (max_dist + 1e-8)

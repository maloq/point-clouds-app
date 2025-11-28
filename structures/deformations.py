"""
Point cloud deformation functions.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from enum import Enum

import sys
sys.path.append('..')
from utils.rotation import euler_to_matrix, quaternion_to_matrix, random_rotation_matrix


class DeformationType(Enum):
    """Available deformation types."""
    ROTATION = "rotation"
    TRANSLATION = "translation"
    UNIFORM_SCALE = "uniform_scale"
    NON_UNIFORM_SCALE = "non_uniform_scale"
    SHEAR = "shear"
    SMOOTH_WARP = "smooth_warp"
    GAUSSIAN_NOISE = "gaussian_noise"
    SPHERIFICATION = "spherification"
    POINT_DROPOUT = "point_dropout"


def apply_rotation(
    points: torch.Tensor,
    rotation_matrix: Optional[torch.Tensor] = None,
    euler_angles: Optional[torch.Tensor] = None,
    quaternion: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Apply rotation to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        rotation_matrix: Rotation matrix [3, 3]
        euler_angles: Euler angles [3] in radians
        quaternion: Quaternion [4] (w, x, y, z)
    
    Returns:
        Rotated point cloud [N, 3]
    """
    if rotation_matrix is None:
        if euler_angles is not None:
            rotation_matrix = euler_to_matrix(euler_angles)
        elif quaternion is not None:
            rotation_matrix = quaternion_to_matrix(quaternion)
        else:
            raise ValueError("Must provide rotation_matrix, euler_angles, or quaternion")
    
    return points @ rotation_matrix.T


def apply_translation(
    points: torch.Tensor,
    translation: torch.Tensor
) -> torch.Tensor:
    """
    Apply translation to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        translation: Translation vector [3]
    
    Returns:
        Translated point cloud [N, 3]
    """
    return points + translation


def apply_uniform_scale(
    points: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    Apply uniform scaling to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        scale: Scale factor
    
    Returns:
        Scaled point cloud [N, 3]
    """
    return points * scale


def apply_non_uniform_scale(
    points: torch.Tensor,
    scale: torch.Tensor
) -> torch.Tensor:
    """
    Apply non-uniform scaling to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        scale: Scale factors [3] for x, y, z
    
    Returns:
        Scaled point cloud [N, 3]
    """
    return points * scale


def apply_shear(
    points: torch.Tensor,
    shear_matrix: Optional[torch.Tensor] = None,
    shear_params: Optional[Tuple[float, float, float, float, float, float]] = None
) -> torch.Tensor:
    """
    Apply shear transformation to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        shear_matrix: Full 3x3 shear matrix
        shear_params: (xy, xz, yx, yz, zx, zy) shear coefficients
    
    Returns:
        Sheared point cloud [N, 3]
    """
    if shear_matrix is None:
        if shear_params is None:
            raise ValueError("Must provide shear_matrix or shear_params")
        
        xy, xz, yx, yz, zx, zy = shear_params
        shear_matrix = torch.tensor([
            [1, xy, xz],
            [yx, 1, yz],
            [zx, zy, 1]
        ], dtype=points.dtype, device=points.device)
    
    return points @ shear_matrix.T


def apply_smooth_warp(
    points: torch.Tensor,
    frequency: float = 1.0,
    amplitude: float = 0.1,
    seed: Optional[int] = None
) -> torch.Tensor:
    """
    Apply smooth sinusoidal warping to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        frequency: Frequency of sinusoidal deformation
        amplitude: Amplitude of deformation
        seed: Random seed for phase offsets
    
    Returns:
        Warped point cloud [N, 3]
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    # Random phase offsets for each axis
    phase = torch.rand(3, 3, device=points.device) * 2 * np.pi
    
    # Apply sinusoidal deformation
    # Each axis is affected by all three input coordinates
    deformation = torch.zeros_like(points)
    
    for i in range(3):
        for j in range(3):
            deformation[:, i] += amplitude * torch.sin(
                frequency * points[:, j] * 2 * np.pi + phase[i, j]
            )
    
    return points + deformation


def apply_gaussian_noise(
    points: torch.Tensor,
    std: float = 0.01
) -> torch.Tensor:
    """
    Add Gaussian noise to point cloud.
    
    Args:
        points: Point cloud [N, 3]
        std: Standard deviation of noise
    
    Returns:
        Noisy point cloud [N, 3]
    """
    noise = torch.randn_like(points) * std
    return points + noise


def apply_spherification(
    points: torch.Tensor,
    intensity: float = 1.0,
    target_radius: float = 1.0
) -> torch.Tensor:
    """
    Morph point cloud toward a sphere with uniformly distributed points.
    
    Args:
        points: Point cloud [N, 3]
        intensity: Interpolation factor [0, 1] where 1 = full sphere
        target_radius: Radius of target sphere
    
    Returns:
        Deformed point cloud [N, 3]
    """
    # Center points
    center = points.mean(dim=0, keepdim=True)
    centered = points - center
    
    # Compute directions from center
    distances = centered.norm(dim=-1, keepdim=True)
    directions = centered / (distances + 1e-8)
    
    # Target: points on sphere surface
    sphere_points = directions * target_radius
    
    # Interpolate between original and sphere
    result = centered * (1 - intensity) + sphere_points * intensity
    
    return result + center


def apply_point_dropout(
    points: torch.Tensor,
    dropout_ratio: float = 0.1,
    replace: bool = True
) -> torch.Tensor:
    """
    Randomly drop points from point cloud.
    
    Args:
        points: Point cloud [N, 3]
        dropout_ratio: Fraction of points to drop
        replace: If True, duplicate remaining points to maintain count
    
    Returns:
        Point cloud with dropped points
    """
    n_points = points.shape[0]
    n_keep = int(n_points * (1 - dropout_ratio))
    
    indices = torch.randperm(n_points, device=points.device)[:n_keep]
    kept_points = points[indices]
    
    if replace and n_keep < n_points:
        # Randomly duplicate points to maintain original count
        n_duplicate = n_points - n_keep
        dup_indices = torch.randint(0, n_keep, (n_duplicate,), device=points.device)
        duplicated = kept_points[dup_indices]
        kept_points = torch.cat([kept_points, duplicated], dim=0)
    
    return kept_points


class Deformation:
    """
    Composable deformation class for point clouds.
    """
    
    def __init__(self, device: torch.device = None):
        """
        Initialize deformation.
        
        Args:
            device: Torch device
        """
        self.device = device
        self.transforms = []
    
    def reset(self):
        """Clear all transforms."""
        self.transforms = []
        return self
    
    def rotate(
        self,
        euler_angles: Optional[Tuple[float, float, float]] = None,
        rotation_matrix: Optional[torch.Tensor] = None,
        random: bool = False
    ):
        """Add rotation transform."""
        if random:
            R = random_rotation_matrix((), device=self.device)
        elif rotation_matrix is not None:
            R = rotation_matrix
        elif euler_angles is not None:
            angles = torch.tensor(euler_angles, dtype=torch.float32, device=self.device)
            R = euler_to_matrix(angles)
        else:
            raise ValueError("Must provide euler_angles, rotation_matrix, or set random=True")
        
        self.transforms.append(('rotation', R))
        return self
    
    def translate(self, translation: Tuple[float, float, float]):
        """Add translation transform."""
        t = torch.tensor(translation, dtype=torch.float32, device=self.device)
        self.transforms.append(('translation', t))
        return self
    
    def scale(self, factor: Union[float, Tuple[float, float, float]]):
        """Add scale transform."""
        if isinstance(factor, (int, float)):
            self.transforms.append(('uniform_scale', factor))
        else:
            s = torch.tensor(factor, dtype=torch.float32, device=self.device)
            self.transforms.append(('non_uniform_scale', s))
        return self
    
    def shear(self, params: Tuple[float, float, float, float, float, float]):
        """Add shear transform."""
        self.transforms.append(('shear', params))
        return self
    
    def warp(self, frequency: float = 1.0, amplitude: float = 0.1, seed: int = None):
        """Add smooth warp transform."""
        self.transforms.append(('warp', (frequency, amplitude, seed)))
        return self
    
    def noise(self, std: float = 0.01):
        """Add Gaussian noise."""
        self.transforms.append(('noise', std))
        return self
    
    def spherify(self, intensity: float = 0.5, target_radius: float = 1.0):
        """Add spherification transform."""
        self.transforms.append(('spherify', (intensity, target_radius)))
        return self
    
    def dropout(self, ratio: float = 0.1, replace: bool = True):
        """Add point dropout."""
        self.transforms.append(('dropout', (ratio, replace)))
        return self
    
    def apply(self, points: torch.Tensor) -> torch.Tensor:
        """
        Apply all transforms to point cloud.
        
        Args:
            points: Point cloud [N, 3]
        
        Returns:
            Transformed point cloud
        """
        result = points.clone()
        
        for transform_type, params in self.transforms:
            if transform_type == 'rotation':
                result = apply_rotation(result, rotation_matrix=params)
            elif transform_type == 'translation':
                result = apply_translation(result, params)
            elif transform_type == 'uniform_scale':
                result = apply_uniform_scale(result, params)
            elif transform_type == 'non_uniform_scale':
                result = apply_non_uniform_scale(result, params)
            elif transform_type == 'shear':
                result = apply_shear(result, shear_params=params)
            elif transform_type == 'warp':
                freq, amp, seed = params
                result = apply_smooth_warp(result, freq, amp, seed)
            elif transform_type == 'noise':
                result = apply_gaussian_noise(result, params)
            elif transform_type == 'spherify':
                intensity, radius = params
                result = apply_spherification(result, intensity, radius)
            elif transform_type == 'dropout':
                ratio, replace = params
                result = apply_point_dropout(result, ratio, replace)
        
        return result
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Apply transforms."""
        return self.apply(points)


class RandomDeformation:
    """
    Random deformation generator for data augmentation.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-np.pi, np.pi),
        translation_range: Tuple[float, float] = (-0.1, 0.1),
        scale_range: Tuple[float, float] = (0.8, 1.2),
        noise_range: Tuple[float, float] = (0.0, 0.02),
        spherify_range: Tuple[float, float] = (0.0, 0.0),
        warp_amplitude_range: Tuple[float, float] = (0.0, 0.0),
        enable_rotation: bool = True,
        enable_translation: bool = False,
        enable_scale: bool = False,
        enable_noise: bool = True,
        enable_spherify: bool = False,
        enable_warp: bool = False,
        device: torch.device = None
    ):
        """
        Initialize random deformation generator.
        
        Args:
            rotation_range: (min, max) rotation angle in radians
            translation_range: (min, max) translation
            scale_range: (min, max) scale factor
            noise_range: (min, max) noise standard deviation
            spherify_range: (min, max) spherification intensity
            warp_amplitude_range: (min, max) warp amplitude
            enable_*: Whether to enable each deformation type
            device: Torch device
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.noise_range = noise_range
        self.spherify_range = spherify_range
        self.warp_amplitude_range = warp_amplitude_range
        
        self.enable_rotation = enable_rotation
        self.enable_translation = enable_translation
        self.enable_scale = enable_scale
        self.enable_noise = enable_noise
        self.enable_spherify = enable_spherify
        self.enable_warp = enable_warp
        
        self.device = device
    
    def sample(self) -> Deformation:
        """
        Sample a random deformation.
        
        Returns:
            Deformation object with random transforms
        """
        deform = Deformation(device=self.device)
        
        if self.enable_rotation:
            angles = np.random.uniform(
                self.rotation_range[0],
                self.rotation_range[1],
                size=3
            )
            deform.rotate(euler_angles=tuple(angles))
        
        if self.enable_translation:
            trans = np.random.uniform(
                self.translation_range[0],
                self.translation_range[1],
                size=3
            )
            deform.translate(tuple(trans))
        
        if self.enable_scale:
            scale = np.random.uniform(
                self.scale_range[0],
                self.scale_range[1]
            )
            deform.scale(scale)
        
        if self.enable_warp and self.warp_amplitude_range[1] > 0:
            amp = np.random.uniform(
                self.warp_amplitude_range[0],
                self.warp_amplitude_range[1]
            )
            if amp > 0:
                deform.warp(frequency=1.0, amplitude=amp)
        
        if self.enable_spherify and self.spherify_range[1] > 0:
            intensity = np.random.uniform(
                self.spherify_range[0],
                self.spherify_range[1]
            )
            if intensity > 0:
                deform.spherify(intensity=intensity)
        
        if self.enable_noise and self.noise_range[1] > 0:
            std = np.random.uniform(
                self.noise_range[0],
                self.noise_range[1]
            )
            if std > 0:
                deform.noise(std=std)
        
        return deform
    
    def __call__(self, points: torch.Tensor) -> torch.Tensor:
        """Apply random deformation to points."""
        return self.sample()(points)

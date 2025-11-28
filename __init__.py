"""
Point cloud structure generation and deformation.
"""

from .generators import (
    StructureType,
    generate_sphere_surface,
    generate_ball_uniform,
    generate_flat_plane,
    generate_cube_surface,
    generate_cube_filled,
    generate_lattice_simple_cubic,
    generate_lattice_bcc,
    generate_lattice_fcc,
    generate_torus,
    generate_helix,
    generate_normal_distribution,
    generate_point_cloud,
    center_point_cloud,
    normalize_point_cloud,
)

from .deformations import (
    DeformationType,
    apply_rotation,
    apply_translation,
    apply_uniform_scale,
    apply_non_uniform_scale,
    apply_shear,
    apply_smooth_warp,
    apply_gaussian_noise,
    apply_spherification,
    apply_point_dropout,
    Deformation,
    RandomDeformation,
)

__all__ = [
    # Structure types
    'StructureType',
    # Generators
    'generate_sphere_surface',
    'generate_ball_uniform',
    'generate_flat_plane',
    'generate_cube_surface',
    'generate_cube_filled',
    'generate_lattice_simple_cubic',
    'generate_lattice_bcc',
    'generate_lattice_fcc',
    'generate_torus',
    'generate_helix',
    'generate_normal_distribution',
    'generate_point_cloud',
    'center_point_cloud',
    'normalize_point_cloud',
    # Deformation types
    'DeformationType',
    # Deformation functions
    'apply_rotation',
    'apply_translation',
    'apply_uniform_scale',
    'apply_non_uniform_scale',
    'apply_shear',
    'apply_smooth_warp',
    'apply_gaussian_noise',
    'apply_spherification',
    'apply_point_dropout',
    # Deformation classes
    'Deformation',
    'RandomDeformation',
]

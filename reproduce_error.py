
import torch
import numpy as np
import sys
import traceback

# Mock imports
sys.path.append('.')
from utils.rotation import euler_to_matrix, geodesic_distance, interpolate_rotations
from metrics.chamfer import chamfer_distance
from metrics.hausdorff import hausdorff_distance
from visualization.analysis import plot_geodesic_path
from structures.generators import generate_point_cloud, StructureType, normalize_point_cloud

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def test_geodesic_plot():
    try:
        struct_type = "sphere_surface"
        n_points = 500
        rot_x, rot_y, rot_z = 45, 45, 0
        
        print("Generating cloud...")
        cloud = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
        cloud = normalize_point_cloud(cloud)
        
        print("Calculating rotations...")
        rot_start = torch.eye(3, device=DEVICE)
        rot_end = euler_to_matrix(torch.tensor([np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)], device=DEVICE))
        
        print(f"rot_start shape: {rot_start.shape}, dtype: {rot_start.dtype}")
        print(f"rot_end shape: {rot_end.shape}, dtype: {rot_end.dtype}")
        
        # Target is cloud rotated by rot_end
        target = cloud @ rot_end.T
        
        # Metrics to plot
        metrics = {
            'Chamfer': chamfer_distance,
            'Hausdorff': hausdorff_distance
        }
        
        print("Calling plot_geodesic_path...")
        fig = plot_geodesic_path(cloud, target, rot_start, rot_end, steps=20, metrics=metrics)
        print("Success!")
        
    except Exception as e:
        print(f"Caught exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_geodesic_plot()

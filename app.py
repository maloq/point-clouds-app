"""
Main Gradio application for Point Cloud Metric Intuition & Alignment.
"""

import gradio as gr
import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
import time
import torch.nn as nn
from torch.func import functional_call, vmap

from structures.generators import StructureType, generate_point_cloud, center_point_cloud, normalize_point_cloud
from structures.deformations import Deformation, RandomDeformation
from metrics.chamfer import chamfer_distance, chamfer_distance_batched
from metrics.emd import emd_approximate
from metrics.hausdorff import hausdorff_distance
from metrics.gromov import gromov_wasserstein_distance
from metrics.spectral import spectral_distance
from visualization.point_cloud import render_cloud, render_overlay, render_heatmap, render_side_by_side
from visualization.analysis import plot_geodesic_path, plot_loss_landscape
from networks.training import AlignmentNetwork, Trainer
from utils.rotation import euler_to_matrix, random_rotation_matrix, geodesic_distance, kabsch_alignment
from utils.model_utils import get_params_vector, set_params_vector

# Global state
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_NETWORK = None
TRAINER = None
LOSS_HISTORY = []
BATCH_NETWORK = None
BATCH_TRAINER = None
BATCH_LOSS_HISTORY = []

def get_structure_types():
    return [t.value for t in StructureType]

# --- Tab 1: Intuition & Metrics ---

def generate_and_visualize(
    struct_type_a, n_points_a, 
    struct_type_b, n_points_b,
    rot_x, rot_y, rot_z,
    noise_std, spherify,
    show_overlay
):
    # Generate A
    cloud_a = generate_point_cloud(StructureType(struct_type_a), int(n_points_a), device=DEVICE)
    cloud_a = normalize_point_cloud(cloud_a)
    
    # Generate B (base)
    cloud_b = generate_point_cloud(StructureType(struct_type_b), int(n_points_b), device=DEVICE)
    cloud_b = normalize_point_cloud(cloud_b)
    
    # Deform B
    deform = Deformation(device=DEVICE)
    deform.rotate(euler_angles=(np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)))
    if noise_std > 0:
        deform.noise(std=noise_std)
    if spherify > 0:
        deform.spherify(intensity=spherify)
        
    cloud_b_deformed = deform(cloud_b)
    
    # Calculate metrics
    chamfer = chamfer_distance(cloud_a, cloud_b_deformed).item()
    try:
        emd = emd_approximate(cloud_a, cloud_b_deformed, method='sinkhorn').item()
    except:
        emd = -1.0
    hausdorff = hausdorff_distance(cloud_a, cloud_b_deformed).item()
    
    try:
        spectral = spectral_distance(cloud_a, cloud_b_deformed).item()
    except:
        spectral = -1.0
        
    metrics_text = f"""
    ### Metrics
    - **Chamfer Distance**: {chamfer:.6f}
    - **Sinkhorn Distance**: {emd:.6f}
    - **Hausdorff Distance**: {hausdorff:.6f}
    - **Spectral Distance**: {spectral:.6f}
    """
    
    # Visualization
    if show_overlay:
        fig = render_overlay(cloud_a, cloud_b_deformed, title="Overlay (A=Blue, B=Red)", height=800)
    else:
        fig = render_side_by_side(cloud_a, cloud_b_deformed, title="Side-by-Side View", height=800)
        
    return fig, metrics_text

def plot_geodesic_callback(struct_type, n_points, rot_x, rot_y, rot_z):
    cloud = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    cloud = normalize_point_cloud(cloud)
    
    rot_start = torch.eye(3, device=DEVICE)
    rot_end = euler_to_matrix(torch.tensor([np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)], device=DEVICE, dtype=torch.float32))
    
    # Target is cloud rotated by rot_end
    target = cloud @ rot_end.T
    
    # Metrics to plot
    metrics = {
        'Chamfer': chamfer_distance,
        'Hausdorff': hausdorff_distance
    }
    
    fig = plot_geodesic_path(cloud, target, rot_start, rot_end, steps=20, metrics=metrics)
    return fig

# --- Tab 2: Single Pair Training ---

def train_single_pair(
    struct_type, n_points,
    rot_x, rot_y, rot_z,
    noise, spherify,
    method, # New argument
    network_type, head_type,
    lr, epochs,
    loss_type
):
    global CURRENT_NETWORK, TRAINER, LOSS_HISTORY
    
    # Setup Data
    source = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    source = normalize_point_cloud(source)
    
    # Target is deformed source
    deform = Deformation(device=DEVICE)
    deform.rotate(euler_angles=(np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)))
    if noise > 0:
        deform.noise(std=noise)
    if spherify > 0:
        deform.spherify(intensity=spherify)
        
    target = deform(source)
    
    # Calculate global bounding box for stable visualization
    all_points = torch.cat([source, target], dim=0)
    min_val = all_points.min().item()
    max_val = all_points.max().item()
    padding = (max_val - min_val) * 0.1
    axis_range = (min_val - padding, max_val + padding)
    
    if method == "Kabsch Algorithm":
        # Algorithmic Alignment
        R, t, transformed = kabsch_alignment(source.unsqueeze(0), target.unsqueeze(0))
        transformed = transformed.squeeze(0)
        
        # Calculate loss (Chamfer)
        loss = chamfer_distance(transformed, target).item()
        
        fig = render_overlay(
            transformed, source, # Source is actually Target in the overlay logic usually? 
            # Wait, render_overlay(cloud1, cloud2). 
            # In previous code: render_overlay(transformed, source, ...)
            # Label1='Aligned Deformed', Label2='Original (Target)'
            # Here: transformed is aligned source. target is the target.
            # So we want to align Source to Target.
            # Transformed Source should match Target.
            transformed, target,
            color1='green', color2='red', 
            label1='Aligned Source', label2='Target', 
            title=f"Kabsch Alignment, Loss: {loss:.6f}",
            axis_range=axis_range,
            height=800
        )
        
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(x=[0], y=[loss], mode='markers', name='Loss'))
        loss_fig.update_layout(title="Alignment Loss (Kabsch)", yaxis_type="log")
        
        yield fig, loss_fig, f"Done! Kabsch Loss: {loss:.6f}"
        return

    # Neural Network Training
    # Setup Network
    CURRENT_NETWORK = AlignmentNetwork(encoder_type=network_type, head_type=head_type, num_points=int(n_points))
    TRAINER = Trainer(CURRENT_NETWORK, lr=lr, loss_type=loss_type, device=DEVICE)
    LOSS_HISTORY = []
    
    # Create batch of size 2 with noisy copy
    source_noisy = source + torch.randn_like(source) * 0.01
    target_noisy = target + torch.randn_like(target) * 0.01
    
    source_batch = torch.stack([source, source_noisy]) # [2, N, 3]
    target_batch = torch.stack([target, target_noisy])
    
    for i in range(int(epochs)):
        # Train: Align Deformed (target) to Original (source)
        # Wait, in original code: loss = TRAINER.train_step(target_batch, source_batch)
        # This means input=target_batch, target=source_batch.
        # So it was aligning Target -> Source?
        # "Train: Align Deformed (target) to Original (source)"
        # If so, transformed is transformed target.
        # Let's stick to the original logic for NN.
        
        loss = TRAINER.train_step(target_batch, source_batch)
        LOSS_HISTORY.append(loss)
        
        if i % 5 == 0 or i == epochs - 1:
            with torch.no_grad():
                outputs = CURRENT_NETWORK(target_batch, source_batch)
                transformed = outputs['transformed_source'][0]
                
            fig = render_overlay(
                transformed, source, 
                color1='green', color2='blue', 
                label1='Aligned Deformed', label2='Original (Target)', 
                title=f"Epoch {i}, Loss: {loss:.6f}",
                axis_range=axis_range,
                height=800
            )
            
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(y=LOSS_HISTORY, mode='lines', name='Loss'))
            loss_fig.update_layout(title="Training Loss", yaxis_type="log")
            
            yield fig, loss_fig, f"Epoch {i}/{epochs}, Loss: {loss:.6f}"
            
    yield fig, loss_fig, f"Done! Final Loss: {loss:.6f}"

# --- Tab 3: Batch Generalization ---

def train_batch_generalization(
    struct_type, n_points,
    max_rot, max_noise,
    network_type, head_type,
    lr, epochs, batch_size,
    loss_type
):
    global BATCH_NETWORK, BATCH_TRAINER, BATCH_LOSS_HISTORY
    
    # Setup Network
    BATCH_NETWORK = AlignmentNetwork(encoder_type=network_type, head_type=head_type, num_points=int(n_points))
    BATCH_TRAINER = Trainer(BATCH_NETWORK, lr=lr, loss_type=loss_type, device=DEVICE)
    BATCH_LOSS_HISTORY = []
    
    # Random deformation generator
    random_deform = RandomDeformation(
        rotation_range=(-np.radians(max_rot), np.radians(max_rot)),
        noise_range=(0, max_noise),
        device=DEVICE
    )
    
    # Base structure
    base_cloud = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    base_cloud = normalize_point_cloud(base_cloud)
    
    for i in range(int(epochs)):
        # Generate batch
        # For simplicity, we use the same base cloud but different deformations
        # Ideally we'd generate different structures if generalizing across shapes
        
        sources = []
        targets = []
        
        for _ in range(int(batch_size)):
            # Target is random deformation of base
            target = random_deform(base_cloud)
            
            # Source is base (or another random deformation?)
            # Task: "align arbitrary rotated and deformed point cloud to original one"
            # So Source = Deformed, Target = Original
            # Or Source = Original, Target = Deformed?
            # Let's say we want to align Source to Target.
            # Let Source be the Deformed one, Target be the Canonical one.
            
            source = random_deform(base_cloud)
            target = base_cloud # Canonical
            
            sources.append(source)
            targets.append(target)
            
        source_batch = torch.stack(sources)
        target_batch = torch.stack(targets)
        
        loss = BATCH_TRAINER.train_step(source_batch, target_batch)
        BATCH_LOSS_HISTORY.append(loss)
        
        if i % 5 == 0 or i == epochs - 1:
            loss_fig = go.Figure()
            loss_fig.add_trace(go.Scatter(y=BATCH_LOSS_HISTORY, mode='lines', name='Loss'))
            loss_fig.update_layout(title="Batch Training Loss", yaxis_type="log")
            
            yield loss_fig, f"Epoch {i}/{epochs}, Loss: {loss:.6f}"
            
    yield loss_fig, f"Done! Final Loss: {loss:.6f}"

def test_generalization(struct_type, n_points, max_rot, max_noise):
    if BATCH_NETWORK is None:
        return None, "Network not trained yet!"
        
    BATCH_NETWORK.eval()
    
    # Generate new sample
    base_cloud = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    base_cloud = normalize_point_cloud(base_cloud)
    
    random_deform = RandomDeformation(
        rotation_range=(-np.radians(max_rot), np.radians(max_rot)),
        noise_range=(0, max_noise),
        device=DEVICE
    )
    
    source = random_deform(base_cloud)
    target = base_cloud
    
    with torch.no_grad():
        outputs = BATCH_NETWORK(source.unsqueeze(0), target.unsqueeze(0))
        transformed = outputs['transformed_source'][0]
        
    fig = render_overlay(transformed, target, color1='green', color2='blue', label1='Aligned', label2='Canonical', title="Generalization Test")
    
    return fig, "Test Complete"

# --- Tab 4: Analysis ---

def run_analysis(struct_type, n_points, net_type, head_type, loss_type):
    # Setup Loss Function
    if loss_type == "chamfer":
        criterion = chamfer_distance
    elif loss_type == "sinkhorn":
        def criterion(x, y):
            # Sinkhorn wrapper for batch
            if x.dim() == 3:
                losses = []
                for i in range(x.shape[0]):
                    losses.append(emd_approximate(x[i], y[i], method='sinkhorn'))
                return torch.stack(losses).mean()
            return emd_approximate(x, y, method='sinkhorn')
    elif loss_type == "pairwise":
        # Placeholder for pairwise if not imported directly, but we can try to use a simple metric
        # or just fallback to chamfer if not available in this scope easily without importing
        # Let's assume we want to use the one from metrics.invariant if available, 
        # but for now let's stick to chamfer/sinkhorn/hausdorff for landscape
        # If user selected pairwise, we might need to import it.
        # Let's use chamfer as fallback or import it.
        from metrics.invariant import pairwise_distance_distribution_distance
        def criterion(x, y):
             if x.dim() == 3:
                losses = []
                for i in range(x.shape[0]):
                    losses.append(pairwise_distance_distribution_distance(x[i], y[i]))
                return torch.stack(losses).mean()
             return pairwise_distance_distribution_distance(x, y)
    else:
        criterion = chamfer_distance

    cloud = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    cloud = normalize_point_cloud(cloud)
    target = cloud.clone() # Optimal is identity
    
    # --- 1. 1D Loss Landscape (0 to 360 degrees) ---
    # Rotate around a random axis
    axis = torch.randn(3, device=DEVICE)
    axis = axis / axis.norm()
    
    angles_deg = np.linspace(0, 360, 100)
    losses_1d = []
    
    for deg in angles_deg:
        rad = np.radians(deg)
        # Axis-angle rotation
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=DEVICE)
        R = torch.eye(3, device=DEVICE) + torch.sin(torch.tensor(rad)) * K + (1 - torch.cos(torch.tensor(rad))) * (K @ K)
        
        rotated = cloud @ R.T
        loss = criterion(rotated, target).item()
        losses_1d.append(loss)
        
    fig_1d = go.Figure()
    fig_1d.add_trace(go.Scatter(x=angles_deg, y=losses_1d, mode='lines', name='Loss'))
    fig_1d.update_layout(
        title=f"1D Loss Landscape ({loss_type}) - Rotation 0-360 deg",
        xaxis_title="Rotation Angle (degrees)",
        yaxis_title="Loss",
        yaxis_type="log"
    )

    # --- 2. 2D Loss Landscape (Local Minima Check) ---
    # Grid of rotations around identity (or around a local minimum if we found one?)
    # Let's do a wider range to see basins
    N = 30
    extent = np.pi # +/- 180 degrees to see full landscape
    x = np.linspace(-extent, extent, N)
    y = np.linspace(-extent, extent, N)
    
    losses_2d = np.zeros((N, N))
    
    # Two random orthogonal axes
    axis1 = torch.randn(3, device=DEVICE)
    axis1 = axis1 / axis1.norm()
    # Make axis2 orthogonal to axis1
    axis2 = torch.randn(3, device=DEVICE)
    axis2 = axis2 - (axis2 @ axis1) * axis1
    axis2 = axis2 / axis2.norm()
    
    for i in range(N):
        for j in range(N):
            # Combined rotation: R = R(axis1, x[i]) @ R(axis2, y[j])
            # Construct R1
            rad1 = torch.tensor(x[i], device=DEVICE)
            K1 = torch.tensor([[0, -axis1[2], axis1[1]], [axis1[2], 0, -axis1[0]], [-axis1[1], axis1[0], 0]], device=DEVICE)
            R1 = torch.eye(3, device=DEVICE) + torch.sin(rad1) * K1 + (1 - torch.cos(rad1)) * (K1 @ K1)
            
            # Construct R2
            rad2 = torch.tensor(y[j], device=DEVICE)
            K2 = torch.tensor([[0, -axis2[2], axis2[1]], [axis2[2], 0, -axis2[0]], [-axis2[1], axis2[0], 0]], device=DEVICE)
            R2 = torch.eye(3, device=DEVICE) + torch.sin(rad2) * K2 + (1 - torch.cos(rad2)) * (K2 @ K2)
            
            R = R1 @ R2
            
            rotated = cloud @ R.T
            losses_2d[i, j] = criterion(rotated, target).item()
            
    fig_2d = plot_loss_landscape(0, losses_2d, extent, title=f"2D Loss Landscape ({loss_type})")
    
    return fig_1d, fig_2d

def run_network_landscape_analysis(
    struct_type, n_points,
    net_type, head_type, loss_type,
    lr, epochs, grid_res,
    rot_x, rot_y, rot_z,
    noise, spherify
):
    # Clean memory before starting
    torch.cuda.empty_cache()
    
    # Handle missing grid_res
    if grid_res is None:
        grid_res = 20
    
    # 1. Setup Data
    # Helper to generate batch
    def generate_batch(bsize):
        # Source is canonical
        s = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
        s = normalize_point_cloud(s)
        s_batch = s.unsqueeze(0).repeat(bsize, 1, 1)
        
        # Target is randomly deformed
        targets = []
        deform = Deformation(device=DEVICE)
        
        # Generate random parameters
        # Uniform random between -rot and +rot
        rx = (torch.rand(bsize, device=DEVICE) * 2 - 1) * np.radians(rot_x)
        ry = (torch.rand(bsize, device=DEVICE) * 2 - 1) * np.radians(rot_y)
        rz = (torch.rand(bsize, device=DEVICE) * 2 - 1) * np.radians(rot_z)
        
        # Apply deformations
        # Note: Deformation class accumulates transforms. We need fresh one or reset.
        # But Deformation is not batched for parameters (it takes tuple of floats).
        # We need to loop or use a batched deformation function if available.
        # Our Deformation class wraps simple functions.
        # Let's loop for simplicity (bsize is small, 32).
        
        for i in range(bsize):
            d = Deformation(device=DEVICE)
            d.rotate(euler_angles=(rx[i].item(), ry[i].item(), rz[i].item()))
            if noise > 0:
                # Random noise up to limit? Or fixed std?
                # User slider says "Noise". Let's assume it's the std.
                # Or if "batch fitting", maybe random noise up to that std?
                # Let's use fixed std for now as per slider.
                d.noise(std=noise)
            if spherify > 0:
                d.spherify(intensity=spherify)
            
            targets.append(d(s))
            
        t_batch = torch.stack(targets)
        return s_batch, t_batch

    # Generate Validation Batch (Fixed for Landscape)
    val_batch_size = 32
    val_source, val_target = generate_batch(val_batch_size)
    
    # Update n_points to actual generated size (e.g. for lattices)
    actual_n_points = val_source.shape[1]
    if actual_n_points != int(n_points):
        print(f"Warning: Requested {n_points} points, but generated {actual_n_points}. Updating network config.")
        n_points = actual_n_points
    
    # 2. Setup Network & Trainer
    model = AlignmentNetwork(encoder_type=net_type, head_type=head_type, num_points=int(n_points))
    trainer = Trainer(model, lr=lr, loss_type=loss_type, device=DEVICE)
    
    # 3. Train and Record Trajectory
    weight_history = []
    loss_history = []
    
    # Force initialization
    model(val_target, val_source)
    
    # Initial weights
    weight_history.append(get_params_vector(model).detach().cpu())
    
    train_batch_size = 32
    
    for i in range(int(epochs)):
        # Generate new training batch
        train_source, train_target = generate_batch(train_batch_size)
        
        loss = trainer.train_step(train_target, train_source) # Align target to source
        loss_history.append(loss)
        weight_history.append(get_params_vector(model).detach().cpu())
        
    # Stack weights: [T, D]
    W = torch.stack(weight_history)
    
    # 4. PCA on Trajectory
    # Center the weights
    W_mean = W.mean(dim=0)
    W_centered = W - W_mean
    
    # PCA using SVD
    # W_centered is [T, D]. If D >> T, better to do PCA on W W^T?
    # torch.pca_lowrank is good.
    U, S, V = torch.pca_lowrank(W_centered, q=2, center=False, niter=10)
    
    # Top 2 directions (V is [D, 2])
    v1 = V[:, 0]
    v2 = V[:, 1]
    
    # Project trajectory onto plane
    # Center on FINAL weights to make the minimum the focal point
    W_final = W[-1]
    W_centered = W - W_final
    proj_x = (W_centered @ v1).cpu().numpy()
    proj_y = (W_centered @ v2).cpu().numpy()
    
    # 5. Create Grid
    # Add padding to cover the whole trajectory
    x_min, x_max = proj_x.min(), proj_x.max()
    y_min, y_max = proj_y.min(), proj_y.max()
    
    # Make grid square and centered
    x_range = x_max - x_min
    y_range = y_max - y_min
    span = max(x_range, y_range) * 1.5 # 50% padding
    
    # Center is 0,0 because we centered W on W_final
    limit = span / 2
    x_grid = np.linspace(-limit, limit, int(grid_res))
    y_grid = np.linspace(-limit, limit, int(grid_res))
    
    xx, yy = np.meshgrid(x_grid, y_grid)
    loss_grid = np.zeros_like(xx)
    
    # 6. Evaluate Loss on Grid (Batched)
    # Prepare Grid Inputs
    grid_coords = np.stack([xx.flatten(), yy.flatten()], axis=1)
    grid_coords_t = torch.tensor(grid_coords, dtype=torch.float32, device=DEVICE)
    
    # Prepare Projection Vectors
    PCs = torch.stack([v1, v2]).to(DEVICE) # [2, D]
    # Base is final weights
    w_base = W_final.to(DEVICE)
    
    # Calculate ALL weight configurations at once -> MOVED TO CHUNK LOOP
    # all_weights = w_base + grid_coords_t @ PCs
    
    # Prepare model metadata for stateless call
    param_names = [n for n, _ in model.named_parameters() if _.requires_grad]
    param_shapes = [p.shape for p in model.parameters() if p.requires_grad]
    param_numels = [p.numel() for p in model.parameters() if p.requires_grad]
    
    def compute_loss_stateless(flat_params, target_b, source_b):
        # Reconstruct parameters dictionary
        params = {}
        idx = 0
        for name, shape, numel in zip(param_names, param_shapes, param_numels):
            params[name] = flat_params[idx : idx + numel].view(shape)
            idx += numel
            
        outputs = functional_call(model, params, (target_b, source_b))
        transformed = outputs['transformed_source']
        
        # Loss
        if loss_type == "chamfer":
            # Chamfer batched
            l = chamfer_distance_batched(transformed, source_b, bidirectional=True).mean()
        elif loss_type == "sinkhorn":
            # Sinkhorn might not be vmap-friendly if it has in-place ops or loops?
            # Our sinkhorn implementation uses loops. vmap might fail or be slow if not pure.
            # Let's try. If it fails, fallback to loop?
            # For now, assume chamfer for speed or simple loop inside vmap (which is fine).
            # But vmap runs the function for a *single* example (conceptually).
            # Here 'target_b' is a BATCH of points.
            # vmap is over the WEIGHTS dimension (dim 0 of flat_params).
            # So inside this function, flat_params is 1D. target_b is [B, N, 3].
            # The output of functional_call will be [B, N, 3].
            # chamfer_distance returns [B] or scalar?
            # Our chamfer_distance_batched returns [B]. .mean() -> scalar.
            # So this function returns a scalar loss for this set of weights.
            
            # Sinkhorn:
            losses = []
            for i in range(transformed.shape[0]):
                losses.append(emd_approximate(transformed[i], source_b[i], method='sinkhorn'))
            l = torch.stack(losses).mean()
        else:
            l = chamfer_distance(transformed, source_b, bidirectional=True).mean()
            
        return l

    # Vectorize
    # in_dims=(0, None, None) -> Split arg0 (weights) along dim 0.
    compute_loss_vmap = vmap(compute_loss_stateless, in_dims=(0, None, None))
    
    # Execute in chunks to avoid OOM
    # Use train mode to use batch statistics (consistent with trajectory)
    model.train()
    
    # Disable track_running_stats for vmap to avoid in-place updates of running_mean/var
    # We only want to USE batch stats, not update the global model stats during visualization
    original_track_running_stats = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            original_track_running_stats[name] = module.track_running_stats
            module.track_running_stats = False
            
    losses_list = []
    batch_size = 128
    
    try:
        with torch.no_grad():
            # Ensure batch inputs are on device
            target_batch = val_target.to(DEVICE)
            source_batch = val_source.to(DEVICE)
            
            num_points = grid_coords_t.shape[0]
            for i in range(0, num_points, batch_size):
                # Chunk grid coords
                grid_chunk = grid_coords_t[i : i + batch_size]
                
                # Generate weights for this chunk
                # (B, 2) @ (2, D) -> (B, D)
                chunk_weights = w_base + grid_chunk @ PCs
                
                # Compute loss
                chunk_losses = compute_loss_vmap(chunk_weights, target_batch, source_batch)
                losses_list.append(chunk_losses)
                
                # Clean up chunk memory
                del chunk_weights
                del chunk_losses
                torch.cuda.empty_cache()
                
            losses = torch.cat(losses_list)
    finally:
        # Restore track_running_stats
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if name in original_track_running_stats:
                    module.track_running_stats = original_track_running_stats[name]
        
    loss_grid = losses.view(int(grid_res), int(grid_res)).cpu().numpy()
    
    # Restore final weights (just in case)
    set_params_vector(model, W[-1].to(DEVICE))
    torch.cuda.empty_cache()
    

    
    # 7. Visualization
    fig = go.Figure()
    
    # Surface
    # Use log scale for better color distribution
    log_loss_grid = np.log10(loss_grid + 1e-10)
    fig.add_trace(go.Surface(x=x_grid, y=y_grid, z=log_loss_grid, surfacecolor=log_loss_grid, colorscale='Viridis', opacity=0.8, name='Loss Landscape'))
    
    # Trajectory
    # We need z-values for trajectory. We have loss_history, but that's training loss (with noise/dropout etc).
    # Ideally we re-evaluate loss on the exact path, or just use recorded loss.
    # Recorded loss is fine for visualization.
    # Note: loss_history has length epochs, weight_history has epochs+1.
    # Let's assume initial loss is high.
    
    # Re-evaluate trajectory losses for consistency with grid
    # Use train mode!
    model.train()
    traj_losses = []
    with torch.no_grad():
        for w_vec in weight_history:
            set_params_vector(model, w_vec.to(DEVICE))
            outputs = model(val_target, val_source)
            # Use batched chamfer and mean to match grid calculation
            l = chamfer_distance_batched(outputs['transformed_source'], val_source).mean().item()
            traj_losses.append(l)
            
    # Log scale for trajectory
    log_traj_losses = np.log10(np.array(traj_losses) + 1e-10)
    
    fig.add_trace(go.Scatter3d(
        x=proj_x, y=proj_y, z=log_traj_losses,
        mode='lines+markers',
        line=dict(color='white', width=3), # White line to connect
        marker=dict(
            size=5, 
            color=list(range(len(traj_losses))), # Color by step
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(title="Step")
        ),
        name='Optimization Path'
    ))
    
    fig.update_layout(
        title="Network Loss Landscape (PCA of Trajectory)",
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis=dict(title="Log10(Loss)")
        ),
        height=800
    )
    
    # 8. Aligned Cloud Visualization
    # Use final weights
    set_params_vector(model, W[-1].to(DEVICE))
    
    # Generate specific test case for visualization (using slider values)
    test_source = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    test_source = normalize_point_cloud(test_source)
    
    test_deform = Deformation(device=DEVICE)
    test_deform.rotate(euler_angles=(np.radians(rot_x), np.radians(rot_y), np.radians(rot_z)))
    if noise > 0:
        test_deform.noise(std=noise)
    if spherify > 0:
        test_deform.spherify(intensity=spherify)
    test_target = test_deform(test_source)
    
    # Batch it for model
    test_source_b = test_source.unsqueeze(0)
    test_target_b = test_target.unsqueeze(0)
    
    # Use eval mode for final visualization (inference)
    model.eval()
    with torch.no_grad():
        outputs = model(test_target_b, test_source_b)
        transformed = outputs['transformed_source'][0]
        
    fig_cloud = render_overlay(
        transformed, test_source,
        color1='green', color2='blue',
        label1='Aligned', label2='Target (Reference)',
        title="Final Alignment Result (Specific Test Case)"
    )
    
    # 9. Training Loss Plot
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=list(range(len(traj_losses))),
        y=traj_losses,
        mode='lines+markers',
        name='Training Loss'
    ))
    fig_loss.update_layout(
        title="Training Loss vs Steps",
        xaxis_title="Step",
        yaxis_title="Loss",
        yaxis_type="log"
    )
    
    return fig, fig_cloud, fig_loss

# --- UI Layout ---

with gr.Blocks(title="Point Cloud Intuition") as app:
    gr.Markdown("# Point Cloud Metric Intuition & Alignment")
    
    with gr.Tab("Intuition & Metrics"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Structure A (Reference)")
                struct_a = gr.Dropdown(get_structure_types(), value="sphere_surface", label="Type")
                n_a = gr.Number(value=500, label="Points")
                
                gr.Markdown("### Structure B (Deformed)")
                struct_b = gr.Dropdown(get_structure_types(), value="sphere_surface", label="Type")
                n_b = gr.Number(value=500, label="Points")
                
            with gr.Column():
                gr.Markdown("### Deformations (for B)")
                rot_x = gr.Slider(0, 360, value=0, label="Rot X")
                rot_y = gr.Slider(0, 360, value=0, label="Rot Y")
                rot_z = gr.Slider(0, 360, value=0, label="Rot Z")
                noise = gr.Slider(0, 0.1, value=0, label="Noise Std")
                spherify = gr.Slider(0, 1, value=0, label="Spherify")
                overlay_chk = gr.Checkbox(value=True, label="Overlay View")
                btn_update = gr.Button("Update Visualization")
                btn_geodesic = gr.Button("Plot Geodesic Metrics")
                

        with gr.Row():
            plot_view = gr.Plot(label="3D View")
            
        with gr.Row():
            with gr.Column():
                metrics_out = gr.Markdown("### Metrics\n...")
            with gr.Column():
                geodesic_plot = gr.Plot(label="Geodesic Metrics")
        
        inputs_list = [struct_a, n_a, struct_b, n_b, rot_x, rot_y, rot_z, noise, spherify, overlay_chk]
        
        # Auto-update on change
        for inp in inputs_list:
            inp.change(
                generate_and_visualize,
                inputs=inputs_list,
                outputs=[plot_view, metrics_out]
            )
            
        btn_update.click(
            generate_and_visualize,
            inputs=inputs_list,
            outputs=[plot_view, metrics_out]
        )
        
        btn_geodesic.click(
            plot_geodesic_callback,
            inputs=[struct_b, n_b, rot_x, rot_y, rot_z],
            outputs=[geodesic_plot]
        )
        
    with gr.Tab("Single Pair Training"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Setup")
                train_struct = gr.Dropdown(get_structure_types(), value="sphere_surface", label="Structure")
                train_n = gr.Number(value=512, label="Points")
                
                gr.Markdown("### Target Deformation")
                t_rot_x = gr.Slider(0, 180, value=45, label="Rot X")
                t_rot_y = gr.Slider(0, 180, value=45, label="Rot Y")
                t_rot_z = gr.Slider(0, 180, value=0, label="Rot Z")
                t_noise = gr.Slider(0, 0.05, value=0.0, label="Noise")
                t_spherify = gr.Slider(0, 1, value=0.0, label="Spherify")
                
            with gr.Column():
                gr.Markdown("### Method")
                method = gr.Dropdown(["Neural Network", "Kabsch Algorithm"], value="Neural Network", label="Alignment Method")
                
                gr.Markdown("### Network Config (if NN)")
                net_type = gr.Dropdown(["mlp", "pointnet", "gnn"], value="pointnet", label="Encoder")
                head_type = gr.Dropdown(["rotation", "flow", "combined"], value="rotation", label="Head")
                loss_type = gr.Dropdown(["chamfer", "sinkhorn", "pairwise"], value="chamfer", label="Loss Function")
                lr = gr.Number(value=0.001, label="Learning Rate")
                epochs = gr.Number(value=100, label="Epochs")
                btn_train = gr.Button("Train Alignment")
                
        with gr.Row():
            train_plot = gr.Plot(label="Alignment Visualization")
            
        with gr.Row():
            loss_plot = gr.Plot(label="Loss Curve")
            
        train_status = gr.Textbox(label="Status")
        
        btn_train.click(
            train_single_pair,
            inputs=[train_struct, train_n, t_rot_x, t_rot_y, t_rot_z, t_noise, t_spherify, method, net_type, head_type, lr, epochs, loss_type],
            outputs=[train_plot, loss_plot, train_status]
        )

    with gr.Tab("Batch Generalization"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training Config")
                batch_struct = gr.Dropdown(get_structure_types(), value="sphere_surface", label="Structure")
                batch_n = gr.Number(value=512, label="Points")
                batch_max_rot = gr.Slider(0, 180, value=90, label="Max Rotation")
                batch_max_noise = gr.Slider(0, 0.05, value=0.01, label="Max Noise")
                
            with gr.Column():
                gr.Markdown("### Network")
                batch_net_type = gr.Dropdown(["mlp", "pointnet", "gnn"], value="pointnet", label="Encoder")
                batch_head_type = gr.Dropdown(["rotation", "flow", "combined"], value="rotation", label="Head")
                batch_loss_type = gr.Dropdown(["chamfer", "sinkhorn", "pairwise"], value="chamfer", label="Loss Function")
                batch_lr = gr.Number(value=0.001, label="Learning Rate")
                batch_epochs = gr.Number(value=50, label="Epochs")
                batch_size = gr.Number(value=8, label="Batch Size")
                btn_batch_train = gr.Button("Train Batch")
                
        with gr.Row():
            batch_loss_plot = gr.Plot(label="Batch Loss")
            batch_status = gr.Textbox(label="Status")
            
        gr.Markdown("### Testing")
        btn_test = gr.Button("Generate New Random Sample")
        test_plot = gr.Plot(label="Test Result")
        test_status = gr.Textbox(label="Test Status")
        
        btn_batch_train.click(
            train_batch_generalization,
            inputs=[batch_struct, batch_n, batch_max_rot, batch_max_noise, batch_net_type, batch_head_type, batch_lr, batch_epochs, batch_size, batch_loss_type],
            outputs=[batch_loss_plot, batch_status]
        )
        
        btn_test.click(
            test_generalization,
            inputs=[batch_struct, batch_n, batch_max_rot, batch_max_noise],
            outputs=[test_plot, test_status]
        )

    with gr.Tab("Analysis"):
        gr.Markdown("### Loss Landscape Analysis")
        gr.Markdown("Visualize the loss landscape to check for local minima, especially for symmetric/crystal-like structures.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Structure Config")
                analysis_struct = gr.Dropdown(get_structure_types(), value="lattice_fcc", label="Structure")
                analysis_n = gr.Number(value=128, label="Points")
                
            with gr.Column():
                gr.Markdown("### Network / Loss Config")
                # Adding same controls as training tab
                analysis_net_type = gr.Dropdown(["mlp", "pointnet", "gnn"], value="pointnet", label="Encoder")
                analysis_head_type = gr.Dropdown(["rotation", "flow", "combined"], value="rotation", label="Head")
                analysis_loss_type = gr.Dropdown(["chamfer", "sinkhorn", "pairwise"], value="chamfer", label="Loss Function")
                
                btn_analysis = gr.Button("Run Rotation Landscape")
            
        with gr.Row():
            landscape_1d_plot = gr.Plot(label="1D Loss Landscape (360 deg)")
            landscape_2d_plot = gr.Plot(label="2D Loss Landscape")
            
        btn_analysis.click(
            run_analysis,
            inputs=[analysis_struct, analysis_n, analysis_net_type, analysis_head_type, analysis_loss_type],
            outputs=[landscape_1d_plot, landscape_2d_plot]
        )
        
        gr.Markdown("---")
        gr.Markdown("### Network Optimization Landscape")
        gr.Markdown("Train a network, perform PCA on the weight trajectory, and visualize the loss landscape around the path.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Training Config")
                net_lr = gr.Number(value=0.001, label="Learning Rate")
                net_epochs = gr.Number(value=50, label="Epochs")
                grid_res = gr.Number(value=20, label="Grid Resolution")
                
                gr.Markdown("### Target Deformation")
                net_rot_x = gr.Slider(0, 180, value=45, label="Rot X")
                net_rot_y = gr.Slider(0, 180, value=45, label="Rot Y")
                net_rot_z = gr.Slider(0, 180, value=0, label="Rot Z")
                net_noise = gr.Slider(0, 0.05, value=0.0, label="Noise")
                net_spherify = gr.Slider(0, 1, value=0.0, label="Spherify")
                
                btn_net_analysis = gr.Button("Run Network Analysis")
                
            with gr.Column():
                net_landscape_plot = gr.Plot(label="Network Loss Landscape (3D)")
                with gr.Row():
                    net_cloud_plot = gr.Plot(label="Final Aligned Cloud")
                    net_loss_plot = gr.Plot(label="Training Loss")
                
        btn_net_analysis.click(
            run_network_landscape_analysis,
            inputs=[
                analysis_struct, analysis_n, analysis_net_type, analysis_head_type, analysis_loss_type, 
                net_lr, net_epochs, grid_res,
                net_rot_x, net_rot_y, net_rot_z, net_noise, net_spherify
            ],
            outputs=[net_landscape_plot, net_cloud_plot, net_loss_plot]
        )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)

"""
Main Gradio application for Point Cloud Metric Intuition & Alignment.
"""

import gradio as gr
import torch
import numpy as np
import plotly.graph_objects as go
from typing import Dict, Any, List
import time

from structures.generators import StructureType, generate_point_cloud, center_point_cloud, normalize_point_cloud
from structures.deformations import Deformation, RandomDeformation
from metrics.chamfer import chamfer_distance
from metrics.emd import emd_approximate
from metrics.hausdorff import hausdorff_distance
from metrics.gromov import gromov_wasserstein_distance
from metrics.spectral import spectral_distance
from visualization.point_cloud import render_cloud, render_overlay, render_heatmap, render_side_by_side
from visualization.analysis import plot_geodesic_path, plot_loss_landscape
from networks.training import AlignmentNetwork, Trainer
from utils.rotation import euler_to_matrix, random_rotation_matrix, geodesic_distance

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
    
    # Setup Network
    CURRENT_NETWORK = AlignmentNetwork(encoder_type=network_type, head_type=head_type, num_points=int(n_points))
    TRAINER = Trainer(CURRENT_NETWORK, lr=lr, loss_type=loss_type, device=DEVICE)
    LOSS_HISTORY = []
    
    # Create batch of size 2 with noisy copy
    source_noisy = source + torch.randn_like(source) * 0.01
    target_noisy = target + torch.randn_like(target) * 0.01
    
    source_batch = torch.stack([source, source_noisy]) # [2, N, 3]
    target_batch = torch.stack([target, target_noisy])
    
    # Calculate global bounding box for stable visualization
    all_points = torch.cat([source, target], dim=0)
    min_val = all_points.min().item()
    max_val = all_points.max().item()
    # Add some padding
    padding = (max_val - min_val) * 0.1
    axis_range = (min_val - padding, max_val + padding)
    
    for i in range(int(epochs)):
        # Train: Align Deformed (target) to Original (source)
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

def run_analysis(struct_type, n_points):
    # 1. Loss Landscape
    # We need a trained network or just a metric landscape?
    # Prompt says: "Perturb rotation parameters in 2 random directions from optimal, calculate loss, plot 2D Heatmap."
    
    cloud = generate_point_cloud(StructureType(struct_type), int(n_points), device=DEVICE)
    cloud = normalize_point_cloud(cloud)
    target = cloud.clone() # Optimal is identity
    
    # Grid of rotations around identity
    N = 20
    extent = 0.5 # radians
    x = np.linspace(-extent, extent, N)
    y = np.linspace(-extent, extent, N)
    
    losses = np.zeros((N, N))
    
    # Random axes
    axis1 = torch.randn(3, device=DEVICE)
    axis1 = axis1 / axis1.norm()
    axis2 = torch.randn(3, device=DEVICE)
    axis2 = axis2 / axis2.norm()
    
    for i in range(N):
        for j in range(N):
            # Construct rotation from two angles
            # R = R(axis1, x[i]) @ R(axis2, y[j])
            # Approximation for small angles
            rot_vec = axis1 * x[i] + axis2 * y[j]
            angle = rot_vec.norm()
            if angle < 1e-6:
                R = torch.eye(3, device=DEVICE)
            else:
                axis = rot_vec / angle
                # Axis-angle to matrix
                K = torch.tensor([
                    [0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]
                ], device=DEVICE)
                R = torch.eye(3, device=DEVICE) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
                
            rotated = cloud @ R.T
            losses[i, j] = chamfer_distance(rotated, target).item()
            
    fig_landscape = plot_loss_landscape(0, losses, extent, title="Chamfer Loss Landscape (Rotation)")
    
    # 2. Multi-metric convergence
    # Simulate a convergence path (e.g. simple gradient descent on rotation)
    # Or just plot metrics vs rotation angle
    
    rot_start = random_rotation_matrix((), device=DEVICE)
    rot_end = torch.eye(3, device=DEVICE)
    
    metrics = {
        'Chamfer': chamfer_distance,
        'Hausdorff': hausdorff_distance,
        'Spectral': spectral_distance
    }
    
    fig_convergence = plot_geodesic_path(cloud, target, rot_start, rot_end, steps=30, metrics=metrics)
    fig_convergence.update_layout(title="Metric Convergence (Geodesic Path to Identity)")
    
    return fig_landscape, fig_convergence

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
                gr.Markdown("### Network")
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
            inputs=[train_struct, train_n, t_rot_x, t_rot_y, t_rot_z, t_noise, t_spherify, net_type, head_type, lr, epochs, loss_type],
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
        gr.Markdown("### Loss Landscape & Metric Convergence")
        with gr.Row():
            analysis_struct = gr.Dropdown(get_structure_types(), value="sphere_surface", label="Structure")
            analysis_n = gr.Number(value=512, label="Points")
            btn_analysis = gr.Button("Run Analysis")
            
        with gr.Row():
            landscape_plot = gr.Plot(label="Loss Landscape")
            convergence_plot = gr.Plot(label="Metric Convergence")
            
        btn_analysis.click(
            run_analysis,
            inputs=[analysis_struct, analysis_n],
            outputs=[landscape_plot, convergence_plot]
        )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)


import torch
import torch.nn as nn
import numpy as np
from torch.func import functional_call, vmap
import sys

sys.path.append('.')
from networks.training import AlignmentNetwork
from metrics.chamfer import chamfer_distance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def reproduce():
    print(f"Using device: {DEVICE}")
    
    # Setup simple data
    B, N = 2, 128
    s = torch.randn(1, N, 3, device=DEVICE)
    t = torch.randn(1, N, 3, device=DEVICE)
    # Add noise to second element to avoid BN collapse
    s_noisy = s + 0.01 * torch.randn_like(s)
    t_noisy = t + 0.01 * torch.randn_like(t)
    source = torch.cat([s, s_noisy], dim=0)
    target = torch.cat([t, t_noisy], dim=0)
    
    # Setup model
    model = AlignmentNetwork(encoder_type="pointnet", head_type="rotation", num_points=N)
    model.to(DEVICE)
    model.train()
    
    # Disable track_running_stats for vmap
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.track_running_stats = False
            
    # Force init
    _ = model(source, target)
    
    # Get base weights
    param_names = [n for n, _ in model.named_parameters() if _.requires_grad]
    param_shapes = [p.shape for p in model.parameters() if p.requires_grad]
    param_numels = [p.numel() for p in model.parameters() if p.requires_grad]
    
    params_vec = []
    for p in model.parameters():
        if p.requires_grad:
            params_vec.append(p.view(-1))
    w_base = torch.cat(params_vec)
    
    print(f"Total params: {w_base.numel()}")
    
    # Create random direction
    v1 = torch.randn_like(w_base)
    v1 = v1 / v1.norm()
    
    # Create grid of weights along v1
    # We expect loss to change as we move along v1
    steps = 10
    coeffs = torch.linspace(-1.0, 1.0, steps, device=DEVICE)
    
    # Weights: [steps, D]
    # w_i = w_base + coeff * v1
    weights_batch = w_base.unsqueeze(0) + coeffs.unsqueeze(1) * v1.unsqueeze(0)
    
    print(f"Weights batch shape: {weights_batch.shape}")
    
    # Define stateless loss function
    def compute_loss_stateless(flat_params, src, tgt):
        # Reconstruct parameters dictionary
        params = {}
        idx = 0
        for name, shape, numel in zip(param_names, param_shapes, param_numels):
            params[name] = flat_params[idx : idx + numel].view(shape)
            idx += numel
            
        # Functional call
        # Note: we pass src, tgt as args
        outputs = functional_call(model, params, (src, tgt))
        transformed = outputs['transformed_source']
        
        # Loss
        l = chamfer_distance(transformed, tgt, bidirectional=True).mean()
        return l

    # Vectorize
    compute_loss_vmap = vmap(compute_loss_stateless, in_dims=(0, None, None))
    
    # Run
    print("Running vmap...")
    losses = compute_loss_vmap(weights_batch, source, target)
    
    print("Losses:")
    print(losses)
    
    if losses.std() < 1e-6:
        print("FAIL: Losses are constant!")
    else:
        print("SUCCESS: Losses vary.")

if __name__ == "__main__":
    reproduce()

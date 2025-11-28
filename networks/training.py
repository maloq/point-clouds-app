"""
Training logic for point cloud alignment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Dict, Any, Tuple
import sys
import os

sys.path.append('..')
from networks.encoders import MLPEncoder, PointNetEncoder, GNNEncoder
from networks.heads import RotationHead, FlowHead, CombinedHead
from metrics.chamfer import chamfer_distance_batched
from metrics.emd import sinkhorn_distance
from metrics.invariant import pairwise_distance_distribution_distance

class AlignmentNetwork(nn.Module):
    """
    End-to-end network for point cloud alignment.
    """
    def __init__(
        self,
        encoder_type: str = "pointnet",
        head_type: str = "rotation",
        num_points: int = 1024,
        latent_dim: int = 512
    ):
        super().__init__()
        self.head_type = head_type
        
        # Encoder
        if encoder_type == "mlp":
            self.encoder = MLPEncoder(num_points=num_points, latent_dim=latent_dim)
        elif encoder_type == "pointnet":
            self.encoder = PointNetEncoder(latent_dim=latent_dim)
        elif encoder_type == "gnn":
            self.encoder = GNNEncoder(latent_dim=latent_dim)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        # Head
        if head_type == "rotation":
            self.head = RotationHead(latent_dim=latent_dim)
        elif head_type == "flow":
            self.head = FlowHead(latent_dim=latent_dim, num_points=num_points)
        elif head_type == "combined":
            self.head = CombinedHead(latent_dim=latent_dim, num_points=num_points)
        else:
            raise ValueError(f"Unknown head type: {head_type}")
            
    def forward(self, source: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            source: Source point cloud [B, N, 3]
            target: Target point cloud [B, N, 3]
            
        Returns:
            Dictionary containing predictions
        """
        # Concatenate source and target features or process separately?
        # Standard approach: Encode both, concatenate latent codes
        # Or: Encode source, condition on target?
        # For simplicity: Siamese network, encode both, concat, then predict transform for source
        
        z_source = self.encoder(source)
        z_target = self.encoder(target)
        
        z_combined = torch.cat([z_source, z_target], dim=1)  # [B, 2*latent_dim]
        
        # We need to adjust head input dim if we concat
        # But heads are initialized with latent_dim.
        # Let's add a projection layer here or assume heads handle it?
        # The heads are initialized with latent_dim.
        # So we should project z_combined back to latent_dim
        
        # Hack: Re-initialize heads with 2*latent_dim if needed, or project here.
        # Better: Project here.
        if not hasattr(self, 'projection'):
            self.projection = nn.Linear(2 * z_source.shape[1], z_source.shape[1]).to(source.device)
            
        z = self.projection(z_combined)
        z = torch.relu(z)
        
        outputs = {}
        
        if self.head_type == "rotation":
            rotation = self.head(z)
            transformed_source = torch.bmm(source, rotation.transpose(1, 2))
            outputs['rotation'] = rotation
            outputs['transformed_source'] = transformed_source
            
        elif self.head_type == "flow":
            flow = self.head(z)
            transformed_source = source + flow
            outputs['flow'] = flow
            outputs['transformed_source'] = transformed_source
            
        elif self.head_type == "combined":
            rotation, flow = self.head(z)
            # Apply flow then rotation? Or rotation then flow?
            # Plan says: "First predict flow, apply it, then predict rotation."
            
            # Apply flow
            source_flowed = source + flow
            
            # Apply rotation
            transformed_source = torch.bmm(source_flowed, rotation.transpose(1, 2))
            
            outputs['rotation'] = rotation
            outputs['flow'] = flow
            outputs['transformed_source'] = transformed_source
            
        return outputs

class Trainer:
    """
    Trainer for the alignment network.
    """
    def __init__(
        self,
        model: AlignmentNetwork,
        lr: float = 1e-3,
        loss_type: str = "chamfer",
        device: torch.device = None
    ):
        self.model = model
        self.loss_type = loss_type
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_history = []
        
    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> float:
        """
        Perform one training step.
        """
        self.model.train()
        source = source.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad()
        
        outputs = self.model(source, target)
        transformed_source = outputs['transformed_source']
        
        # Loss Calculation
        if self.loss_type == "chamfer":
            loss = chamfer_distance_batched(transformed_source, target, bidirectional=True).mean()
            
        elif self.loss_type == "sinkhorn":
            # Sinkhorn doesn't support batching natively in our impl, loop over batch
            losses = []
            for i in range(transformed_source.shape[0]):
                l = sinkhorn_distance(transformed_source[i], target[i], epsilon=0.1, max_iter=50)
                losses.append(l)
            loss = torch.stack(losses).mean()
            
        elif self.loss_type == "pairwise":
            # Pairwise distribution distance
            losses = []
            for i in range(transformed_source.shape[0]):
                l = pairwise_distance_distribution_distance(transformed_source[i], target[i], n_bins=50, distance_type='l2')
                losses.append(l)
            loss = torch.stack(losses).mean()
            
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Regularization for flow (minimize magnitude)
        if 'flow' in outputs:
            flow_loss = torch.mean(outputs['flow'] ** 2)
            loss += 0.1 * flow_loss
            
        # Regularization for rotation (orthogonality is handled by 6D rep, but maybe identity bias?)
        
        loss.backward()
        self.optimizer.step()
        
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        return loss_val
        
    def save_checkpoint(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history
        }, path)
        
    def load_checkpoint(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss_history = checkpoint['loss_history']
            return True
        return False

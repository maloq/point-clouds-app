"""
Point cloud encoders.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class MLPEncoder(nn.Module):
    """
    Simple MLP encoder.
    Flattens the point cloud and passes it through dense layers.
    Note: Not permutation invariant!
    """
    def __init__(self, num_points: int = 1024, input_dim: int = 3, latent_dim: int = 512):
        super().__init__()
        self.flatten_dim = num_points * input_dim
        
        self.net = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point clouds [B, N, 3]
            
        Returns:
            Latent code [B, latent_dim]
        """
        B, N, D = x.shape
        x = x.view(B, -1)
        return self.net(x)

class PointNetEncoder(nn.Module):
    """
    PointNet encoder.
    Shared MLP -> MaxPool -> MLP.
    Permutation invariant.
    """
    def __init__(self, input_dim: int = 3, latent_dim: int = 512):
        super().__init__()
        
        # Point-wise MLP
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Global MLP
        self.mlp2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point clouds [B, N, 3]
            
        Returns:
            Latent code [B, latent_dim]
        """
        B, N, D = x.shape
        
        # Apply MLP1 to each point
        # Reshape to [B*N, D] for Linear layers or use Conv1d
        # Using Linear here for clarity
        x = x.view(-1, D)
        x = self.mlp1(x)
        x = x.view(B, N, -1)  # [B, N, 1024]
        
        # Max pooling over points
        x = torch.max(x, dim=1)[0]  # [B, 1024]
        
        # Global MLP
        x = self.mlp2(x)  # [B, latent_dim]
        
        return x

class GNNEncoder(nn.Module):
    """
    Simple GNN encoder using k-NN graph and max pooling.
    """
    def __init__(self, input_dim: int = 3, latent_dim: int = 512, k: int = 20):
        super().__init__()
        self.k = k
        
        self.conv1 = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Linear(64 * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        
    def get_knn_graph(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Get k-nearest neighbors.
        Returns: [B, N, k, D]
        """
        B, N, D = x.shape
        
        # Pairwise distances
        dist = torch.cdist(x, x)  # [B, N, N]
        
        # Get k nearest neighbors
        idx = dist.topk(k=k, dim=-1, largest=False)[1]  # [B, N, k]
        
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, k)
        
        # Gather neighbors
        # [B, N, k, D]
        neighbors = x[batch_idx, idx]
        
        return neighbors
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point clouds [B, N, 3]
            
        Returns:
            Latent code [B, latent_dim]
        """
        B, N, D = x.shape
        
        # Layer 1
        neighbors = self.get_knn_graph(x, self.k)  # [B, N, k, D]
        
        # Edge features: [x_i, x_j - x_i]
        x_expanded = x.unsqueeze(2).expand(B, N, self.k, D)
        edge_feat = torch.cat([x_expanded, neighbors - x_expanded], dim=-1)  # [B, N, k, 2*D]
        
        # Apply conv1
        edge_feat = edge_feat.view(B * N * self.k, -1)
        edge_feat = self.conv1(edge_feat)
        edge_feat = edge_feat.view(B, N, self.k, -1)
        
        # Max pool over neighbors
        x1 = edge_feat.max(dim=2)[0]  # [B, N, 64]
        
        # Layer 2
        neighbors = self.get_knn_graph(x1, self.k)
        x1_expanded = x1.unsqueeze(2).expand(B, N, self.k, -1)
        edge_feat = torch.cat([x1_expanded, neighbors - x1_expanded], dim=-1)
        
        edge_feat = edge_feat.view(B * N * self.k, -1)
        edge_feat = self.conv2(edge_feat)
        edge_feat = edge_feat.view(B, N, self.k, -1)
        
        x2 = edge_feat.max(dim=2)[0]  # [B, N, 128]
        
        # Global pooling
        x_global = x2.max(dim=1)[0]  # [B, 128]
        
        # Final MLP
        out = self.mlp(x_global)
        
        return out

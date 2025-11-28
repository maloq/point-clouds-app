"""
Network heads for predicting rotation and flow.
"""

import torch
from typing import Tuple
import torch.nn as nn
import sys
sys.path.append('..')
from utils.rotation import rotation_6d_to_matrix

class RotationHead(nn.Module):
    """
    Predicts a 3x3 rotation matrix from a latent code.
    Uses 6D continuous rotation representation.
    """
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6D rotation representation
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent code [B, latent_dim]
            
        Returns:
            Rotation matrix [B, 3, 3]
        """
        d6 = self.net(x)
        return rotation_6d_to_matrix(d6)

class FlowHead(nn.Module):
    """
    Predicts per-point flow (displacement) from a latent code.
    """
    def __init__(self, latent_dim: int = 512, num_points: int = 1024):
        super().__init__()
        self.num_points = num_points
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, num_points * 3)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent code [B, latent_dim]
            
        Returns:
            Flow vectors [B, N, 3]
        """
        B = x.shape[0]
        flow = self.net(x)
        return flow.view(B, self.num_points, 3)

class CombinedHead(nn.Module):
    """
    Predicts both flow and rotation.
    """
    def __init__(self, latent_dim: int = 512, num_points: int = 1024):
        super().__init__()
        self.rotation_head = RotationHead(latent_dim)
        self.flow_head = FlowHead(latent_dim, num_points)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Latent code [B, latent_dim]
            
        Returns:
            rotation: [B, 3, 3]
            flow: [B, N, 3]
        """
        rotation = self.rotation_head(x)
        flow = self.flow_head(x)
        return rotation, flow

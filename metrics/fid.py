"""
Fréchet Inception Distance (FID) for point clouds.
Uses a randomly initialized PointNet to extract features and compares their statistics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from scipy import linalg

class RandomPointNet(nn.Module):
    """
    Randomly initialized PointNet for feature extraction.
    We don't train this, just use it to project point clouds to a high-dim space.
    """
    def __init__(self, input_dim: int = 3, feature_dim: int = 1024):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Point clouds [B, N, 3]
            
        Returns:
            Global features [B, feature_dim]
        """
        # Apply MLP to each point
        x = self.mlp1(x)  # [B, N, feature_dim]
        
        # Max pooling over points
        x = torch.max(x, dim=1)[0]  # [B, feature_dim]
        
        return x

def calculate_activation_statistics(
    files_or_tensor: torch.Tensor,
    model: nn.Module,
    batch_size: int = 50,
    device: torch.device = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate mean and covariance of features.
    
    Args:
        files_or_tensor: Point clouds [B, N, 3]
        model: Feature extractor
        batch_size: Batch size for processing
        device: Torch device
        
    Returns:
        mu: Mean of features
        sigma: Covariance of features
    """
    model.eval()
    if device is not None:
        model.to(device)
        files_or_tensor = files_or_tensor.to(device)
        
    n_samples = files_or_tensor.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    features_list = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            batch = files_or_tensor[start:end]
            
            feat = model(batch)
            features_list.append(feat.cpu().numpy())
            
    features = np.concatenate(features_list, axis=0)
    
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma

def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6
) -> float:
    """
    Numpy implementation of the Fréchet Distance.
    The Fréchet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))
            
    Args:
        mu1: Mean of first distribution
        sigma1: Covariance of first distribution
        mu2: Mean of second distribution
        sigma2: Covariance of second distribution
        eps: Small constant for numerical stability
        
    Returns:
        Fréchet distance
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    
    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
        
    tr_covmean = np.trace(covmean)
    
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def compute_fid(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 50,
    device: torch.device = None
) -> float:
    """
    Compute FID between two sets of point clouds.
    
    Args:
        x: Point clouds [B, N, 3]
        y: Point clouds [B, N, 3]
        batch_size: Batch size
        device: Torch device
        
    Returns:
        FID score
    """
    model = RandomPointNet()
    
    mu1, sigma1 = calculate_activation_statistics(x, model, batch_size, device)
    mu2, sigma2 = calculate_activation_statistics(y, model, batch_size, device)
    
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid

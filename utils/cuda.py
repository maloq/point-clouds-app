"""
CUDA utilities for device management.
"""

import torch
from typing import Optional


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: If True, prefer CUDA over CPU
    
    Returns:
        torch.device
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def get_device_info() -> dict:
    """
    Get information about available devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': None,
        'device_name': None,
        'memory_allocated': None,
        'memory_cached': None,
    }
    
    if torch.cuda.is_available():
        info['current_device'] = torch.cuda.current_device()
        info['device_name'] = torch.cuda.get_device_name(info['current_device'])
        info['memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
        info['memory_cached'] = torch.cuda.memory_reserved() / 1024**2  # MB
    
    return info


def to_device(tensor: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Move tensor to device.
    
    Args:
        tensor: Input tensor
        device: Target device (if None, uses best available)
    
    Returns:
        Tensor on target device
    """
    if device is None:
        device = get_device()
    return tensor.to(device)


def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def synchronize():
    """Synchronize CUDA operations."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()

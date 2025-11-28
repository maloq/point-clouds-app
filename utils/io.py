"""
I/O utilities for saving and loading checkpoints and data.
"""

import torch
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def ensure_checkpoint_dir():
    """Ensure checkpoint directory exists."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def save_checkpoint(
    state_dict: Dict[str, Any],
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a checkpoint.
    
    Args:
        state_dict: Dictionary containing model state, optimizer state, etc.
        name: Base name for the checkpoint
        metadata: Additional metadata to save
    
    Returns:
        Path to saved checkpoint
    """
    ensure_checkpoint_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pt"
    filepath = CHECKPOINT_DIR / filename
    
    save_data = {
        'state_dict': state_dict,
        'metadata': metadata or {},
        'timestamp': timestamp,
    }
    
    torch.save(save_data, filepath)
    
    return str(filepath)


def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """
    Load a checkpoint.
    
    Args:
        filepath: Path to checkpoint file
    
    Returns:
        Checkpoint dictionary
    """
    return torch.load(filepath, map_location='cpu')


def list_checkpoints(name_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List available checkpoints.
    
    Args:
        name_filter: Optional filter for checkpoint names
    
    Returns:
        List of checkpoint info dictionaries
    """
    ensure_checkpoint_dir()
    
    checkpoints = []
    for f in CHECKPOINT_DIR.glob("*.pt"):
        if name_filter and name_filter not in f.name:
            continue
        
        try:
            data = torch.load(f, map_location='cpu')
            checkpoints.append({
                'path': str(f),
                'name': f.stem,
                'timestamp': data.get('timestamp', 'unknown'),
                'metadata': data.get('metadata', {}),
            })
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)


def save_point_clouds(
    point_clouds: Dict[str, torch.Tensor],
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save point clouds to file.
    
    Args:
        point_clouds: Dictionary of point cloud tensors
        name: Base name for the file
        metadata: Additional metadata
    
    Returns:
        Path to saved file
    """
    ensure_checkpoint_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_pc_{timestamp}.pt"
    filepath = CHECKPOINT_DIR / filename
    
    save_data = {
        'point_clouds': {k: v.cpu() for k, v in point_clouds.items()},
        'metadata': metadata or {},
        'timestamp': timestamp,
    }
    
    torch.save(save_data, filepath)
    
    return str(filepath)


def load_point_clouds(filepath: str) -> Dict[str, torch.Tensor]:
    """
    Load point clouds from file.
    
    Args:
        filepath: Path to point cloud file
    
    Returns:
        Dictionary of point cloud tensors
    """
    data = torch.load(filepath, map_location='cpu')
    return data['point_clouds']


def save_training_history(
    history: Dict[str, List[float]],
    name: str,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save training history.
    
    Args:
        history: Dictionary of metric lists
        name: Base name for the file
        config: Training configuration
    
    Returns:
        Path to saved file
    """
    ensure_checkpoint_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_history_{timestamp}.json"
    filepath = CHECKPOINT_DIR / filename
    
    save_data = {
        'history': history,
        'config': config or {},
        'timestamp': timestamp,
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    return str(filepath)


def load_training_history(filepath: str) -> Dict[str, Any]:
    """
    Load training history.
    
    Args:
        filepath: Path to history file
    
    Returns:
        Training history dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_sequence(
    sequence: List[torch.Tensor],
    name: str,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save a sequence of point clouds (e.g., training progression).
    
    Args:
        sequence: List of point cloud tensors
        name: Base name for the file
        metadata: Additional metadata
    
    Returns:
        Path to saved file
    """
    ensure_checkpoint_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_seq_{timestamp}.pt"
    filepath = CHECKPOINT_DIR / filename
    
    save_data = {
        'sequence': [pc.cpu() for pc in sequence],
        'metadata': metadata or {},
        'timestamp': timestamp,
    }
    
    torch.save(save_data, filepath)
    
    return str(filepath)


def load_sequence(filepath: str) -> List[torch.Tensor]:
    """
    Load a sequence of point clouds.
    
    Args:
        filepath: Path to sequence file
    
    Returns:
        List of point cloud tensors
    """
    data = torch.load(filepath, map_location='cpu')
    return data['sequence']

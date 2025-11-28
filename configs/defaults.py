"""
Default configuration parameters for the application.
"""

import torch

# Point Cloud Defaults
DEFAULT_NUM_POINTS = 1024
DEFAULT_BATCH_SIZE = 32
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Defaults
TRAIN_LEARNING_RATE = 1e-3
TRAIN_EPOCHS = 100
TRAIN_BETA1 = 0.9
TRAIN_BETA2 = 0.999

# Network Defaults
LATENT_DIM = 512
ENCODER_TYPE = "pointnet"  # mlp, pointnet, gnn
HEAD_TYPE = "rotation"     # rotation, flow, combined

# Visualization Defaults
PLOT_POINT_SIZE = 2
PLOT_OPACITY = 0.8
COLOR_SOURCE = "#1f77b4"  # Blue
COLOR_TARGET = "#ff7f0e"  # Orange
COLOR_PRED = "#2ca02c"    # Green

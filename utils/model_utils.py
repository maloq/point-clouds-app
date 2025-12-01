
import torch
import torch.nn as nn

def get_params_vector(model: nn.Module) -> torch.Tensor:
    """
    Flatten all trainable parameters of a model into a single 1D vector.
    """
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param.view(-1))
    return torch.cat(params)

def set_params_vector(model: nn.Module, vector: torch.Tensor):
    """
    Load a 1D vector of parameters back into the model.
    """
    offset = 0
    for param in model.parameters():
        if param.requires_grad:
            numel = param.numel()
            # Copy the slice of the vector into the parameter
            param.data.copy_(vector[offset:offset+numel].view(param.size()))
            offset += numel

import torch

def generate_padding_mask(x: torch.Tensor, pad_value: float = 0.0):
    return torch.tensor(x == pad_value, dtype=torch.bool)

def get_lengths(x: torch.Tensor, pad_value: float = 0.0):
    return torch.sum(x != pad_value, dim=1)
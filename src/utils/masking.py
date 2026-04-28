import torch

def observed_value_mask(values: torch.Tensor) -> torch.Tensor:
    return (~torch.isnan(values)).float()

def fill_missing_with_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(values, nan=0.0) * mask

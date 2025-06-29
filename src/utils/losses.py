import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Union, Dict, Optional

def rolling_std(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Calculate the rolling standard deviation of a sequence.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T), where B is batch size and T is sequence length.
        k (int, optional): Window size for rolling calculation. Defaults to 3.

    Returns:
        torch.Tensor: Rolling standard deviation tensor of shape (B, T).
    """
    pads = (k // 2, k - 1 - k // 2)
    x_pad = torch.nn.functional.pad(x, pads, mode='replicate')
    x_unf = x_pad.unfold(1, k, 1)  # (B, T, k)
    return x_unf.std(dim=2)

def volatility_mismatch_loss(y_pred: torch.Tensor, index_ts: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute the volatility mismatch loss between predicted and true sequences.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (B, T).
        index_ts (torch.Tensor): Ground truth tensor of shape (B, T).
        k (int, optional): Window size for rolling standard deviation. Defaults to 3.

    Returns:
        torch.Tensor: Scalar tensor representing the volatility mismatch loss.
    """
    pred_vol = rolling_std(y_pred, k)
    index_vol = rolling_std(index_ts, k)
    return torch.mean((pred_vol - index_vol) ** 2)

def edge_weighted_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Calculate mean squared error with higher weights on the first and last elements.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (B, T).
        y_true (torch.Tensor): Ground truth tensor of shape (B, T).

    Returns:
        torch.Tensor: Scalar tensor representing the weighted mean squared error.
    """
    weights = torch.ones(y_true.shape[1], device=y_true.device)
    weights[0] = 2.0
    weights[-1] = 2.0
    loss = ((y_pred - y_true) ** 2) * weights
    return loss.mean()

def smoothness_loss(y_pred: torch.Tensor) -> torch.Tensor:
    """Compute a smoothness loss encouraging smooth second differences in the predicted sequence.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (B, T).

    Returns:
        torch.Tensor: Scalar tensor representing the smoothness loss.
    """
    d1 = y_pred[:, 1:] - y_pred[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    return torch.mean(d2 ** 2)

def local_plateau_loss(y_pred: torch.Tensor, y_true: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Calculate loss based on the average values around the local minimum in the sequence.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (B, T).
        y_true (torch.Tensor): Ground truth tensor of shape (B, T).
        k (int, optional): Window size around the local minimum to consider. Defaults to 3.

    Returns:
        torch.Tensor: Scalar tensor representing the local plateau loss.
    """
    idx_min = y_true.argmin(dim=1)
    loss = 0
    for b in range(y_pred.shape[0]):
        i = idx_min[b].item()
        start = max(i - k, 0)
        end = min(i + k + 1, y_pred.shape[1])
        true_avg = y_true[b, start:end].mean()
        pred_avg = y_pred[b, start:end].mean()
        loss += (true_avg - pred_avg) ** 2
    return loss / y_pred.shape[0]

def local_extrema_mask(x: torch.Tensor) -> torch.Tensor:
    """Generate a mask indicating the positions of local extrema (maxima and minima) in the sequence.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T).

    Returns:
        torch.Tensor: Float tensor mask of shape (B, T) with 1s at local extrema positions and 0s elsewhere.
    """
    left = x[:, 1:-1] > x[:, :-2]
    right = x[:, 1:-1] > x[:, 2:]
    local_max = left & right

    left = x[:, 1:-1] < x[:, :-2]
    right = x[:, 1:-1] < x[:, 2:]
    local_min = left & right

    extrema = local_max | local_min  # shape: (B, T-2)
    return torch.nn.functional.pad(extrema.float(), (1,1))  # pad to (B, T)

def composite_loss(y_pred: torch.Tensor, y_true: torch.Tensor, return_all: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
    """Compute a composite loss combining multiple loss components for time series reconstruction.

    Args:
        y_pred (torch.Tensor): Predicted tensor of shape (B, T).
        y_true (torch.Tensor): Ground truth tensor of shape (B, T).
        return_all (bool, optional): If True, return a tuple of total loss and dictionary of individual losses. Defaults to False.

    Returns:
        torch.Tensor or (torch.Tensor, dict): Total loss scalar tensor or tuple of total loss and dict of individual losses.
    """
    seq_len = y_true.shape[1]
    weights = torch.ones(seq_len, device=y_true.device)
    weights[0] = 2.0
    weights[-1] = 2.0
    mse_loss = ((y_pred - y_true) ** 2 * weights).mean()

    min_val_loss = nn.functional.mse_loss(y_pred.min(dim=1).values, y_true.min(dim=1).values)
    max_val_loss = nn.functional.mse_loss(y_pred.max(dim=1).values, y_true.max(dim=1).values)

    pred_min_pos = y_pred.argmin(dim=1).float()
    pred_max_pos = y_pred.argmax(dim=1).float()
    true_min_pos = y_true.argmin(dim=1).float()
    true_max_pos = y_true.argmax(dim=1).float()

    min_pos_loss = nn.functional.mse_loss(pred_min_pos / seq_len, true_min_pos / seq_len)
    max_pos_loss = nn.functional.mse_loss(pred_max_pos / seq_len, true_max_pos / seq_len)

    smooth_loss = smoothness_loss(y_pred)

    plateau_loss = local_plateau_loss(y_pred, y_true)

    extrema_mask = local_extrema_mask(y_true)  # (B, T)
    extrema_loss = ((y_pred - y_true) ** 2 * extrema_mask).mean()
    vol_loss = volatility_mismatch_loss(y_pred, y_true) if y_true is not None else torch.tensor(0.0, device=y_pred.device)

    total_loss = (
        mse_loss +
        0.8 * (min_val_loss + max_val_loss) +
        0.4 * (min_pos_loss + max_pos_loss) +
        0.1 * smooth_loss +
        0.6 * extrema_loss +
        0.5 * plateau_loss +
        1.0 * vol_loss
    )

    if return_all:
        return total_loss, {
            "mse": mse_loss.item(),
            "min_val": min_val_loss.item(),
            "max_val": max_val_loss.item(),
            "min_pos": min_pos_loss.item(),
            "max_pos": max_pos_loss.item(),
            "smooth": smooth_loss.item(),
            "extrema": extrema_loss.item(),
            "plateau": plateau_loss.item(),
            "volatility": vol_loss.item()
        }
    return total_loss

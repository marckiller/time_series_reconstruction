import torch
import torch.nn as nn

def apply_masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Computes the masked mean squared error between tensors a and b.
    """
    return ((a - b) ** 2 * mask).sum() / mask.sum().clamp(min=1)

def composite_loss(y_pred, y_true, mask, x_ts_mask, weights: dict):
    """
    Computes a flexible composite loss with configurable weights for each component.
    Only computes and includes components whose weight is provided and > 0.
    All losses use masking as appropriate.

    Args:
        y_pred (Tensor): Predicted sequences, shape (B, L)
        y_true (Tensor): Ground truth sequences, shape (B, L)
        mask (Tensor): Binary mask indicating valid positions in y_true, shape (B, L)
        x_ts_mask (Tensor or None): Binary mask indicating observed positions in input time series, shape (B, L)
        weights (dict): Dictionary specifying weights for each component. Supported keys:
            - 'mse': masked mean squared error over full sequence
            - 'min_val': MSE between sequence minima
            - 'max_val': MSE between sequence maxima
            - 'min_pos': MSE between positions (normalized) of minimum
            - 'max_pos': MSE between positions (normalized) of maximum
            - 'roughness': penalizes second derivative (smoothness constraint)
            - 'pull': local loss around observed points in x_ts_mask

            'mse': Measures the overall masked mean squared error across the sequence.
            'min_val': Penalizes differences between the minimum values of predicted and true sequences.
            'max_val': Penalizes differences between the maximum values of predicted and true sequences.
            'min_pos': Penalizes differences in the (normalized) position of the minimum value in the sequence.
            'max_pos': Penalizes differences in the (normalized) position of the maximum value in the sequence.
            'roughness': Penalizes large second derivatives (enforces smoothness in predictions).
            'pull': Emphasizes accuracy in a window around observed points in x_ts_mask.

    Returns:
        total (Tensor): Weighted sum of active loss components
        losses (dict): Dictionary of individual loss components
    """
    losses = {}
    total = 0.0

    if weights.get('mse', 0.0) > 0:
        losses['mse'] = apply_masked_mse(y_pred, y_true, mask)
        total += weights['mse'] * losses['mse']

    if weights.get('min_val', 0.0) > 0:
        # Use mask for the first timestep as a proxy for sequence mask
        losses['min_val'] = apply_masked_mse(
            y_pred.min(dim=1).values, y_true.min(dim=1).values, mask[:, 0]
        )
        total += weights['min_val'] * losses['min_val']

    if weights.get('max_val', 0.0) > 0:
        losses['max_val'] = apply_masked_mse(
            y_pred.max(dim=1).values, y_true.max(dim=1).values, mask[:, 0]
        )
        total += weights['max_val'] * losses['max_val']

    if weights.get('min_pos', 0.0) > 0 or weights.get('max_pos', 0.0) > 0:
        seq_len = y_true.shape[1]
        if weights.get('min_pos', 0.0) > 0:
            pred_min_pos = y_pred.argmin(dim=1).float() / seq_len
            true_min_pos = y_true.argmin(dim=1).float() / seq_len
            losses['min_pos'] = nn.functional.mse_loss(pred_min_pos, true_min_pos)
            total += weights['min_pos'] * losses['min_pos']
        if weights.get('max_pos', 0.0) > 0:
            pred_max_pos = y_pred.argmax(dim=1).float() / seq_len
            true_max_pos = y_true.argmax(dim=1).float() / seq_len
            losses['max_pos'] = nn.functional.mse_loss(pred_max_pos, true_max_pos)
            total += weights['max_pos'] * losses['max_pos']

    if weights.get('roughness', 0.0) > 0:
        d1 = y_pred[:, 1:] - y_pred[:, :-1]
        losses['roughness'] = torch.mean(d1[:, 1:] ** 2)
        total += weights['roughness'] * losses['roughness']

    if weights.get('pull', 0.0) > 0 and x_ts_mask is not None:
        window = 2
        B, L = x_ts_mask.shape
        pull_mask = torch.zeros_like(mask)
        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            shifted = torch.roll(x_ts_mask, shifts=offset, dims=1)
            pull_mask += shifted
        pull_mask = (pull_mask > 0).float() * (mask == 1).float()
        if pull_mask.sum() > 0:
            losses['pull'] = apply_masked_mse(y_pred, y_true, pull_mask)
            total += weights['pull'] * losses['pull']
        else:
            losses['pull'] = torch.tensor(0.0, device=y_pred.device)

    return total, losses

import numpy as np
import torch


STATIC_FEATURES = [
    "corr_30",
    "corr_60",
    "open_pos",
    "close_pos",
    "body_to_range",
    "direction",
    "index_open_pos",
    "index_close_pos",
    "index_body_to_range",
    "index_direction",
]

OPEN_POS_IDX = STATIC_FEATURES.index("open_pos")
CLOSE_POS_IDX = STATIC_FEATURES.index("close_pos")
CORR_30_IDX = STATIC_FEATURES.index("corr_30")
CORR_60_IDX = STATIC_FEATURES.index("corr_60")


def linear_baseline(x_ts: torch.Tensor, x_mask: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
    batch, length = x_ts.shape
    out = torch.empty_like(x_ts)
    xs_np = x_ts.detach().cpu().numpy()
    mask_np = x_mask.detach().cpu().numpy().astype(bool)
    static_np = x_static.detach().cpu().numpy()
    grid = np.arange(length)

    for i in range(batch):
        anchors_x = [0, length - 1]
        anchors_y = [static_np[i, OPEN_POS_IDX], static_np[i, CLOSE_POS_IDX]]
        observed_idx = np.where(mask_np[i])[0]
        for idx in observed_idx:
            if idx == 0 or idx == length - 1:
                continue
            anchors_x.append(int(idx))
            anchors_y.append(float(xs_np[i, idx]))

        order = np.argsort(anchors_x)
        anchors_x_arr = np.asarray(anchors_x, dtype=float)[order]
        anchors_y_arr = np.asarray(anchors_y, dtype=float)[order]
        filled = np.interp(grid, anchors_x_arr, anchors_y_arr)
        filled[mask_np[i]] = xs_np[i, mask_np[i]]
        out[i] = torch.as_tensor(filled, dtype=x_ts.dtype, device=x_ts.device)
    return out


def index_residual_baseline(
    x_ts: torch.Tensor,
    x_mask: torch.Tensor,
    x_index: torch.Tensor,
    x_static: torch.Tensor,
) -> torch.Tensor:
    batch, length = x_ts.shape
    out = torch.empty_like(x_ts)
    xs_np = x_ts.detach().cpu().numpy()
    mask_np = x_mask.detach().cpu().numpy().astype(bool)
    idx_np = x_index.detach().cpu().numpy()
    static_np = x_static.detach().cpu().numpy()

    for i in range(batch):
        anchors = {0: static_np[i, OPEN_POS_IDX], length - 1: static_np[i, CLOSE_POS_IDX]}
        for pos in np.where(mask_np[i])[0]:
            anchors[int(pos)] = float(xs_np[i, pos])

        index_values = idx_np[i].astype(float)
        finite = np.isfinite(index_values)
        positions = np.arange(length)
        if finite.sum() == 0:
            index_values = np.linspace(0.0, 1.0, length)
        elif finite.sum() < length:
            index_values = np.interp(positions, positions[finite], index_values[finite])

        corr_values = static_np[i, [CORR_30_IDX, CORR_60_IDX]]
        corr_values = corr_values[np.isfinite(corr_values)]
        beta = float(np.clip(np.mean(corr_values), -1.0, 1.0)) if len(corr_values) else 0.0

        filled = np.empty(length, dtype=float)
        anchor_positions = sorted(anchors)
        for left, right in zip(anchor_positions[:-1], anchor_positions[1:]):
            segment_positions = np.arange(left, right + 1)
            target_linear = np.linspace(anchors[left], anchors[right], len(segment_positions))
            index_linear = np.linspace(index_values[left], index_values[right], len(segment_positions))
            index_residual = index_values[segment_positions] - index_linear
            filled[segment_positions] = target_linear + beta * index_residual

        first = anchor_positions[0]
        last = anchor_positions[-1]
        filled[:first] = anchors[first]
        filled[last:] = anchors[last]
        filled[mask_np[i]] = xs_np[i, mask_np[i]]
        out[i] = torch.as_tensor(filled, dtype=x_ts.dtype, device=x_ts.device)
    return out

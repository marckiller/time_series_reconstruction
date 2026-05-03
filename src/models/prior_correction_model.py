import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.05):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


def masked_first_difference(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    diff = torch.zeros_like(values)
    diff_mask = torch.zeros_like(mask)
    diff[:, 1:] = values[:, 1:] - values[:, :-1]
    diff_mask[:, 1:] = mask[:, 1:] * mask[:, :-1]
    return diff * diff_mask, diff_mask


class PriorCorrectionModel(nn.Module):
    """Conv1D model that learns a bounded correction to an index-residual prior.

    The baseline prior is computed outside the model from hourly OHLC, sparse target
    anchors, the index path, and target-index correlations. The network only learns
    a residual correction, which keeps the deployed model close to the transparent
    baseline while allowing data-driven local adjustments.
    """

    def __init__(
        self,
        seq_len: int = 60,
        static_dim: int = 10,
        channels: int = 64,
        n_blocks: int = 4,
        dropout: float = 0.05,
        correction_scale: float = 0.25,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.correction_scale = correction_scale
        self.input_proj = nn.Sequential(nn.Conv1d(10, channels, kernel_size=1), nn.GELU())
        dilations = [1, 2, 4, 8]
        self.blocks = nn.ModuleList(
            [
                ResidualConvBlock(
                    channels=channels,
                    kernel_size=5,
                    dilation=dilations[i % len(dilations)],
                    dropout=dropout,
                )
                for i in range(n_blocks)
            ]
        )
        self.static_net = nn.Sequential(
            nn.Linear(2 * static_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.output = nn.Conv1d(channels, 1, kernel_size=1)

    def forward(
        self,
        X_index: torch.Tensor,
        X_index_mask: torch.Tensor,
        X_ts: torch.Tensor,
        X_ts_mask: torch.Tensor,
        X_prior: torch.Tensor,
        X_prior_mask: torch.Tensor,
        X_static: torch.Tensor,
        X_static_mask: torch.Tensor,
    ) -> torch.Tensor:
        index_level = X_index * X_index_mask
        target_level = X_ts * X_ts_mask
        prior = X_prior * X_prior_mask

        prior_diff, prior_diff_mask = masked_first_difference(prior, X_prior_mask)
        index_diff, index_diff_mask = masked_first_difference(index_level, X_index_mask)
        target_diff, target_diff_mask = masked_first_difference(target_level, X_ts_mask)

        x = torch.stack(
            [
                prior,
                prior_diff,
                index_level,
                index_diff,
                X_index_mask,
                index_diff_mask,
                target_level,
                target_diff,
                X_ts_mask,
                target_diff_mask,
            ],
            dim=1,
        )
        h = self.input_proj(x)
        static_input = torch.cat([X_static * X_static_mask, X_static_mask], dim=1)
        h = h + self.static_net(static_input).unsqueeze(-1)
        for block in self.blocks:
            h = block(h)

        correction = self.correction_scale * torch.tanh(self.output(h).squeeze(1))
        pred = prior + correction
        return pred * (1 - X_ts_mask) + X_ts * X_ts_mask

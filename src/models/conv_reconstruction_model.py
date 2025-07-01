import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReconstructionModel(nn.Module):
    """
    A convolutional neural network model for masked time series reconstruction.

    Args:
        seq_len (int): Length of the time series sequences.
        static_dim (int): Dimension of the static input features.
        conv_channels (int): Number of output channels for each convolutional filter.

    Forward Inputs:
        X_index (Tensor): Index time series input of shape (B, L).
        X_index_mask (Tensor): Binary mask for X_index of shape (B, L).
        X_ts (Tensor): Ground truth time series input of shape (B, L).
        X_ts_mask (Tensor): Binary mask for X_ts indicating observed values (B, L).
        X_static (Tensor): Static input features of shape (B, static_dim).
        X_static_mask (Tensor): Binary mask for static features (B, static_dim).

    Returns:
        Tensor of shape (B, L): Reconstructed time series.
    """
    def __init__(self, seq_len=60, static_dim=10, conv_channels=32):
        super().__init__()
        self.seq_len = seq_len
        self.static_dim = static_dim

        # --- CNN for X_index ---
        self.convs = nn.ModuleList([
            nn.Conv1d(1, conv_channels, kernel_size=k, padding='same')
            for k in [2, 5, 10, 20]
        ])
        self.index_proj = nn.Linear(128 * seq_len, seq_len)

        # --- Dense for X_static ---
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len)
        )

    def forward(self, X_index, X_index_mask, X_ts, X_ts_mask, X_static, X_static_mask):
        B, L = X_index.shape

        # --- CNN branch ---
        x_idx = X_index * X_index_mask  # masking
        x_idx = x_idx.unsqueeze(1)  # (B, 1, L)
        conv_outs = [F.relu(conv(x_idx)) for conv in self.convs]  # (B, C, L)
        x_cnn = torch.cat(conv_outs, dim=1)  # (B, C_total, L)
        x_cnn = x_cnn.view(B, -1)  # (B, C_total * L)
        x_cnn = self.index_proj(x_cnn)  # (B, L)

        # --- Static branch ---
        x_static_masked = X_static * X_static_mask  # masking
        x_static_out = self.static_net(x_static_masked)  # (B, L)

        # --- Combine model components ---
        pred = x_cnn + x_static_out

        # --- Incorporate X_ts as ground truth where available ---
        pred = pred * (1 - X_ts_mask) + X_ts * X_ts_mask

        return pred
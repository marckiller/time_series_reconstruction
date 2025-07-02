import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedReconstructionModel(nn.Module):
    def __init__(self, seq_len=60, static_dim=10, conv_channels=32):
        super().__init__()
        self.seq_len = seq_len
        self.static_dim = static_dim

        # --- CNN dla X_index ---
        self.convs = nn.ModuleList([
            nn.Conv1d(1, conv_channels, kernel_size=k, padding='same')
            for k in [2, 5, 10, 20]
        ])
        self.index_proj = nn.Linear(128 * seq_len, seq_len)  # 4 convs × 32 channels = 128

        # --- Dense dla X_static ---
        self.static_net = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len)
        )

        # --- Dense dla X_ts + X_ts_mask ---
        self.ts_net = nn.Sequential(
            nn.Linear(2 * seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len)
        )

    def forward(self, X_index, X_index_mask, X_ts, X_ts_mask, X_static, X_static_mask):
        B, L = X_index.shape

        # --- CNN branch ---
        x_idx = X_index * X_index_mask  # (B, L)
        x_idx = x_idx.unsqueeze(1)  # (B, 1, L)
        conv_outs = [F.relu(conv(x_idx)) for conv in self.convs]  # [(B, C, L), ...]
        x_cnn = torch.cat(conv_outs, dim=1)  # (B, 128, L)
        x_cnn = x_cnn.view(B, -1)  # (B, 128 * L)
        x_cnn = self.index_proj(x_cnn)  # (B, L)

        # --- Static branch ---
        x_static_masked = X_static * X_static_mask  # (B, D)
        x_static_out = self.static_net(x_static_masked)  # (B, L)

        # --- Time series branch ---
        x_ts_masked = X_ts * X_ts_mask  # (B, L)
        x_ts_input = torch.cat([x_ts_masked, X_ts_mask], dim=1)  # (B, 2L)
        x_ts_out = self.ts_net(x_ts_input)  # (B, L)

        # --- Summation of modeling components ---
        pred = x_cnn + x_static_out + x_ts_out

        # --- Inserting known values ---
        pred = pred * (1 - X_ts_mask) + X_ts * X_ts_mask

        return pred
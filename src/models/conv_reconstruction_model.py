import torch
import torch.nn as nn

class ConvPredictionModel(nn.Module):
    """
    A convolutional neural network model for time series prediction that combines multiple convolutional branches
    with a static feature processing branch. The model processes sequential time series data through convolutional
    layers with different kernel sizes, and static features through fully connected layers, then concatenates
    the outputs to produce a final prediction.

    Architecture:
    - Three convolutional branches with kernel sizes 2, 5, and 9, each consisting of two Conv1d + ReLU layers followed by flattening.
    - A static feature branch with two fully connected layers with ReLU activations.
    - A final fully connected output layer that combines features from all branches to predict the output sequence.
    """

    def __init__(self, seq_len: int = 60, static_dim: int = 10, output_dim: int = 60):
        super().__init__()
        self.seq_len = seq_len
        self.static_dim = static_dim
        self.output_dim = output_dim

        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 8, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Conv1d(8, 8, kernel_size=k, padding=k // 2),
                nn.ReLU(),
                nn.Flatten()
            )
            for k in [2, 5, 9]
        ])

        self.static_branch = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Dummy forward to calculate output shape
        with torch.no_grad():
            dummy_ts = torch.zeros(1, 1, seq_len)
            conv_outs = [branch(dummy_ts) for branch in self.conv_branches]
            conv_dim = sum([out.shape[1] for out in conv_outs])
            dummy_static = torch.zeros(1, static_dim)
            static_dim_out = self.static_branch(dummy_static).shape[1]
            total_input_dim = conv_dim + static_dim_out

        self.output_layer = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x_seq: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x_seq (torch.Tensor): Time series input tensor of shape (batch_size, seq_len).
            x_static (torch.Tensor): Static features tensor of shape (batch_size, static_dim).

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, output_dim).
        """
        x_seq = x_seq.unsqueeze(1)  # (B, 1, seq_len)
        conv_outs = [branch(x_seq) for branch in self.conv_branches]
        x_conv = torch.cat(conv_outs, dim=1)
        x_static_out = self.static_branch(x_static)
        x_all = torch.cat([x_conv, x_static_out], dim=1)
        return self.output_layer(x_all)
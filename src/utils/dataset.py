import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union

def build_dataset(
    df: pd.DataFrame,
    inputs: List[str],
    outputs: List[str],
    columns_map: Dict[str, List[str]],
    expand_sequence_columns: bool = True
) -> Tuple[torch.Tensor, ...]:
    """
    Builds model-ready tensors from a DataFrame.

    Parameters:
        df: input DataFrame
        inputs: list of input names (e.g. ['X_seq', 'X_static'])
        outputs: list of output names (e.g. ['y_seq', 'y_mask'])
        columns_map: mapping of each name to its column list, e.g.
                     {'X_seq': ['ts_min_max_norm'], 'X_static': ['open_pos', 'close_pos'], ...}
        expand_sequence_columns: if True, sequence-like columns (e.g. time series) will be expanded

    Returns:
        Tuple of torch.Tensors in order: inputs..., outputs...
    """

    def is_sequence_like(x):
        return isinstance(x, (list, tuple, np.ndarray))

    def process_block(cols: List[str]) -> torch.Tensor:
        arrays = []
        for col in cols:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame.")

            series = df[col]

            if series.isnull().any():
                raise ValueError(f"Column '{col}' contains NaN values.")

            if expand_sequence_columns:
                if not is_sequence_like(series.iloc[0]):
                    raise TypeError(f"Expected sequence-like data in column '{col}', got {type(series.iloc[0])}.")

                lengths = [len(x) for x in series if is_sequence_like(x)]
                if len(set(lengths)) > 1:
                    raise ValueError(f"Inconsistent sequence lengths in column '{col}': {set(lengths)}")

                arr = np.array(series.tolist())  # shape: (N, L)
            else:
                arr = np.array(series).reshape(-1, 1)  # shape: (N, 1)

            arrays.append(arr)

        concatenated = np.concatenate(arrays, axis=1)
        return torch.tensor(concatenated, dtype=torch.float32)

    all_keys = inputs + outputs
    tensors = [process_block(columns_map[k]) for k in all_keys]

    return tuple(tensors)

from torch.utils.data import Dataset

class MaskedTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for training models on masked time series data.

    This dataset is intended to simulate missing or irregular data patterns 
    by applying different masking strategies to three input modalities:
    - X_index: time or position indices
    - X_ts: temporal (time-varying) features
    - X_static: static features

    It also processes target sequences y and returns a mask for valid targets.

    Parameters:
        X_index (torch.Tensor): Tensor of shape (N, L1) with index-like values.
        X_ts (torch.Tensor): Tensor of shape (N, L2) with time series features.
        X_static (torch.Tensor): Tensor of shape (N, D) with static features.
        y (torch.Tensor): Tensor of shape (N, L3) with target values.
        mask_config (dict, optional): Dictionary to control masking behavior:
            - 'ts_keep_prob' (float): Probability of keeping a value in X_ts during random masking.
            - 'index_keep_prob' (float): Probability of keeping a value in X_index.
            - 'static_p' (float): Probability of masking a static feature.

    Each call to __getitem__ returns a dictionary:
        {
            'X_index': (L1,), masked indices,
            'X_index_mask': (L1,), 1 if value was kept, 0 otherwise,
            'X_ts': (L2,), masked time series input,
            'X_ts_mask': (L2,), corresponding binary mask,
            'X_static': (D,), masked static features,
            'X_static_mask': (D,), corresponding binary mask,
            'y': (L3,), target sequence with NaNs replaced by zero,
            'y_mask': (L3,), 1 where original y was not NaN
        }
    """
    def __init__(
        self,
        X_index: torch.Tensor,
        X_ts: torch.Tensor,
        X_static: torch.Tensor,
        y: torch.Tensor,
        mask_config: Dict[str, Union[float, int]] = None
    ):
        self.X_index = X_index
        self.X_ts = X_ts
        self.X_static = X_static
        self.y = y
        self.mask_config = mask_config or {}

    def __len__(self):
        return self.X_index.shape[0]

    def interval_mask(self, length: int, every: int, start: int = 0) -> torch.Tensor:
        mask = torch.zeros(length)
        mask[start::every] = 1.0
        return mask

    def random_mask(self, x: torch.Tensor, base_mask: torch.Tensor, p: float) -> Tuple[torch.Tensor, torch.Tensor]:
        random_keep = (torch.rand_like(x) > p).float()
        mask = base_mask * random_keep
        x_masked = torch.nan_to_num(x, nan=0.0) * mask
        return x_masked, mask
    
    def set_mask_probabilities(self, mask_config: Dict[str, Union[float, int]]):
        """
        Set the masking probabilities for the dataset.

        Parameters:
            mask_config (dict): Dictionary with keys 'ts_keep_prob', 'index_keep_prob', and 'static_p'.
        """
        self.mask_config = mask_config

    def __getitem__(self, idx):
        xi = self.X_index[idx]
        xts = self.X_ts[idx]
        xs = self.X_static[idx]
        yt = self.y[idx]

        L = xi.shape[0]

        xi_base_mask = (~torch.isnan(xi)).float()
        xts_base_mask = (~torch.isnan(xts)).float()
        xs_base_mask = (~torch.isnan(xs)).float()
        y_mask       = (~torch.isnan(yt)).float()

        keep_prob = self.mask_config.get('ts_keep_prob', 0.2)
        ts_mask = (torch.rand_like(xts) < keep_prob).float()
            
        ts_mask = xts_base_mask * ts_mask
        xts_masked = torch.nan_to_num(xts, nan=0.0) * ts_mask

        keep_prob = self.mask_config.get('index_keep_prob', 1.0)
        random_keep = (torch.rand_like(xi) < keep_prob).float()
        idx_mask = xi_base_mask * random_keep
        xi_masked = torch.nan_to_num(xi, nan=0.0) * idx_mask

        static_keep_prob = self.mask_config.get('static_keep_prob', 1.0)
        random_keep = (torch.rand_like(xs) < static_keep_prob).float()
        xs_mask = xs_base_mask * random_keep
        xs_masked = torch.nan_to_num(xs, nan=0.0) * xs_mask

        yt_filled = torch.nan_to_num(yt, nan=0.0)

        return {
            'X_index': xi_masked,
            'X_index_mask': idx_mask,
            'X_ts': xts_masked,
            'X_ts_mask': ts_mask,
            'X_static': xs_masked,
            'X_static_mask': xs_mask,
            'y': yt_filled,
            'y_mask': y_mask,
            'X_index_raw': xi
        }
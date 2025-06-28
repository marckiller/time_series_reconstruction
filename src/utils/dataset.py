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
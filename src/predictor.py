import torch
import numpy as np

class JSONPredictor:
    def __init__(self, model: torch.nn.Module, device=None):
        self.model = model
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def json_to_tensor(self, data):
        def to_tensor(arr):
            arr_np = np.array(arr, dtype=np.float32)
            tensor = torch.from_numpy(arr_np)
            return tensor

        X_index = to_tensor(data['X_index'])
        X_ts = to_tensor(data['X_ts'])
        X_static = to_tensor(data['X_static'])

        # Masks: 1 where not nan, 0 where nan
        X_index_mask = (~torch.isnan(X_index)).float()
        X_ts_mask = (~torch.isnan(X_ts)).float()
        X_static_mask = (~torch.isnan(X_static)).float()

        # Replace NaNs with zero
        X_index = torch.nan_to_num(X_index, nan=0.0)
        X_ts = torch.nan_to_num(X_ts, nan=0.0)
        X_static = torch.nan_to_num(X_static, nan=0.0)

        return {
            'X_index': X_index,
            'X_index_mask': X_index_mask,
            'X_ts': X_ts,
            'X_ts_mask': X_ts_mask,
            'X_static': X_static,
            'X_static_mask': X_static_mask
        }

    def predict(self, data_json):
        example = self.json_to_tensor(data_json)
        with torch.no_grad():
            xb_idx = example['X_index'].unsqueeze(0).to(self.device)
            xb_idx_m = example['X_index_mask'].unsqueeze(0).to(self.device)
            xb_ts = example['X_ts'].unsqueeze(0).to(self.device)
            xb_ts_m = example['X_ts_mask'].unsqueeze(0).to(self.device)
            xb_static = example['X_static'].unsqueeze(0).to(self.device)
            xb_static_m = example['X_static_mask'].unsqueeze(0).to(self.device)

            y_pred = self.model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
            return y_pred.squeeze(0).cpu().numpy()
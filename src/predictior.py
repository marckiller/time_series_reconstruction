import torch

class Predictor:
    def __init__(self, model: torch.nn.Module, device=None):
        self.model = model
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, example: dict) -> torch.Tensor:
        with torch.no_grad():
            xb_idx     = example['X_index'].unsqueeze(0).to(self.device)
            xb_idx_m   = example['X_index_mask'].unsqueeze(0).to(self.device)
            xb_ts      = example['X_ts'].unsqueeze(0).to(self.device)
            xb_ts_m    = example['X_ts_mask'].unsqueeze(0).to(self.device)
            xb_static  = example['X_static'].unsqueeze(0).to(self.device)
            xb_static_m= example['X_static_mask'].unsqueeze(0).to(self.device)

            y_pred = self.model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
            return y_pred.squeeze(0).cpu()
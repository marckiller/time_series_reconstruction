import torch
import matplotlib.pyplot as plt
import random
import numpy as np
import os

def plot_random_samples(model, dataset, device, n=6, rows=2, cols=3, output_path=None):
    model.eval()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))

    indices = random.sample(range(len(dataset)), n)
    for ax, idx in zip(axes.flat, indices):
        sample = dataset[idx]

        with torch.no_grad():
            xb_idx     = sample['X_index'].unsqueeze(0).to(device)
            xb_idx_m   = sample['X_index_mask'].unsqueeze(0).to(device)
            xb_ts      = sample['X_ts'].unsqueeze(0).to(device)
            xb_ts_m    = sample['X_ts_mask'].unsqueeze(0).to(device)
            xb_static  = sample['X_static'].unsqueeze(0).to(device)
            xb_static_m= sample['X_static_mask'].unsqueeze(0).to(device)

            y_true     = sample['X_ts_raw'].cpu().numpy()
            y_pred     = model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
            y_pred     = y_pred.squeeze().cpu().numpy()

        y_true_squeezed = y_true.squeeze()
        ts_mask = sample['X_ts_mask'].squeeze().numpy()
        ts_visible = ts_mask[:, 0] == 1 if ts_mask.ndim == 2 else ts_mask == 1

        ax.plot(y_true_squeezed, 'bo', alpha=0.2, label='y (masked)')
        ax.plot(np.where(ts_visible)[0], y_true_squeezed[ts_visible], 'bo', label='y (visible by model)')

        ax.plot(y_pred, color='orange', linestyle='--', label='Pred')

        index_data = sample['X_index_raw'].squeeze().numpy()
        index_mask = sample['X_index_mask'].squeeze().numpy()
        ax.scatter(np.arange(len(index_data)), index_data, color='gray', alpha=0.2, s=10, label='Index (masked)')
        ax.scatter(np.where(index_mask == 1)[0], index_data[index_mask == 1], color='gray', alpha=1.0, s=10, label='Index (visible)')

        ax.set_title(f"Sample {idx}")
        if ax == axes.flat[0]:
            ax.legend()

    plt.tight_layout()
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        plt.savefig(os.path.join(output_path, "sample_plots.png"))
    plt.show()

if __name__ == "__main__":
    import yaml
    import pandas as pd
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

    from src.utils.dataset import build_dataset, MaskedTimeSeriesDataset
    import importlib

    with open("config/config_experiment.yaml", "r") as f:
        config = yaml.safe_load(f)["plot_samples"]

    df = pd.read_parquet(config["input_path"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if config["date_range"].get("use_date_range", True):
        start = pd.to_datetime(config["date_range"]["start"])
        end = pd.to_datetime(config["date_range"]["end"])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    #df.dropna(inplace=True) #allow for static features to be NaN
    print("Total rows before sampling:", len(df))
    df = df.sample(n=config["n_samples"], random_state=42)
    print(len(df), "samples loaded from dataset")

    model_cfg = config["model"]
    model_module = importlib.import_module(model_cfg["module"])
    model_class = getattr(model_module, model_cfg["class_name"])
    model_kwargs = model_cfg.get("kwargs", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()

    mask_cfg = dict(config.get("mask_config", {}))

    X_index, X_ts, y = build_dataset(df, ['X_index', 'X_ts'], ['y'], {
        'X_index': ['index_ts_low_high_norm'],
        'X_ts': ['ts_low_high_norm'],
        'y': ['ts_low_high_norm']
    })

    (X_static,) = build_dataset(df, inputs=['X_static'], outputs=[], columns_map={
        'X_static': [
            'corr_30', 'corr_60',
            'open_pos', 'close_pos', 'body_to_range', 'direction',
            'index_open_pos', 'index_close_pos', 'index_body_to_range', 'index_direction'
        ]
    }, expand_sequence_columns=False)

    dataset = MaskedTimeSeriesDataset(X_index, X_ts, X_static, y, mask_config=mask_cfg)
    for i in range(len(dataset)):
        dataset[i]['X_index_raw'] = X_index[i]

    plot_config = config.get("plot_config", {})
    plot_random_samples(
        model, dataset, device,
        n=config["n_samples"],
        rows=plot_config.get("rows", 2),
        cols=plot_config.get("cols", 3),
        output_path=config.get("output_path")
    )

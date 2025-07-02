import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import importlib
import random

import torch
import pandas as pd
import numpy as np
import yaml

with open("config/config_experiment.yaml", "r") as f:
    config = yaml.safe_load(f)["compute_metrics"]

df = pd.read_parquet(config["input_path"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

if config["date_range"].get("use_date_range", True):
    start, end = pd.to_datetime(config["date_range"]["start"]), pd.to_datetime(config["date_range"]["end"])
    df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

df.dropna(inplace=True)

random.seed(config["random_seed"])
df = df.sample(n=config["n_samples"], random_state=config["random_seed"])

model_cfg = config["model"]
model_module = importlib.import_module(model_cfg["module"])
model_class = getattr(model_module, model_cfg["class_name"])
model_kwargs = model_cfg.get("kwargs", {})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_class(**model_kwargs).to(device)
model.load_state_dict(torch.load(config["model_path"], map_location=device))
model.eval()

mask_cfg = dict(config.get("mask_config", {}))

print(mask_cfg)

from src.utils.dataset import build_dataset, MaskedTimeSeriesDataset


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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

def compute_metrics(y_true, y_pred, metrics):
    results = {}
    if "mse" in metrics:
        results["mse"] = mean_squared_error(y_true, y_pred)
    if "mae" in metrics:
        results["mae"] = mean_absolute_error(y_true, y_pred)
    if "r2" in metrics:
        results["r2"] = r2_score(y_true, y_pred)
    return results

results = {}

for ts_keep_prob in config["ts_keep_probs"]:
    for index_keep_prob in config["index_keep_probs"]:
        print(f"Evaluating for ts_keep_prob {ts_keep_prob}, index_keep_prob {index_keep_prob}...")

        dataset.set_mask_probabilities({
            'ts_keep_prob': ts_keep_prob,
            'index_keep_prob': index_keep_prob,
            'static_p': mask_cfg.get('static_p', 1.0)
        })
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        total_ts, total_idx = 0, 0
        visible_ts, visible_idx = 0, 0

        for batch in loader:
            ts_mask = batch['X_ts_mask'].numpy()
            idx_mask = batch['X_index_mask'].numpy()
            
            total_ts += ts_mask.size
            total_idx += idx_mask.size
            visible_ts += ts_mask.sum()
            visible_idx += idx_mask.sum()

        avg_ts_shown = visible_ts / total_ts
        avg_idx_shown = visible_idx / total_idx

        print(f"Avg shown - ts: {avg_ts_shown:.2%}, index: {avg_idx_shown:.2%}")

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(loader):
                xb_idx     = batch['X_index'].to(device)
                xb_idx_m   = batch['X_index_mask'].to(device)
                xb_ts      = batch['X_ts'].to(device)
                xb_ts_m    = batch['X_ts_mask'].to(device)
                xb_static  = batch['X_static'].to(device)
                xb_static_m= batch['X_static_mask'].to(device)
                y_true     = batch['y'].cpu().numpy()

                y_pred = model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
                y_pred = y_pred.cpu().numpy()

                all_preds.append(y_pred)
                all_targets.append(y_true)

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        results[(ts_keep_prob, index_keep_prob)] = compute_metrics(all_targets, all_preds, config["metrics"])

os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)
os.makedirs(config["output_plot_path"], exist_ok=True)
with open(config["output_path"], "w") as f:
    yaml.dump(results, f)

import json
json_path = config["output_json_path"]
cleaned = {
    f"{ts}_{idx}": {k: float(v) for k, v in met.items()}
    for (ts, idx), met in results.items()
}
with open(json_path, "w") as f:
    json.dump(cleaned, f, indent=2)

import matplotlib.pyplot as plt
import seaborn as sns

ts_values = config["ts_keep_probs"]
index_values = config["index_keep_probs"]
metrics_list = config["metrics"]
static_p_val = mask_cfg.get("static_p", 1.0)

for metric in metrics_list:
    heatmap_data = np.zeros((len(index_values), len(ts_values)))
    for i, idx in enumerate(index_values):
        for j, ts in enumerate(ts_values):
            heatmap_data[i, j] = results[(ts, idx)][metric]

    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, xticklabels=ts_values, yticklabels=index_values, annot=True, fmt=".4g")
    plt.xlabel("Instrument sequence: % visible")
    plt.ylabel("Index sequence: % visible")
    plt.title(f"{metric.upper()} (static_p = {static_p_val}) - {model_cfg['class_name']}")
    plt.tight_layout()

    out_path = os.path.join(config.get("output_plot_path", "results/plots"), f"{metric}.png")
    plt.savefig(out_path)
    plt.close()

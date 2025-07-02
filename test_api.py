import requests
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def test_api_predict_incremental_masking(n_plots=60):
    with open("config/config_experiment.yaml", "r") as f:
        config = yaml.safe_load(f)["plot_samples"]

    df = pd.read_parquet(config["input_path"])
    df = df.dropna(subset=[
        'index_ts_low_high_norm', 'ts_low_high_norm', 'corr_30', 'corr_60',
        'open_pos', 'close_pos', 'body_to_range', 'direction',
        'index_open_pos', 'index_close_pos', 'index_body_to_range', 'index_direction'
    ])

    row = df.iloc[0].to_dict()

    original_ts = np.array(row["ts_low_high_norm"])
    ts_length = len(original_ts)
    index_ts = np.array(row["index_ts_low_high_norm"])

    base_payload = {
        "index_ts_low_high_norm": row["index_ts_low_high_norm"].tolist(),
        "corr_30": row["corr_30"],
        "corr_60": row["corr_60"],
        "open_pos": row["open_pos"],
        "close_pos": row["close_pos"],
        "body_to_range": row["body_to_range"],
        "direction": row["direction"],
        "index_open_pos": row["index_open_pos"],
        "index_close_pos": row["index_close_pos"],
        "index_body_to_range": row["index_body_to_range"],
        "index_direction": row["index_direction"]
    }

    url = "http://127.0.0.1:8000/predict"

    np.random.seed(42)
    mask_order = np.random.permutation(ts_length)

    rows = 6
    cols = 10
    fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
    axes = axes.flatten()

    for i in range(n_plots):
        ts_masked = original_ts.copy()

        indices_to_mask = mask_order[:i]
        ts_masked[indices_to_mask] = np.nan

        payload = base_payload.copy()
        ts_masked_list = ts_masked.tolist()
        #nan -> None
        ts_masked_list = [None if (isinstance(x, float) and np.isnan(x)) else x for x in ts_masked_list]
        payload["ts_low_high_norm"] = ts_masked_list

        response = requests.post(url, json=payload)
        assert response.status_code == 200
        pred = np.array(response.json()["prediction"])

        ax = axes[i]
        ax.plot(original_ts, 'b-', alpha=0.3, label="Original")
        ax.plot(ts_masked, 'k.', label="Input masked")
        ax.plot(pred, 'r--', label="Prediction")
        ax.plot(index_ts, 'g-', label="Index (price trajectory)")
        ax.set_title(f"Masked {i} points")
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(loc='upper right')

    for j in range(n_plots, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_api_predict_incremental_masking()
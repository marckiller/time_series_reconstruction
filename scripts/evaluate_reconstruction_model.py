import argparse
import importlib
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines import STATIC_FEATURES, index_residual_baseline, linear_baseline
from src.utils.dataset import MaskedTimeSeriesDataset, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction model and baselines.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="eval")
    parser.add_argument("--model-module", default="src.models.prior_correction_model")
    parser.add_argument("--model-class", default="PriorCorrectionModel")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--ts-keep-prob", type=float, default=1.0)
    parser.add_argument("--index-keep-prob", type=float, default=1.0)
    parser.add_argument("--static-keep-prob", type=float, default=1.0)
    parser.add_argument("--plot-samples", type=int, default=25)
    parser.add_argument("--no-clip", action="store_true")
    return parser.parse_args()


def select_device(requested: str = "auto"):
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_frame(path: str, limit: int | None, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if limit is not None and len(df) > limit:
        df = df.sample(n=limit, random_state=seed)
    return df.reset_index(drop=True)


def build_tensors(df: pd.DataFrame):
    sequence_inputs = ["X_index", "X_ts"]
    if "X_prior" in df.columns:
        sequence_inputs.append("X_prior")
    tensors = build_dataset(
        df,
        sequence_inputs,
        ["y"],
        {
            "X_index": ["X_index"],
            "X_ts": ["X_ts"],
            "X_prior": ["X_prior"],
            "y": ["y"],
        },
    )
    if "X_prior" in df.columns:
        X_index, X_ts, X_prior, y = tensors
    else:
        X_index, X_ts, y = tensors
        X_prior = None
    (X_static,) = build_dataset(
        df,
        inputs=["X_static"],
        outputs=[],
        columns_map={"X_static": STATIC_FEATURES},
        expand_sequence_columns=False,
    )
    return X_index, X_ts, X_prior, X_static, y


class PriorTimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X_index, X_ts, X_prior, X_static, y):
        self.X_index = X_index
        self.X_ts = X_ts
        self.X_prior = X_prior
        self.X_static = X_static
        self.y = y

    def __len__(self):
        return self.X_index.shape[0]

    def __getitem__(self, idx):
        xi = self.X_index[idx]
        xts = self.X_ts[idx]
        xp = self.X_prior[idx]
        xs = self.X_static[idx]
        yt = self.y[idx]

        xi_mask = torch.isfinite(xi).float()
        xts_mask = torch.isfinite(xts).float()
        xp_mask = torch.isfinite(xp).float()
        xs_mask = torch.isfinite(xs).float()
        y_mask = torch.isfinite(yt).float()
        loss_mask = y_mask * (1 - xts_mask)

        return {
            "X_index": torch.nan_to_num(xi, nan=0.0) * xi_mask,
            "X_index_mask": xi_mask,
            "X_ts": torch.nan_to_num(xts, nan=0.0) * xts_mask,
            "X_ts_mask": xts_mask,
            "X_prior": torch.nan_to_num(xp, nan=0.0) * xp_mask,
            "X_prior_mask": xp_mask,
            "X_static": torch.nan_to_num(xs, nan=0.0) * xs_mask,
            "X_static_mask": xs_mask,
            "y": torch.nan_to_num(yt, nan=0.0),
            "y_mask": y_mask,
            "loss_mask": loss_mask,
        }


def build_loader(df: pd.DataFrame, args):
    X_index, X_ts, X_prior, X_static, y = build_tensors(df)
    if X_prior is not None:
        dataset = PriorTimeSeriesDataset(
            X_index=X_index,
            X_ts=X_ts,
            X_prior=X_prior,
            X_static=X_static,
            y=y,
        )
    else:
        dataset = MaskedTimeSeriesDataset(
            X_index=X_index,
            X_ts=X_ts,
            X_static=X_static,
            y=y,
            mask_config={
                "ts_keep_prob": args.ts_keep_prob,
                "index_keep_prob": args.index_keep_prob,
                "static_keep_prob": args.static_keep_prob,
            },
        )
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


def load_model(args, static_dim: int, device):
    module = importlib.import_module(args.model_module)
    model_class = getattr(module, args.model_class)
    model = model_class(seq_len=60, static_dim=static_dim)
    state = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def masked_metrics(pred, target, mask):
    point_diff = (pred - target)[mask == 1]
    if point_diff.numel() == 0:
        return {
            "mse": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "n_points": 0,
            "diff_mse": np.nan,
            "directional_accuracy": np.nan,
            "diff_corr": np.nan,
            "n_diffs": 0,
            "range_error": np.nan,
            "volatility_error": np.nan,
            "n_ranges": 0,
        }

    mse = torch.mean(point_diff**2).item()
    mae = torch.mean(torch.abs(point_diff)).item()

    diff_mask = mask[:, 1:] * mask[:, :-1]
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    valid_pred_diff = pred_diff[diff_mask == 1]
    valid_target_diff = target_diff[diff_mask == 1]

    if valid_pred_diff.numel() > 0:
        diff_delta = valid_pred_diff - valid_target_diff
        diff_mse = torch.mean(diff_delta**2).item()
        directional_accuracy = (
            torch.sign(valid_pred_diff) == torch.sign(valid_target_diff)
        ).float().mean().item()
        if valid_pred_diff.numel() >= 2:
            pred_centered = valid_pred_diff - valid_pred_diff.mean()
            target_centered = valid_target_diff - valid_target_diff.mean()
            denom = torch.linalg.vector_norm(pred_centered) * torch.linalg.vector_norm(target_centered)
            diff_corr = ((pred_centered * target_centered).sum() / denom).item() if denom.item() > 1e-8 else np.nan
        else:
            diff_corr = np.nan
        n_diffs = int(valid_pred_diff.numel())
    else:
        diff_mse = np.nan
        directional_accuracy = np.nan
        diff_corr = np.nan
        n_diffs = 0

    range_errors = []
    volatility_errors = []
    for i in range(pred.shape[0]):
        row_mask = mask[i].bool()
        if row_mask.sum().item() < 2:
            continue
        pred_row = pred[i][row_mask]
        target_row = target[i][row_mask]
        pred_range = pred_row.max() - pred_row.min()
        target_range = target_row.max() - target_row.min()
        range_errors.append(torch.abs(pred_range - target_range))

        row_diff_mask = diff_mask[i].bool()
        if row_diff_mask.sum().item() >= 2:
            pred_row_diff = pred_diff[i][row_diff_mask]
            target_row_diff = target_diff[i][row_diff_mask]
            volatility_errors.append(torch.abs(pred_row_diff.std(unbiased=False) - target_row_diff.std(unbiased=False)))

    range_error = torch.stack(range_errors).mean().item() if range_errors else np.nan
    volatility_error = torch.stack(volatility_errors).mean().item() if volatility_errors else np.nan

    return {
        "mse": mse,
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "n_points": int(point_diff.numel()),
        "diff_mse": diff_mse,
        "directional_accuracy": directional_accuracy,
        "diff_corr": diff_corr,
        "n_diffs": n_diffs,
        "range_error": range_error,
        "volatility_error": volatility_error,
        "n_ranges": len(range_errors),
    }


def collect_group_key(df: pd.DataFrame, batch_start: int, batch_size: int):
    rows = df.iloc[batch_start : batch_start + batch_size]
    if "sparse_mode" in rows.columns:
        return rows["sparse_mode"].tolist()
    if "target_observed_points" in rows.columns:
        bins = []
        for value in rows["target_observed_points"].tolist():
            if value <= 10:
                bins.append("obs_01_10")
            elif value <= 30:
                bins.append("obs_11_30")
            elif value <= 50:
                bins.append("obs_31_50")
            else:
                bins.append("obs_51_60")
        return bins
    return ["all"] * len(rows)


def append_group_records(records, group_keys, preds, target, mask, prefix):
    for i, group in enumerate(group_keys):
        metrics = masked_metrics(preds[i : i + 1].cpu(), target[i : i + 1].cpu(), mask[i : i + 1].cpu())
        records.append({"group": group, "method": prefix, **metrics})


def model_forward(model, batch, device):
    xb_idx = batch["X_index"].to(device)
    xb_idx_m = batch["X_index_mask"].to(device)
    xb_ts = batch["X_ts"].to(device)
    xb_ts_m = batch["X_ts_mask"].to(device)
    xb_static = batch["X_static"].to(device)
    xb_static_m = batch["X_static_mask"].to(device)
    if "X_prior" in batch:
        return model(
            xb_idx,
            xb_idx_m,
            xb_ts,
            xb_ts_m,
            batch["X_prior"].to(device),
            batch["X_prior_mask"].to(device),
            xb_static,
            xb_static_m,
        )
    return model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)


def evaluate(args):
    torch.manual_seed(args.seed)
    df = load_frame(args.input_path, args.limit, args.seed)
    loader = build_loader(df, args)
    first = next(iter(loader))
    device = select_device(args.device)
    model = load_model(args, static_dim=first["X_static"].shape[-1], device=device)

    summary_rows = []
    group_rows = []
    plot_rows = []
    batch_start = 0

    print(f"device: {device} | rows: {len(df)} | ts_keep_prob={args.ts_keep_prob}")
    with torch.no_grad():
        for batch in loader:
            xb_idx = batch["X_index"].to(device)
            xb_idx_m = batch["X_index_mask"].to(device)
            xb_ts = batch["X_ts"].to(device)
            xb_ts_m = batch["X_ts_mask"].to(device)
            xb_static = batch["X_static"].to(device)
            xb_static_m = batch["X_static_mask"].to(device)
            yb = batch["y"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            pred_model = model_forward(model, batch, device)
            pred_linear = linear_baseline(xb_ts, xb_ts_m, xb_static)
            if "X_prior" in batch:
                pred_index = batch["X_prior"].to(device)
            else:
                pred_index = index_residual_baseline(xb_ts, xb_ts_m, xb_idx, xb_static)
            if not args.no_clip:
                pred_model = torch.clamp(pred_model, 0.0, 1.0)
                pred_linear = torch.clamp(pred_linear, 0.0, 1.0)
                pred_index = torch.clamp(pred_index, 0.0, 1.0)

            for name, pred in [
                ("model", pred_model),
                ("linear", pred_linear),
                ("index", pred_index),
            ]:
                summary_rows.append({"method": name, **masked_metrics(pred.cpu(), yb.cpu(), loss_mask.cpu())})

            group_keys = collect_group_key(df, batch_start, len(yb))
            append_group_records(group_rows, group_keys, pred_model, yb, loss_mask, "model")
            append_group_records(group_rows, group_keys, pred_linear, yb, loss_mask, "linear")
            append_group_records(group_rows, group_keys, pred_index, yb, loss_mask, "index")

            if len(plot_rows) < args.plot_samples:
                take = min(args.plot_samples - len(plot_rows), len(yb))
                meta = df.iloc[batch_start : batch_start + take]
                for i in range(take):
                    plot_rows.append(
                        {
                            "meta": meta.iloc[i].to_dict(),
                            "X_index": xb_idx[i].cpu().numpy(),
                            "X_ts": xb_ts[i].cpu().numpy(),
                            "X_ts_mask": xb_ts_m[i].cpu().numpy(),
                            "y": yb[i].cpu().numpy(),
                            "loss_mask": loss_mask[i].cpu().numpy(),
                            "pred_model": pred_model[i].cpu().numpy(),
                            "pred_linear": pred_linear[i].cpu().numpy(),
                            "pred_index": pred_index[i].cpu().numpy(),
                        }
                    )

            batch_start += len(yb)

    return df, pd.DataFrame(summary_rows), pd.DataFrame(group_rows), plot_rows


def weighted_average(g: pd.DataFrame, value_col: str, weight_col: str):
    valid = g[value_col].notna() & (g[weight_col] > 0)
    if not valid.any():
        return np.nan
    return np.average(g.loc[valid, value_col], weights=g.loc[valid, weight_col])


def aggregate_metric_frame(g: pd.DataFrame) -> pd.Series:
    mse = weighted_average(g, "mse", "n_points")
    return pd.Series(
        {
            "mse": mse,
            "mae": weighted_average(g, "mae", "n_points"),
            "rmse": np.sqrt(mse) if pd.notna(mse) else np.nan,
            "diff_mse": weighted_average(g, "diff_mse", "n_diffs"),
            "directional_accuracy": weighted_average(g, "directional_accuracy", "n_diffs"),
            "diff_corr": weighted_average(g, "diff_corr", "n_diffs"),
            "range_error": weighted_average(g, "range_error", "n_ranges"),
            "volatility_error": weighted_average(g, "volatility_error", "n_ranges"),
            "n_points": int(g["n_points"].sum()),
            "n_diffs": int(g["n_diffs"].sum()),
            "n_ranges": int(g["n_ranges"].sum()),
        }
    )


def aggregate_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.groupby("method").apply(aggregate_metric_frame).reset_index()


def aggregate_group_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows[rows["n_points"] > 0].copy()
    return rows.groupby(["group", "method"]).apply(
        lambda g: pd.concat([aggregate_metric_frame(g), pd.Series({"n_samples": int(len(g))})])
    ).reset_index()


def plot_predictions(plot_rows, output_path: Path):
    if not plot_rows:
        return
    rows, cols = 5, 5
    n = min(len(plot_rows), rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.5), squeeze=False)
    x = np.arange(60)

    for ax, item in zip(axes.ravel(), plot_rows[:n]):
        meta = item["meta"]

        ax.plot(x, item["X_index"], "--", color="#777777", linewidth=1.0, label="index")
        ax.plot(x, item["pred_model"], color="#ff7f0e", linewidth=1.2, label="model")
        ax.plot(x, item["pred_linear"], color="#2ca02c", linewidth=0.9, alpha=0.8, label="linear")
        ax.plot(x, item["pred_index"], color="#9467bd", linewidth=0.9, alpha=0.8, label="idx base")
        observed = item["X_ts_mask"] == 1
        hidden_eval = item["loss_mask"] == 1
        ax.scatter(x[observed], item["X_ts"][observed], color="#d62728", s=10, zorder=4)
        ax.scatter(x[hidden_eval], item["y"][hidden_eval], facecolors="none", edgecolors="#000000", s=14, zorder=4)
        ax.axhline(0, color="#aaaaaa", linewidth=0.7, alpha=0.5)
        ax.axhline(1, color="#aaaaaa", linewidth=0.7, alpha=0.5)
        title = meta.get("target_ticker", "")
        if "sparse_mode" in meta:
            title += f" | {meta['sparse_mode']}"
        if "target_observed_points" in meta:
            title += f" | obs={int(meta['target_observed_points'])}"
        title += f" | c30={meta.get('corr_30', np.nan):.2f}"
        ax.set_title(title, fontsize=7)
        ax.set_xlim(0, 59)
        ax.set_ylim(-0.15, 1.15)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.15)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    _, summary_rows, group_rows, plot_rows = evaluate(args)
    summary = aggregate_metrics(summary_rows)
    groups = aggregate_group_metrics(group_rows)

    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    groups.to_csv(output_dir / "metrics_by_group.csv", index=False)
    plot_predictions(plot_rows, output_dir / "prediction_samples.png")

    print("\nOverall metrics")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to {output_dir}")


if __name__ == "__main__":
    main()

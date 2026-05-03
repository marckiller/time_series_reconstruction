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

from src.baselines import STATIC_FEATURES, index_residual_baseline
from src.utils.dataset import MaskedTimeSeriesDataset, build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Plot model vs index-residual baseline examples.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--model-module", default="src.models.prior_correction_model")
    parser.add_argument("--model-class", default="PriorCorrectionModel")
    parser.add_argument("--tickers", default=None, help="Comma-separated target tickers to include.")
    parser.add_argument("--samples-per-ticker", type=int, default=4)
    parser.add_argument("--max-tickers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ts-keep-prob", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--anonymous", action="store_true")
    parser.add_argument("--no-clip", action="store_true")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_tickers(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def load_frame(args) -> pd.DataFrame:
    df = pd.read_parquet(args.input_path)
    tickers = parse_tickers(args.tickers)
    if tickers is not None:
        df = df[df["target_ticker"].isin(tickers)].copy()

    rng = np.random.default_rng(args.seed)
    selected_parts = []
    all_tickers = sorted(df["target_ticker"].dropna().unique())
    if len(all_tickers) > args.max_tickers:
        ticker_order = sorted(rng.choice(all_tickers, size=args.max_tickers, replace=False).tolist())
    else:
        ticker_order = all_tickers
    for ticker in ticker_order:
        part = df[df["target_ticker"] == ticker]
        if part.empty:
            continue
        take = min(args.samples_per_ticker, len(part))
        selected_parts.append(part.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))

    if not selected_parts:
        raise ValueError("No rows selected for plotting.")
    return pd.concat(selected_parts, ignore_index=True)


def build_tensors(df: pd.DataFrame):
    X_index, X_ts, y = build_dataset(
        df,
        ["X_index", "X_ts"],
        ["y"],
        {
            "X_index": ["X_index"],
            "X_ts": ["X_ts"],
            "y": ["y"],
        },
    )
    (X_static,) = build_dataset(
        df,
        inputs=["X_static"],
        outputs=[],
        columns_map={"X_static": STATIC_FEATURES},
        expand_sequence_columns=False,
    )
    return X_index, X_ts, X_static, y


def build_loader(df: pd.DataFrame, args):
    X_index, X_ts, X_static, y = build_tensors(df)
    dataset = MaskedTimeSeriesDataset(
        X_index=X_index,
        X_ts=X_ts,
        X_static=X_static,
        y=y,
        mask_config={
            "ts_keep_prob": args.ts_keep_prob,
            "index_keep_prob": 1.0,
            "static_keep_prob": 1.0,
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


def collect_predictions(df: pd.DataFrame, args):
    torch.manual_seed(args.seed)
    loader = build_loader(df, args)
    first = next(iter(loader))
    device = select_device()
    model = load_model(args, static_dim=first["X_static"].shape[-1], device=device)

    rows = []
    offset = 0
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

            pred_index = index_residual_baseline(xb_ts, xb_ts_m, xb_idx, xb_static)
            pred_index_mask = torch.ones_like(pred_index)
            try:
                pred_model = model(
                    xb_idx,
                    xb_idx_m,
                    xb_ts,
                    xb_ts_m,
                    pred_index.to(device),
                    pred_index_mask.to(device),
                    xb_static,
                    xb_static_m,
                )
            except TypeError:
                pred_model = model(xb_idx, xb_idx_m, xb_ts, xb_ts_m, xb_static, xb_static_m)
            if not args.no_clip:
                pred_model = torch.clamp(pred_model, 0.0, 1.0)
                pred_index = torch.clamp(pred_index, 0.0, 1.0)

            meta = df.iloc[offset : offset + len(yb)]
            for i in range(len(yb)):
                rows.append(
                    {
                        "meta": meta.iloc[i].to_dict(),
                        "X_index": xb_idx[i].cpu().numpy(),
                        "X_ts": xb_ts[i].cpu().numpy(),
                        "X_ts_mask": xb_ts_m[i].cpu().numpy(),
                        "y": yb[i].cpu().numpy(),
                        "loss_mask": loss_mask[i].cpu().numpy(),
                        "pred_model": pred_model[i].cpu().numpy(),
                        "pred_index": pred_index[i].cpu().numpy(),
                    }
                )
            offset += len(yb)
    return rows


def plot_rows(rows, output_path: Path, anonymous: bool):
    n = len(rows)
    single_panel = n == 1
    cols = min(4, n)
    rows_count = int(np.ceil(n / cols))
    figsize = (9.8, 5.4) if single_panel else (cols * 4.0, rows_count * 3.0)
    fig, axes = plt.subplots(rows_count, cols, figsize=figsize, squeeze=False)
    x = np.arange(60)

    ticker_alias = {}
    for item in rows:
        ticker = item["meta"].get("target_ticker", "target")
        if ticker not in ticker_alias:
            ticker_alias[ticker] = f"Instrument {len(ticker_alias) + 1}"

    for ax, item in zip(axes.ravel(), rows):
        meta = item["meta"]
        observed = item["X_ts_mask"] == 1
        hidden_eval = item["loss_mask"] == 1

        ax.plot(x, item["X_index"], "--", color="#6f6f6f", linewidth=1.1, label="index path")
        ax.plot(x, item["pred_index"], color="#7b4ab2", linewidth=1.3, alpha=0.9, label="index-residual baseline")
        ax.plot(x, item["pred_model"], color="#f28e2b", linewidth=1.9, label="prior-correction model")
        ax.scatter(x[observed], item["X_ts"][observed], color="#d62728", s=22, zorder=4, label="visible target")
        ax.scatter(
            x[hidden_eval],
            item["y"][hidden_eval],
            facecolors="none",
            edgecolors="#111111",
            s=28,
            zorder=4,
            label="hidden target",
        )
        ax.axhline(0, color="#aaaaaa", linewidth=0.7, alpha=0.5)
        ax.axhline(1, color="#aaaaaa", linewidth=0.7, alpha=0.5)

        ticker = meta.get("target_ticker", "target")
        title_ticker = ticker_alias[ticker] if anonymous else ticker
        if single_panel:
            title = (
                f"Example 1 | "
                f"visible target points={int(observed.sum())} | "
                f"hidden evaluation points={int(hidden_eval.sum())} | "
                f"30h return corr={meta.get('corr_30', np.nan):.2f}"
            )
        else:
            title = f"{title_ticker} | visible={int(observed.sum())} | c30={meta.get('corr_30', np.nan):.2f}"
        ax.set_title(title, fontsize=11 if single_panel else 8)
        ax.set_xlim(0, 59)
        ax.set_ylim(-0.12, 1.12)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.16)
        if single_panel:
            ax.set_xlabel("Minute inside interval")
            ax.set_ylabel("Normalized target price")
            ax.legend(loc="upper left", fontsize=9, ncol=2)

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    if single_panel:
        fig.tight_layout()
    else:
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(), loc="lower center", ncol=5, fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    df = load_frame(args)
    prediction_rows = collect_predictions(df, args)
    plot_rows(prediction_rows, Path(args.output_path), anonymous=args.anonymous)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()

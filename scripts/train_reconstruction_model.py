import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.dataset import MaskedTimeSeriesDataset, build_dataset
from src.utils.loss.masked_loss_functions import apply_masked_mse


STATIC_FEATURES = [
    "corr_30",
    "corr_60",
    "open_pos",
    "close_pos",
    "body_to_range",
    "direction",
    "index_open_pos",
    "index_close_pos",
    "index_body_to_range",
    "index_direction",
]

TRAIN_COLUMNS = ["X_index", "X_ts", "y", *STATIC_FEATURES]
PRIOR_COLUMNS = ["X_index", "X_ts", "X_prior", "y", *STATIC_FEATURES]


def parse_args():
    parser = argparse.ArgumentParser(description="Experimental reconstruction training.")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", default="run")

    parser.add_argument("--model-module", default="src.models.prior_correction_model")
    parser.add_argument("--model-class", default="PriorCorrectionModel")
    parser.add_argument("--pretrained-path", default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    parser.add_argument("--ts-keep-prob", type=float, default=1.0)
    parser.add_argument("--index-keep-prob", type=float, default=1.0)
    parser.add_argument("--static-keep-prob", type=float, default=1.0)

    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--diff-weight", type=float, default=0.1)
    parser.add_argument("--cosine-diff-weight", type=float, default=0.0)
    parser.add_argument("--curvature-weight", type=float, default=0.0)
    parser.add_argument("--range-weight", type=float, default=0.0)
    parser.add_argument("--volatility-weight", type=float, default=0.0)
    parser.add_argument("--pull-weight", type=float, default=0.05)
    parser.add_argument("--pull-window", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
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
    t0 = time.time()
    print(f"loading {path}", flush=True)
    df = pd.read_parquet(path)
    required = PRIOR_COLUMNS if "X_prior" in df.columns else TRAIN_COLUMNS
    df = df[required]
    print(f"loaded {len(df)} rows from {path} in {time.time() - t0:.1f}s", flush=True)
    if limit is not None and len(df) > limit:
        t0 = time.time()
        print(f"sampling {limit} rows", flush=True)
        df = df.sample(n=limit, random_state=seed)
        print(f"sampled {len(df)} rows in {time.time() - t0:.1f}s", flush=True)
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


def build_loader(df: pd.DataFrame, args, shuffle: bool):
    t0 = time.time()
    print(f"building tensors for {len(df)} rows", flush=True)
    X_index, X_ts, X_prior, X_static, y = build_tensors(df)
    print(f"built tensors in {time.time() - t0:.1f}s", flush=True)
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
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=False,
    )


def load_model(args, static_dim: int):
    module = importlib.import_module(args.model_module)
    model_class = getattr(module, args.model_class)
    model = model_class(seq_len=60, static_dim=static_dim)
    if args.pretrained_path:
        state = torch.load(args.pretrained_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"loaded pretrained weights from {args.pretrained_path}")
    return model


def masked_diff_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff_mask = mask[:, 1:] * mask[:, :-1]
    if diff_mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    return apply_masked_mse(pred_diff, target_diff, diff_mask)


def masked_curvature_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    curvature_mask = mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]
    if curvature_mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    pred_curve = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    target_curve = target[:, 2:] - 2 * target[:, 1:-1] + target[:, :-2]
    return apply_masked_mse(pred_curve, target_curve, curvature_mask)


def masked_cosine_diff_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff_mask = mask[:, 1:] * mask[:, :-1]
    valid_rows = diff_mask.sum(dim=1) >= 2
    if valid_rows.sum().item() == 0:
        return pred.new_tensor(0.0)

    pred_diff = (pred[:, 1:] - pred[:, :-1]) * diff_mask
    target_diff = (target[:, 1:] - target[:, :-1]) * diff_mask
    pred_diff = pred_diff[valid_rows]
    target_diff = target_diff[valid_rows]

    pred_norm = torch.linalg.vector_norm(pred_diff, dim=1).clamp(min=1e-8)
    target_norm = torch.linalg.vector_norm(target_diff, dim=1).clamp(min=1e-8)
    cosine = (pred_diff * target_diff).sum(dim=1) / (pred_norm * target_norm)
    return (1 - cosine).mean()


def masked_range_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid_rows = mask.sum(dim=1) >= 2
    if valid_rows.sum().item() == 0:
        return pred.new_tensor(0.0)

    pred_valid = pred[valid_rows]
    target_valid = target[valid_rows]
    mask_valid = mask[valid_rows].bool()

    pred_min = pred_valid.masked_fill(~mask_valid, float("inf")).min(dim=1).values
    pred_max = pred_valid.masked_fill(~mask_valid, float("-inf")).max(dim=1).values
    target_min = target_valid.masked_fill(~mask_valid, float("inf")).min(dim=1).values
    target_max = target_valid.masked_fill(~mask_valid, float("-inf")).max(dim=1).values

    return torch.mean((pred_min - target_min) ** 2 + (pred_max - target_max) ** 2)


def masked_volatility_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff_mask = mask[:, 1:] * mask[:, :-1]
    valid_rows = diff_mask.sum(dim=1) >= 2
    if valid_rows.sum().item() == 0:
        return pred.new_tensor(0.0)

    pred_diff = pred[:, 1:] - pred[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    losses = []
    for i in torch.where(valid_rows)[0]:
        row_mask = diff_mask[i].bool()
        pred_vol = pred_diff[i][row_mask].std(unbiased=False)
        target_vol = target_diff[i][row_mask].std(unbiased=False)
        losses.append((pred_vol - target_vol) ** 2)
    return torch.stack(losses).mean()


def pull_loss(pred: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor, observed_mask: torch.Tensor, window: int) -> torch.Tensor:
    if observed_mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    pull_mask = torch.zeros_like(target_mask)
    for offset in range(-window, window + 1):
        shifted = torch.roll(observed_mask, shifts=offset, dims=1)
        if offset < 0:
            shifted[:, offset:] = 0
        elif offset > 0:
            shifted[:, :offset] = 0
        pull_mask = torch.maximum(pull_mask, shifted)
    pull_mask = pull_mask * target_mask * (1 - observed_mask)
    if pull_mask.sum().item() == 0:
        return pred.new_tensor(0.0)
    return apply_masked_mse(pred, target, pull_mask)


def combined_loss(pred, target, loss_mask, y_mask, observed_mask, args):
    zero = pred.new_tensor(0.0)

    mse = apply_masked_mse(pred, target, loss_mask)
    diff = masked_diff_loss(pred, target, loss_mask)

    cosine_diff = (
        masked_cosine_diff_loss(pred, target, loss_mask)
        if args.cosine_diff_weight > 0
        else zero
    )
    curvature = (
        masked_curvature_loss(pred, target, loss_mask)
        if args.curvature_weight > 0
        else zero
    )
    range_ = masked_range_loss(pred, target, loss_mask) if args.range_weight > 0 else zero
    volatility = (
        masked_volatility_loss(pred, target, loss_mask)
        if args.volatility_weight > 0
        else zero
    )
    pull = (
        pull_loss(pred, target, y_mask, observed_mask, args.pull_window)
        if args.pull_weight > 0
        else zero
    )

    total = args.mse_weight * mse + args.diff_weight * diff
    if args.cosine_diff_weight > 0:
        total = total + args.cosine_diff_weight * cosine_diff
    if args.curvature_weight > 0:
        total = total + args.curvature_weight * curvature
    if args.range_weight > 0:
        total = total + args.range_weight * range_
    if args.volatility_weight > 0:
        total = total + args.volatility_weight * volatility
    if args.pull_weight > 0:
        total = total + args.pull_weight * pull

    return total, {
        "mse": mse.detach(),
        "diff": diff.detach(),
        "cosine_diff": cosine_diff.detach(),
        "curvature": curvature.detach(),
        "range": range_.detach(),
        "volatility": volatility.detach(),
        "pull": pull.detach(),
    }


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


def run_epoch(model, loader, optimizer, device, args, train: bool):
    model.train(train)
    totals = {
        "loss": 0.0,
        "mse": 0.0,
        "diff": 0.0,
        "cosine_diff": 0.0,
        "curvature": 0.0,
        "range": 0.0,
        "volatility": 0.0,
        "pull": 0.0,
    }
    batches = 0
    points = 0.0

    for batch in loader:
        xb_ts_m = batch["X_ts_mask"].to(device)
        yb = batch["y"].to(device)
        yb_mask = batch["y_mask"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        if loss_mask.sum().item() == 0:
            continue

        with torch.set_grad_enabled(train):
            pred = model_forward(model, batch, device)
            loss, parts = combined_loss(pred, yb, loss_mask, yb_mask, xb_ts_m, args)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

        totals["loss"] += float(loss.detach().cpu())
        for key in ["mse", "diff", "cosine_diff", "curvature", "range", "volatility", "pull"]:
            totals[key] += float(parts[key].cpu())
        batches += 1
        points += float(loss_mask.sum().detach().cpu())

    denom = max(batches, 1)
    return {key: value / denom for key, value in totals.items()} | {"batches": batches, "points": points}


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "args.json", vars(args))

    train_df = load_frame(args.train_path, args.limit_train, args.seed)
    val_df = load_frame(args.val_path, args.limit_val, args.seed)
    print("building train loader", flush=True)
    train_loader = build_loader(train_df, args, shuffle=True)
    print("building val loader", flush=True)
    val_loader = build_loader(val_df, args, shuffle=False)

    print("reading first batch", flush=True)
    sample = next(iter(train_loader))
    static_dim = sample["X_static"].shape[-1]
    device = select_device(args.device)

    model = load_model(args, static_dim=static_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    best_path = output_dir / "best_model.pt"

    print(f"output: {output_dir}")
    print(f"device: {device}")
    print(f"train rows: {len(train_df)} | val rows: {len(val_df)} | batch: {args.batch_size}")
    print(
        f"loss: {args.mse_weight}*mse + {args.diff_weight}*diff + "
        f"{args.cosine_diff_weight}*cosine_diff + {args.curvature_weight}*curvature + "
        f"{args.range_weight}*range + {args.volatility_weight}*volatility + {args.pull_weight}*pull"
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, args, train=False)
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "seconds": elapsed,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        print(
            f"epoch {epoch:03d} | {elapsed:6.1f}s | "
            f"train loss={train_metrics['loss']:.6f} mse={train_metrics['mse']:.6f} diff={train_metrics['diff']:.6f} | "
            f"val loss={val_metrics['loss']:.6f} mse={val_metrics['mse']:.6f} diff={val_metrics['diff']:.6f}"
        )

        improved = val_metrics["loss"] < best_val - args.early_stopping_min_delta
        if improved:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
            print(f"saved {best_path}")
        else:
            epochs_without_improvement += 1
            if args.early_stopping_patience > 0:
                print(
                    f"no val improvement for {epochs_without_improvement}/"
                    f"{args.early_stopping_patience} epochs"
                )

        if (
            args.early_stopping_patience > 0
            and epochs_without_improvement >= args.early_stopping_patience
        ):
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    print(f"done. best_val={best_val:.6f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()

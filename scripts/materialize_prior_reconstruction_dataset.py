import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.baselines import CLOSE_POS_IDX, CORR_30_IDX, CORR_60_IDX, OPEN_POS_IDX, STATIC_FEATURES


RAW_COLUMNS = ["X_index", "X_ts", "y", *STATIC_FEATURES]


def parse_args():
    parser = argparse.ArgumentParser(description="Materialize index-residual prior for reconstruction datasets.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--keep-prob", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def interpolate_index_np(index_values):
    index_values = np.asarray(index_values, dtype=float)
    length = len(index_values)
    finite = np.isfinite(index_values)
    positions = np.arange(length)

    if finite.sum() == 0:
        return np.linspace(0.0, 1.0, length)
    if finite.sum() == length:
        return index_values
    return np.interp(positions, positions[finite], index_values[finite])


def index_residual_prior_np(x_index, x_ts, static_values):
    x_index = interpolate_index_np(x_index)
    x_ts = np.asarray(x_ts, dtype=float)
    static_values = np.asarray(static_values, dtype=float)
    length = len(x_index)

    corr_values = static_values[[CORR_30_IDX, CORR_60_IDX]]
    corr_values = corr_values[np.isfinite(corr_values)]
    beta = float(np.clip(np.mean(corr_values), -1.0, 1.0)) if len(corr_values) else 0.0

    anchors = {
        0: float(static_values[OPEN_POS_IDX]),
        length - 1: float(static_values[CLOSE_POS_IDX]),
    }
    observed = np.where(np.isfinite(x_ts))[0]
    for pos in observed:
        anchors[int(pos)] = float(x_ts[pos])

    filled = np.empty(length, dtype=float)
    anchor_positions = sorted(anchors)

    for left, right in zip(anchor_positions[:-1], anchor_positions[1:]):
        segment = np.arange(left, right + 1)
        target_linear = np.linspace(anchors[left], anchors[right], len(segment))
        index_linear = np.linspace(x_index[left], x_index[right], len(segment))
        filled[segment] = target_linear + beta * (x_index[segment] - index_linear)

    first = anchor_positions[0]
    last = anchor_positions[-1]
    filled[:first] = anchors[first]
    filled[last:] = anchors[last]
    filled[np.isfinite(x_ts)] = x_ts[np.isfinite(x_ts)]
    return filled.astype(float).tolist()


def mask_target_np(x_ts, keep_prob, rng):
    x_ts = np.asarray(x_ts, dtype=float).copy()
    if keep_prob >= 1.0:
        return x_ts.tolist()
    observed = np.isfinite(x_ts)
    keep = rng.random(len(x_ts)) < keep_prob
    x_ts[observed & ~keep] = np.nan
    return x_ts.tolist()


def main():
    args = parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Use --overwrite to replace it.")

    t0 = time.time()
    df = pd.read_parquet(args.input_path, columns=RAW_COLUMNS)
    print(f"loaded {len(df)} rows from {args.input_path} in {time.time() - t0:.1f}s")

    if args.limit is not None and len(df) > args.limit:
        df = df.sample(n=args.limit, random_state=args.seed).reset_index(drop=True)
        print(f"sampled {len(df)} rows")

    rng = np.random.default_rng(args.seed)
    static_matrix = df[STATIC_FEATURES].to_numpy(dtype=float)
    priors = []
    masked_ts = []

    t0 = time.time()
    for i, row in enumerate(df.itertuples(index=False)):
        row_dict = row._asdict()
        x_ts_masked = mask_target_np(row_dict["X_ts"], keep_prob=args.keep_prob, rng=rng)
        prior = index_residual_prior_np(row_dict["X_index"], x_ts_masked, static_matrix[i])
        masked_ts.append(x_ts_masked)
        priors.append(prior)
        if (i + 1) % 100000 == 0:
            print(f"processed {i + 1} rows in {time.time() - t0:.1f}s")

    out = df.copy()
    out["X_ts"] = masked_ts
    out["X_prior"] = priors
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"saved {len(out)} rows to {output_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.preprocessing import compute_interval_metrics


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


def parse_args():
    parser = argparse.ArgumentParser(description="Build synthetic reconstruction dataset.")
    parser.add_argument("--canonical-dir", default="data/synthetic/canonical")
    parser.add_argument("--output-path", default="data/synthetic/reconstruction_dataset.parquet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-hours-per-pair", type=int, default=None)
    parser.add_argument(
        "--sparse-modes",
        nargs="+",
        default=["random", "every_5", "every_10", "every_15", "empty"],
    )
    return parser.parse_args()


def read_table(path_base: Path, columns: list[str] | None = None) -> pd.DataFrame:
    parquet = path_base.with_suffix(".parquet")
    csv = path_base.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet, columns=columns)
    if csv.exists():
        return pd.read_csv(csv, usecols=columns)
    raise FileNotFoundError(f"Missing table: {path_base}.parquet or {path_base}.csv")


def normalize_to_range(values: np.ndarray, low: float, high: float) -> list[float]:
    if pd.isna(low) or pd.isna(high) or high == low:
        return [float("nan")] * len(values)
    out = (values - low) / (high - low)
    return [float(x) if np.isfinite(x) else float("nan") for x in out]


def make_sparse_series(values: list[float], mode: str, rng: np.random.Generator) -> list[float]:
    arr = np.asarray(values, dtype=float).copy()
    mask = np.zeros(len(arr), dtype=bool)

    if mode == "empty":
        pass
    elif mode.startswith("every_"):
        step = int(mode.split("_", 1)[1])
        if step <= 0:
            raise ValueError(f"Invalid sparse mode: {mode}")
        mask[::step] = True
    elif mode == "random":
        keep_prob = rng.uniform(0.0, 0.5)
        mask = rng.random(len(arr)) < keep_prob
    else:
        raise ValueError(f"Unknown sparse mode: {mode}")

    arr[~mask] = np.nan
    return [float(x) if np.isfinite(x) else float("nan") for x in arr]


def build_hour_sequences(hour_df: pd.DataFrame, minute_df: pd.DataFrame) -> dict[pd.Timestamp, np.ndarray]:
    minute = minute_df.copy()
    minute["timestamp"] = pd.to_datetime(minute["timestamp"])
    minute = minute.set_index("timestamp")

    sequences = {}
    for ts in pd.to_datetime(hour_df["timestamp"]):
        idx = pd.date_range(ts, periods=60, freq="1min")
        values = minute.reindex(idx)["close"].to_numpy(dtype=float)
        if len(values) == 60:
            sequences[ts] = values
    return sequences


def add_returns_and_correlations(target_hour: pd.DataFrame, index_hour: pd.DataFrame) -> pd.DataFrame:
    target = target_hour.copy()
    index = index_hour.copy()
    target["timestamp"] = pd.to_datetime(target["timestamp"])
    index["timestamp"] = pd.to_datetime(index["timestamp"])

    target = target.sort_values("timestamp").reset_index(drop=True)
    index = index.sort_values("timestamp").reset_index(drop=True)

    target["ret_log"] = np.log(target["close"] / target["close"].shift(1))
    index["ret_log"] = np.log(index["close"] / index["close"].shift(1))

    merged = target[["timestamp", "ret_log"]].merge(
        index[["timestamp", "ret_log"]],
        on="timestamp",
        suffixes=("_target", "_index"),
    )
    for window in [30, 60]:
        merged[f"corr_{window}"] = (
            merged["ret_log_target"].rolling(window=window).corr(merged["ret_log_index"])
        )

    return target.merge(merged[["timestamp", "corr_30", "corr_60"]], on="timestamp", how="left")


def prepare_hour_features(target_hour: pd.DataFrame, index_hour: pd.DataFrame) -> pd.DataFrame:
    target_with_corr = add_returns_and_correlations(target_hour, index_hour)
    target_metrics = compute_interval_metrics(target_with_corr)
    index_metrics = compute_interval_metrics(index_hour).add_prefix("index_")

    df = target_with_corr.merge(target_metrics, on="timestamp", how="left")
    df = df.merge(
        index_hour.add_prefix("index_"),
        left_on="timestamp",
        right_on="index_timestamp",
        how="left",
    ).drop(columns=["index_timestamp"])
    df = df.merge(
        index_metrics,
        left_on="timestamp",
        right_on="index_timestamp",
        how="left",
    ).drop(columns=["index_timestamp"])
    return df


def build_pair_dataset(pair_row: pd.Series, canonical_dir: Path, rng: np.random.Generator, sparse_modes: list[str], max_hours: int | None) -> pd.DataFrame:
    index_ticker = pair_row["index_ticker"]
    target_ticker = pair_row["target_ticker"]

    target_minute = read_table(canonical_dir / "minute_ohlc" / target_ticker, columns=["timestamp", "close"])
    index_minute = read_table(canonical_dir / "minute_ohlc" / index_ticker, columns=["timestamp", "close"])
    target_hour = read_table(canonical_dir / "hour_ohlc" / target_ticker)
    index_hour = read_table(canonical_dir / "hour_ohlc" / index_ticker)

    target_hour["timestamp"] = pd.to_datetime(target_hour["timestamp"])
    index_hour["timestamp"] = pd.to_datetime(index_hour["timestamp"])
    target_hour = target_hour[target_hour["expected_minutes"] == 60].copy()
    index_hour = index_hour[index_hour["expected_minutes"] == 60].copy()

    features = prepare_hour_features(target_hour, index_hour)
    target_sequences = build_hour_sequences(target_hour, target_minute)
    index_sequences = build_hour_sequences(index_hour, index_minute)

    rows = []
    for row in features.itertuples(index=False):
        ts = row.timestamp
        if ts not in target_sequences or ts not in index_sequences:
            continue
        if any(pd.isna(getattr(row, name)) for name in STATIC_FEATURES):
            continue

        y = normalize_to_range(target_sequences[ts], row.low, row.high)
        x_index = normalize_to_range(index_sequences[ts], row.index_low, row.index_high)
        if all(pd.isna(v) for v in y) or all(pd.isna(v) for v in x_index):
            continue

        sparse_mode = str(rng.choice(sparse_modes))
        x_ts = make_sparse_series(y, sparse_mode, rng)

        rows.append(
            {
                "timestamp": ts,
                "target_ticker": target_ticker,
                "index_ticker": index_ticker,
                "X_index": x_index,
                "X_ts": x_ts,
                "y": y,
                "sparse_mode": sparse_mode,
                "target_corr_config": pair_row["target_corr"],
                "target_sigma_config": pair_row["target_sigma"],
                **{name: getattr(row, name) for name in STATIC_FEATURES},
            }
        )

        if max_hours is not None and len(rows) >= max_hours:
            break

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    canonical_dir = Path(args.canonical_dir)
    summary = read_table(canonical_dir / "synthetic_generation_summary")
    if args.max_pairs is not None:
        summary = summary.head(args.max_pairs)
    rng = np.random.default_rng(args.seed)

    parts = []
    for pair in summary.itertuples(index=False):
        df = build_pair_dataset(
            pd.Series(pair._asdict()),
            canonical_dir=canonical_dir,
            rng=rng,
            sparse_modes=args.sparse_modes,
            max_hours=args.max_hours_per_pair,
        )
        parts.append(df)
        print(f"{pair.target_ticker}: {len(df)} samples")

    dataset = pd.concat(parts, ignore_index=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)
    print(f"\nSaved {len(dataset)} samples to {output_path}")
    print(dataset["sparse_mode"].value_counts().to_string())


if __name__ == "__main__":
    main()

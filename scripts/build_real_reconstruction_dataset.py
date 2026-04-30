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
    parser = argparse.ArgumentParser(description="Build real reconstruction dataset from canonical OHLC files.")
    parser.add_argument("--canonical-dir", default="data/canonical")
    parser.add_argument("--output-path", default="data/real/reconstruction_dataset.parquet")
    parser.add_argument("--index-ticker", default="WIG20")
    parser.add_argument("--min-index-observed-minutes", type=int, default=60)
    parser.add_argument("--min-target-observed-minutes", type=int, default=1)
    parser.add_argument("--max-tickers", type=int, default=None)
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


def build_ticker_dataset(
    target_ticker: str,
    index_ticker: str,
    canonical_dir: Path,
    index_minute: pd.DataFrame,
    index_hour: pd.DataFrame,
    min_index_observed_minutes: int,
    min_target_observed_minutes: int,
) -> pd.DataFrame:
    target_minute = read_table(canonical_dir / "minute_ohlc" / target_ticker, columns=["timestamp", "close"])
    target_hour = read_table(canonical_dir / "hour_ohlc" / target_ticker)

    target_hour["timestamp"] = pd.to_datetime(target_hour["timestamp"])
    target_hour = target_hour[
        (target_hour["expected_minutes"] == 60)
        & (target_hour["observed_minutes"] >= min_target_observed_minutes)
        & target_hour["open"].notna()
        & target_hour["high"].notna()
        & target_hour["low"].notna()
        & target_hour["close"].notna()
    ].copy()

    usable_index_hour = index_hour[
        (index_hour["expected_minutes"] == 60)
        & (index_hour["observed_minutes"] >= min_index_observed_minutes)
        & index_hour["open"].notna()
        & index_hour["high"].notna()
        & index_hour["low"].notna()
        & index_hour["close"].notna()
    ].copy()

    features = prepare_hour_features(target_hour, usable_index_hour)
    target_sequences = build_hour_sequences(target_hour, target_minute)
    index_sequences = build_hour_sequences(usable_index_hour, index_minute)

    rows = []
    for row in features.itertuples(index=False):
        ts = row.timestamp
        if ts not in target_sequences or ts not in index_sequences:
            continue
        if any(pd.isna(getattr(row, name)) for name in STATIC_FEATURES):
            continue

        y = normalize_to_range(target_sequences[ts], row.low, row.high)
        x_index = normalize_to_range(index_sequences[ts], row.index_low, row.index_high)
        if all(pd.isna(v) for v in y) or any(pd.isna(v) for v in x_index):
            continue

        rows.append(
            {
                "timestamp": ts,
                "target_ticker": target_ticker,
                "index_ticker": index_ticker,
                "X_index": x_index,
                "X_ts": y,
                "y": y,
                "target_observed_points": int(np.isfinite(np.asarray(y, dtype=float)).sum()),
                **{name: getattr(row, name) for name in STATIC_FEATURES},
            }
        )

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    canonical_dir = Path(args.canonical_dir)

    index_minute = read_table(canonical_dir / "minute_ohlc" / args.index_ticker, columns=["timestamp", "close"])
    index_hour = read_table(canonical_dir / "hour_ohlc" / args.index_ticker)
    index_hour["timestamp"] = pd.to_datetime(index_hour["timestamp"])

    target_paths = sorted((canonical_dir / "hour_ohlc").glob("*.parquet"))
    if not target_paths:
        target_paths = sorted((canonical_dir / "hour_ohlc").glob("*.csv"))

    target_tickers = [p.stem for p in target_paths if p.stem != args.index_ticker]
    if args.max_tickers is not None:
        target_tickers = target_tickers[: args.max_tickers]

    parts = []
    for ticker in target_tickers:
        df = build_ticker_dataset(
            target_ticker=ticker,
            index_ticker=args.index_ticker,
            canonical_dir=canonical_dir,
            index_minute=index_minute,
            index_hour=index_hour,
            min_index_observed_minutes=args.min_index_observed_minutes,
            min_target_observed_minutes=args.min_target_observed_minutes,
        )
        parts.append(df)
        print(f"{ticker}: {len(df)} samples")

    if not parts:
        raise ValueError("No target datasets were built.")

    dataset = pd.concat(parts, ignore_index=True)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output_path, index=False)

    print(f"\nSaved {len(dataset)} samples to {output_path}")
    print(dataset.groupby("target_ticker")["target_observed_points"].agg(["count", "mean", "min", "max"]).to_string())


if __name__ == "__main__":
    main()

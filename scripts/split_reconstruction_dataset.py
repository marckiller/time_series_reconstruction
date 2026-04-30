import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description="Split reconstruction dataset into train/val/test files.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["date", "random"], required=True)
    parser.add_argument("--train-end", default=None, help="Date split: train rows have timestamp < train-end.")
    parser.add_argument("--val-end", default=None, help="Date split: val rows have train-end <= timestamp < val-end.")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Random split train fraction.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def write_split(df: pd.DataFrame, output_dir: Path, name: str):
    path = output_dir / f"{name}.parquet"
    df.to_parquet(path, index=False)
    print(f"{name}: {len(df)} rows -> {path}")


def split_by_date(df: pd.DataFrame, train_end: str, val_end: str):
    if train_end is None or val_end is None:
        raise ValueError("--train-end and --val-end are required for date mode")

    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    train_end_ts = pd.to_datetime(train_end)
    val_end_ts = pd.to_datetime(val_end)

    train = work[work["timestamp"] < train_end_ts]
    val = work[(work["timestamp"] >= train_end_ts) & (work["timestamp"] < val_end_ts)]
    test = work[work["timestamp"] >= val_end_ts]
    return {"train": train, "val": val, "test": test}


def split_random(df: pd.DataFrame, train_frac: float, seed: int):
    if not 0.0 < train_frac < 1.0:
        raise ValueError("--train-frac must be between 0 and 1")
    train, val = train_test_split(df, train_size=train_frac, random_state=seed, shuffle=True)
    return {"train": train, "val": val}


def main():
    args = parse_args()
    df = pd.read_parquet(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "date":
        splits = split_by_date(df, args.train_end, args.val_end)
    else:
        splits = split_random(df, args.train_frac, args.seed)

    for name, split_df in splits.items():
        write_split(split_df, output_dir, name)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Create synthetic canonical minute/hour OHLC files.")
    parser.add_argument("--output-dir", default="data/synthetic/canonical")
    parser.add_argument("--n-hours", type=int, default=20_000)
    parser.add_argument("--start-time", default="2015-01-01 09:00:00")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-sigma", type=float, default=0.0006)
    parser.add_argument("--target-sigmas", type=float, nargs="+", default=[0.0008, 0.0012])
    parser.add_argument("--correlations", type=float, nargs="+", default=[0.2, 0.5, 0.8])
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    return parser.parse_args()


def generate_correlated_log_returns(
    n: int,
    corr: float,
    index_sigma: float,
    target_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not -1.0 <= corr <= 1.0:
        raise ValueError("corr must be in [-1, 1]")

    z_index = rng.normal(0.0, 1.0, size=n)
    z_noise = rng.normal(0.0, 1.0, size=n)
    z_target = corr * z_index + np.sqrt(max(0.0, 1.0 - corr**2)) * z_noise
    return index_sigma * z_index, target_sigma * z_target


def returns_to_prices(log_returns: np.ndarray, initial_price: float) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(log_returns)])
    return initial_price * np.exp(cumulative)


def prices_to_minute_ohlc(
    prices: np.ndarray,
    timestamps: pd.DatetimeIndex,
    ticker: str,
) -> pd.DataFrame:
    opens = prices[:-1]
    closes = prices[1:]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "ticker": ticker,
            "open": opens,
            "high": np.maximum(opens, closes),
            "low": np.minimum(opens, closes),
            "close": closes,
            "volume": np.nan,
            "record_count": 1,
        }
    )


def aggregate_to_hour(minute_df: pd.DataFrame) -> pd.DataFrame:
    ticker = minute_df["ticker"].iloc[0]
    work = minute_df.set_index("timestamp")
    hour = work.resample("1h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        record_count=("record_count", "sum"),
    )
    hour["observed_minutes"] = work["record_count"].resample("1h").size()
    hour["expected_minutes"] = 60
    hour.insert(0, "ticker", ticker)
    return hour.reset_index()


def write_table(df: pd.DataFrame, path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    else:
        df.to_csv(path.with_suffix(".csv"), index=False)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    n_minutes = args.n_hours * 60
    timestamps = pd.date_range(args.start_time, periods=n_minutes, freq="1min")

    summary_rows = []
    pair_id = 0
    for target_sigma in args.target_sigmas:
        for corr in args.correlations:
            pair_id += 1
            index_ticker = f"SYN_INDEX_{pair_id:03d}"
            target_ticker = f"SYN_TARGET_{pair_id:03d}"

            index_returns, target_returns = generate_correlated_log_returns(
                n=n_minutes,
                corr=corr,
                index_sigma=args.index_sigma,
                target_sigma=target_sigma,
                rng=rng,
            )
            index_prices = returns_to_prices(index_returns, initial_price=100.0)
            target_prices = returns_to_prices(target_returns, initial_price=50.0)

            index_minute = prices_to_minute_ohlc(index_prices, timestamps, index_ticker)
            target_minute = prices_to_minute_ohlc(target_prices, timestamps, target_ticker)
            index_hour = aggregate_to_hour(index_minute)
            target_hour = aggregate_to_hour(target_minute)

            write_table(index_minute, output_dir / "minute_ohlc" / index_ticker, args.format)
            write_table(target_minute, output_dir / "minute_ohlc" / target_ticker, args.format)
            write_table(index_hour, output_dir / "hour_ohlc" / index_ticker, args.format)
            write_table(target_hour, output_dir / "hour_ohlc" / target_ticker, args.format)

            realized_corr = float(np.corrcoef(index_returns, target_returns)[0, 1])
            summary_rows.append(
                {
                    "pair_id": pair_id,
                    "index_ticker": index_ticker,
                    "target_ticker": target_ticker,
                    "n_hours": args.n_hours,
                    "n_minutes": n_minutes,
                    "index_sigma": args.index_sigma,
                    "target_sigma": target_sigma,
                    "target_corr": corr,
                    "realized_corr": realized_corr,
                }
            )
            print(
                f"{target_ticker}: corr={corr:.2f}, realized={realized_corr:.3f}, "
                f"target_sigma={target_sigma:g}"
            )

    summary = pd.DataFrame(summary_rows)
    write_table(summary, output_dir / "synthetic_generation_summary", args.format)
    print(f"\nSaved synthetic canonical OHLC to {output_dir}")


if __name__ == "__main__":
    main()

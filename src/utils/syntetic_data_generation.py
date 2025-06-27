import pandas as pd
from datetime import timedelta
import numpy as np
import random
import string

def generate_correlated_returns(
    n_ticks: int,
    target_corr: float,
    sigma_index: float,
    sigma_instr: float,
    seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two correlated return series of length `n_ticks`.

    Parameters:
        n_ticks (int): Number of return values to generate.
        target_corr (float): Desired Pearson correlation coefficient between the two series.
        sigma_index (float): Standard deviation of returns for the index.
        sigma_instr (float): Standard deviation of returns for the instrument.
        seed (int | None): Random seed for reproducibility.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of (index_returns, instr_returns),
        both of length `n_ticks`.
    """
    if seed is not None:
        np.random.seed(seed)

    if abs(target_corr) >= 0.999999:
        base = np.random.normal(0, sigma_index, size=n_ticks)
        other = target_corr * (sigma_instr/sigma_index) * base
        index_returns = base
        instr_returns = other
    else:
        u1 = np.random.normal(0, 1, size=n_ticks)
        u2 = np.random.normal(0, 1, size=n_ticks)
        correlated_u2 = target_corr * u1 + np.sqrt(1 - target_corr**2) * u2
        index_returns = sigma_index * u1
        instr_returns = sigma_instr * correlated_u2
    return index_returns, instr_returns

def returns_to_prices(
    returns: np.ndarray,
    return_type: str = "additive",
    initial_price: float = 100.0
) -> np.ndarray:
    """
    Convert a return series into a price series based on the specified return type.

    Parameters:
        returns (np.ndarray): Series of returns.
        return_type (str): Type of returns; either "additive" for simple returns
                           or "log" for logarithmic returns.
        initial_price (float): Starting price for the series.

    Returns:
        np.ndarray: Reconstructed price series based on the returns.
    """
    returns = np.asarray(returns)
    prices = np.empty_like(returns)
    prices[0] = initial_price
    if return_type == "log":
        cum_log_returns = np.cumsum(returns)
        prices = initial_price * np.exp(cum_log_returns)
    else:
        cum_growth_factors = np.cumprod(1 + returns)
        prices = initial_price * cum_growth_factors
    return prices

def prices_to_ohlc(prices: np.ndarray, ticks_per_interval: int) -> list[tuple[float, float, float, float]]:
    """
    Convert a full price series into OHLC values for each interval.

    Parameters:
        prices (np.ndarray): Full price series.
        ticks_per_interval (int): Number of ticks in each OHLC interval.

    Returns:
        list[tuple[float, float, float, float]]: List of OHLC tuples (open, high, low, close) per interval.
    """
    ohlc = []
    n_intervals = len(prices) // ticks_per_interval
    for i in range(n_intervals):
        interval_prices = prices[i * ticks_per_interval:(i + 1) * ticks_per_interval]
        open_price = interval_prices[0]
        high_price = np.max(interval_prices)
        low_price = np.min(interval_prices)
        close_price = interval_prices[-1]
        ohlc.append((open_price, high_price, low_price, close_price))
    return ohlc

def generate_timestamps(
    n_intervals: int,
    start_time: str,
    time_interval: str | pd.Timedelta | timedelta | int | float
) -> pd.DatetimeIndex:
    """
    Generate a list of timestamps for each OHLC interval.

    Parameters:
        n_intervals (int): Number of timestamps to generate.
        start_time (str): Starting time as a string parsable by pandas.
        time_interval (str | pd.Timedelta | timedelta | int | float): Interval between timestamps.
            Can be a pandas offset string (e.g., '1min'), a Timedelta, or a number (interpreted as seconds).

    Returns:
        pd.DatetimeIndex: Series of timestamps.
    """
    start = pd.to_datetime(start_time)
    if isinstance(time_interval, str):
        timestamps = pd.date_range(start, periods=n_intervals, freq=time_interval)
    elif isinstance(time_interval, (pd.Timedelta, timedelta)):
        freq = pd.Timedelta(time_interval)
        timestamps = pd.date_range(start, periods=n_intervals, freq=freq)
    else:
        freq = pd.Timedelta(seconds=time_interval)
        timestamps = pd.date_range(start, periods=n_intervals, freq=freq)
    return timestamps


def generate_synthetic_ohlc(
    n_intervals: int,
    ticks_per_interval: int,
    time_interval: str | pd.Timedelta | timedelta | int | float,
    target_corr: float,
    sigma_index: float,
    sigma_instr: float,
    start_time: str,
    return_type: str = "additive",
    seed: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic OHLC time series for two correlated instruments.

    Parameters:
        n_intervals (int): Number of OHLC intervals to generate.
        ticks_per_interval (int): Number of ticks within each OHLC interval.
        time_interval (str | pd.Timedelta | timedelta | int | float): Interval between OHLC bars.
        target_corr (float): Desired correlation between the return series of the two instruments.
        sigma_index (float): Standard deviation of index returns.
        sigma_instr (float): Standard deviation of instrument returns.
        start_time (str): Start timestamp for the time series (parsable by pandas).
        return_type (str): Return type; "additive" or "log". Defaults to "additive".
        seed (int | None): Optional random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A pair of DataFrames with OHLC data for the index and the instrument.
    """
    n_ticks = n_intervals * ticks_per_interval
    idx_returns, instr_returns = generate_correlated_returns(
        n_ticks, target_corr, sigma_index, sigma_instr, seed)
    
    index_prices = returns_to_prices(idx_returns, return_type=return_type, initial_price=100.0)
    instr_prices = returns_to_prices(instr_returns, return_type=return_type, initial_price=100.0)
    
    ohlc_index = prices_to_ohlc(index_prices, ticks_per_interval)
    ohlc_instr = prices_to_ohlc(instr_prices, ticks_per_interval)
    
    timestamps = generate_timestamps(n_intervals, start_time, time_interval)
    
    index_df = pd.DataFrame(ohlc_index, columns=["open", "high", "low", "close"])
    instrument_df = pd.DataFrame(ohlc_instr, columns=["open", "high", "low", "close"])
    index_df.insert(0, "timestamp", timestamps)
    instrument_df.insert(0, "timestamp", timestamps)
    
    return index_df, instrument_df

def aggregate_ohlc(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Aggregate OHLC time series to a higher-level interval (e.g., from minute to hourly data).

    Parameters:
        df (pd.DataFrame): DataFrame containing OHLC data with a 'timestamp' column.
        interval (str): Pandas offset string defining the aggregation interval (e.g., '1h', '15min', '1d').

    Returns:
        pd.DataFrame: Aggregated OHLC data with new timestamps as index.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    agg = df.resample(interval).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    return agg.reset_index()

def extract_time_series_per_interval(
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    price_column: str = "close",
    include_first_open: bool = True,
    fill_value: float = np.nan,
    high_percentile: float = 90.0,
    low_percentile: float = 10.0,
    add_min_max: bool = True,
    add_dist_to_extremes: bool = True,
    add_percentile_masks: bool = True,
    add_norm_local: bool = True,
    add_norm_global: bool = True,
    add_heat: bool = True
) -> pd.DataFrame:
    from scipy.special import expit

    result, masks = [], []
    min_values, max_values = [], []
    dist_to_max_series = []
    dist_to_min_series = []
    near_max_masks = []
    near_min_masks = []
    norm_local_series = []
    norm_global_series = []
    heat_max_series = []
    heat_min_series = []

    low_df = low_df.copy()
    low_df['timestamp'] = pd.to_datetime(low_df['timestamp'])
    low_df.set_index('timestamp', inplace=True)
    high_df = high_df.copy()
    high_df['timestamp'] = pd.to_datetime(high_df['timestamp'])

    step = high_df['timestamp'].diff().median() or pd.Timedelta("1h")
    expected_len = int(round(step.total_seconds() / 60))
    if include_first_open:
        expected_len += 1

    for _, row in high_df.iterrows():
        start_time = pd.to_datetime(row['timestamp'])
        end_time = start_time + step
        interval_df = low_df.loc[(low_df.index >= start_time) & (low_df.index < end_time)]

        series = interval_df[price_column].tolist()
        if include_first_open and not interval_df.empty:
            series = [interval_df['open'].iloc[0]] + series

        mask = [1] * len(series)
        if len(series) < expected_len:
            pad_len = expected_len - len(series)
            series += [fill_value] * pad_len
            mask += [0] * pad_len
        elif len(series) > expected_len:
            series = series[:expected_len]
            mask = mask[:expected_len]

        result.append(series)
        masks.append(mask)

        valid_series = [v for v, m in zip(series, mask) if m == 1]
        min_val = np.min(valid_series) if valid_series else np.nan
        max_val = np.max(valid_series) if valid_series else np.nan

        if add_min_max:
            min_values.append(min_val)
            max_values.append(max_val)

        if add_dist_to_extremes:
            max_indices = [i for i, v in enumerate(series) if v == max_val]
            min_indices = [i for i, v in enumerate(series) if v == min_val]
            dist_max = [min([abs(i - m) for m in max_indices]) / (expected_len - 1) if mask[i] else np.nan for i in range(expected_len)]
            dist_min = [min([abs(i - m) for m in min_indices]) / (expected_len - 1) if mask[i] else np.nan for i in range(expected_len)]
            dist_to_max_series.append(dist_max)
            dist_to_min_series.append(dist_min)

        if add_percentile_masks:
            if valid_series:
                p_high = np.percentile(valid_series, high_percentile)
                p_low = np.percentile(valid_series, low_percentile)
            else:
                p_high, p_low = np.nan, np.nan

            near_max = [1 if (v is not np.nan and m == 1 and v >= p_high) else 0 for v, m in zip(series, mask)]
            near_min = [1 if (v is not np.nan and m == 1 and v <= p_low) else 0 for v, m in zip(series, mask)]
            near_max_masks.append(near_max)
            near_min_masks.append(near_min)

        if add_norm_local or add_heat:
            norm_local = [
                (v - min_val) / (max_val - min_val) if m == 1 and max_val != min_val else np.nan
                for v, m in zip(series, mask)
            ]
            if add_norm_local:
                norm_local_series.append(norm_local)

        if add_norm_global:
            global_low = interval_df['low'].min() if not interval_df.empty else np.nan
            global_high = interval_df['high'].max() if not interval_df.empty else np.nan
            norm_global = [
                (v - global_low) / (global_high - global_low)
                if m == 1 and global_high != global_low else np.nan
                for v, m in zip(series, mask)
            ]
            norm_global_series.append(norm_global)

        if add_heat:
            alpha = 10
            threshold = 0.9
            heat_max = [expit(alpha * (nv - threshold)) if nv is not np.nan else np.nan for nv in norm_local]
            heat_min = [expit(-alpha * (nv - (1 - threshold))) if nv is not np.nan else np.nan for nv in norm_local]
            heat_max_series.append(heat_max)
            heat_min_series.append(heat_min)

    data = {
        "timestamp": high_df['timestamp'],
        "series": result,
        "mask": masks,
    }
    if add_min_max:
        data["min_value"] = min_values
        data["max_value"] = max_values
    if add_dist_to_extremes:
        data["dist_to_max"] = dist_to_max_series
        data["dist_to_min"] = dist_to_min_series
    if add_percentile_masks:
        data["near_max_mask"] = near_max_masks
        data["near_min_mask"] = near_min_masks
    if add_norm_local:
        data["norm_price_local"] = norm_local_series
    if add_norm_global:
        data["norm_price_global"] = norm_global_series
    if add_heat:
        data["heat_max"] = heat_max_series
        data["heat_min"] = heat_min_series

    return pd.DataFrame(data)

def compute_returns(
    df: pd.DataFrame,
    price_column: str = "close",
    return_type: str = "additive",
    output_column: str = "ret"
) -> pd.DataFrame:
    """
    Compute returns from a price series in a DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing price data with a 'timestamp' column.
        price_column (str): Column name containing the price values.
        return_type (str): Type of returns to compute; either "additive" or "log".
        output_column (str): Name of the column to store the computed returns.

    Returns:
        pd.DataFrame: DataFrame with computed returns in the specified output column.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    if return_type == "log":
        df[output_column] = np.log(df[price_column] / df[price_column].shift(1))
    else:
        df[output_column] = df[price_column].pct_change()

    return df.reset_index()

def compute_rolling_correlation(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column1: str,
    column2: str,
    window: int,
    output_column: str = "corr"
) -> pd.DataFrame:
    """
    Compute rolling correlation between two time series columns from two DataFrames.

    Parameters:
        df1 (pd.DataFrame): First DataFrame with 'timestamp' and specified column1.
        df2 (pd.DataFrame): Second DataFrame with 'timestamp' and specified column2.
        column1 (str): Column name in df1 to use for correlation.
        column2 (str): Column name in df2 to use for correlation.
        window (int): Rolling window size.
        output_column (str): Name of the output column for correlation.

    Returns:
        pd.DataFrame: DataFrame with 'timestamp' and rolling correlation column.
    """
    df1 = df1.copy()
    df2 = df2.copy()
    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
    df1.set_index('timestamp', inplace=True)
    df2.set_index('timestamp', inplace=True)
    df1_renamed = df1[[column1]].rename(columns={column1: "col1"})
    df2_renamed = df2[[column2]].rename(columns={column2: "col2"})
    merged = df1_renamed.join(df2_renamed, how='inner')
    merged[output_column] = merged["col1"].rolling(window).corr(merged["col2"])
    return merged[[output_column]].reset_index()

def compute_interval_metrics(
    df: pd.DataFrame,
    include: list[str] = None
) -> pd.DataFrame:
    """
    Compute various metrics for each OHLC interval in the DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close'].
        include (list[str], optional): List of metric names to include. If None, compute all available.

    Available metrics:
        - 'range': high - low
        - 'open_pos': (open - low) / (high - low)
        - 'close_pos': (close - low) / (high - low)
        - 'rel_volatility': (high - low) / close
        - 'mid_price': (high + low) / 2
        - 'body_size': abs(close - open)
        - 'upper_shadow': high - max(open, close)
        - 'lower_shadow': min(open, close) - low
        - 'body_to_range': abs(close - open) / (high - low)
        - 'direction': np.sign(close - open)
        - 'volatility_ratio': (high - low) / abs(close - open)

    Returns:
        pd.DataFrame: DataFrame with the original 'timestamp' and selected metric columns.
    """
    df = df.copy()
    metrics = {}

    # Range
    if include is None or 'range' in include:
        metrics['range'] = df['high'] - df['low']
    # Open position in range
    if include is None or 'open_pos' in include:
        denom = (df['high'] - df['low']).replace(0, np.nan)
        metrics['open_pos'] = (df['open'] - df['low']) / denom
    # Close position in range
    if include is None or 'close_pos' in include:
        denom = (df['high'] - df['low']).replace(0, np.nan)
        metrics['close_pos'] = (df['close'] - df['low']) / denom
    # Relative volatility
    if include is None or 'rel_volatility' in include:
        denom = df['close'].replace(0, np.nan)
        metrics['rel_volatility'] = (df['high'] - df['low']) / denom
    # Mid price
    if include is None or 'mid_price' in include:
        metrics['mid_price'] = (df['high'] + df['low']) / 2
    # Body size
    if include is None or 'body_size' in include:
        metrics['body_size'] = (df['close'] - df['open']).abs()
    # Upper shadow
    if include is None or 'upper_shadow' in include:
        metrics['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    # Lower shadow
    if include is None or 'lower_shadow' in include:
        metrics['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    # Body to range
    if include is None or 'body_to_range' in include:
        denom = (df['high'] - df['low']).replace(0, np.nan)
        metrics['body_to_range'] = (df['close'] - df['open']).abs() / denom
    # Direction (+1 for up, -1 for down, 0 for doji)
    if include is None or 'direction' in include:
        metrics['direction'] = np.sign(df['close'] - df['open'])
    # Volatility ratio: range divided by body size
    if include is None or 'volatility_ratio' in include:
        denom = (df['close'] - df['open']).abs().replace(0, np.nan)
        metrics['volatility_ratio'] = (df['high'] - df['low']) / denom

    metrics_df = pd.DataFrame(metrics)
    metrics_df.insert(0, "timestamp", df["timestamp"].values)
    return metrics_df

def compare_series_minmax_to_ohlc(time_series_df: pd.DataFrame, ohlc_df: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(time_series_df, ohlc_df, on="timestamp", how="inner")
    denom = (merged["high"] - merged["low"]).replace(0, np.nan)

    min_pos = (merged["min_value"] - merged["low"]) / denom
    max_pos = (merged["max_value"] - merged["low"]) / denom
    range_covered = (merged["max_value"] - merged["min_value"]) / denom

    min_hit = (min_pos.fillna(-1).round(6) == 0.0).astype(int)
    max_hit = (max_pos.fillna(-1).round(6) == 1.0).astype(int)

    max_distance_to_high = (merged["high"] - merged["max_value"]) / denom
    avg_distance_to_bounds = (min_pos + max_distance_to_high) / 2

    return pd.DataFrame({
        "timestamp": merged["timestamp"],
        "min_pos": min_pos,
        "max_pos": max_pos,
        "range_covered": range_covered,
        "min_hit": min_hit,
        "max_hit": max_hit,
        "max_distance_to_high": max_distance_to_high,
        "avg_distance_to_bounds": avg_distance_to_bounds
    })

def add_normalized_series(
    ts_df: pd.DataFrame,
    ohlc_df: pd.DataFrame,
    price_key: str = "series"
) -> pd.DataFrame:
    norm_price_local = []
    norm_price_global = []

    for i in range(len(ts_df)):
        series = ts_df.iloc[i][price_key]
        mask = ts_df.iloc[i]["mask"]

        min_val = ts_df.iloc[i]["min_value"]
        max_val = ts_df.iloc[i]["max_value"]
        low = ohlc_df.iloc[i]["low"]
        high = ohlc_df.iloc[i]["high"]

        if max_val != min_val:
            local = [(v - min_val) / (max_val - min_val) if m == 1 else np.nan for v, m in zip(series, mask)]
        else:
            local = [np.nan for _ in series]
        norm_price_local.append(local)

        if high != low:
            global_ = [(v - low) / (high - low) if m == 1 else np.nan for v, m in zip(series, mask)]
        else:
            global_ = [np.nan for _ in series]
        norm_price_global.append(global_)

    ts_df = ts_df.copy()
    ts_df["norm_price_local"] = norm_price_local
    ts_df["norm_price_global"] = norm_price_global

    return ts_df

def merge_all_features(
    time_series_index,
    hour_ohlc_index,
    corr30h,
    corr60h,
    corr60d,
    hour_ohlc_index_metrics,
    hour_ohlc_instr_metrics,
    index_min_max_to_ohlc
):
    from functools import reduce

    dfs = [
        time_series_index,
        hour_ohlc_index,
        corr30h,
        corr60h,
        corr60d,
        hour_ohlc_index_metrics.add_prefix("index_"),
        hour_ohlc_instr_metrics.add_prefix("instr_"),
        index_min_max_to_ohlc
    ]

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='timestamp', how='inner'), dfs)
    return merged_df

def append_to_master_dataframe(master_df: pd.DataFrame | None, df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a single dataframe `df` to the accumulated `master_df`.
    If `master_df` is None or empty, return a copy of `df`.
    """
    if master_df is None or master_df.empty:
        return df.copy()
    return pd.concat([master_df, df], ignore_index=True)

def build_summary_dataframe(
    hour_ohlc_instr: pd.DataFrame,
    time_series_instr: pd.DataFrame,
    corr30h: pd.DataFrame,
    corr60h: pd.DataFrame,
    corr60d: pd.DataFrame | None = None,
    hour_ohlc_instr_metrics: pd.DataFrame | None = None,
    hour_ohlc_index: pd.DataFrame | None = None,
    time_series_index: pd.DataFrame | None = None,
    hour_ohlc_index_metrics: pd.DataFrame | None = None,
    index_min_max_to_ohlc: pd.DataFrame | None = None
) -> pd.DataFrame:
    """
    Create a summary DataFrame containing:
    - Instrument OHLC and time series info,
    - Rolling correlations with index,
    - Index OHLC, time series and derived metrics (all prefixed with 'index_').
    """

    # Step 1: base — copy OHLC data and add a random 5-letter ticker
    df = hour_ohlc_instr.copy()
    df['ticker'] = ''.join(random.choices(string.ascii_uppercase, k=5))

    # Step 2: merge instrument time series
    time_series_cols = ['timestamp', 'series']
    for col in ['mask', 'min_value', 'max_value', 'norm_price_local', 'norm_price_global']:
        if col in time_series_instr.columns:
            time_series_cols.append(col)
    df = df.merge(time_series_instr[time_series_cols], on='timestamp', how='left')

    # Step 3: merge correlation features
    df = df.merge(corr30h[['timestamp', 'rolling_corr_h']], on='timestamp', how='left')
    df = df.merge(corr60h[['timestamp', 'rolling_corr_60h']], on='timestamp', how='left')
    if corr60d is not None and 'rolling_corr_60d' in corr60d.columns:
        df = df.merge(corr60d[['timestamp', 'rolling_corr_60d']], on='timestamp', how='left')

    # Step 4: merge instrument OHLC metrics
    if hour_ohlc_instr_metrics is not None:
        df = df.merge(
            hour_ohlc_instr_metrics[['timestamp', 'open_pos', 'close_pos', 'body_to_range']],
            on='timestamp', how='left'
        )

    # Step 5: merge index OHLC (prefix: index_)
    if hour_ohlc_index is not None:
        df = df.merge(
            hour_ohlc_index.add_prefix('index_'),
            left_on='timestamp', right_on='index_timestamp', how='left'
        ).drop(columns=['index_timestamp'])

    # Step 6: merge index time series features (prefix: index_)
    if time_series_index is not None:
        index_cols = ['timestamp', 'series']
        optional_index_cols = [
            'mask', 'min_value', 'max_value', 'dist_to_max', 'dist_to_min',
            'near_max_mask', 'near_min_mask', 'norm_price_local',
            'norm_price_global', 'heat_max', 'heat_min'
        ]
        for col in optional_index_cols:
            if col in time_series_index.columns:
                index_cols.append(col)
        df = df.merge(
            time_series_index[index_cols].add_prefix('index_'),
            left_on='timestamp', right_on='index_timestamp', how='left'
        ).drop(columns=['index_timestamp'])

    # Step 7: merge index OHLC metrics (prefix: index_)
    if hour_ohlc_index_metrics is not None:
        df = df.merge(
            hour_ohlc_index_metrics.add_prefix('index_'),
            left_on='timestamp', right_on='index_timestamp', how='left'
        ).drop(columns=['index_timestamp'])

    # Step 8: merge index extremum-related features (prefix: index_)
    if index_min_max_to_ohlc is not None:
        df = df.merge(
            index_min_max_to_ohlc.add_prefix('index_'),
            left_on='timestamp', right_on='index_timestamp', how='left'
        ).drop(columns=['index_timestamp'])

    return df
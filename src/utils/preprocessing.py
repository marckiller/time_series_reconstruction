import pandas as pd
import numpy as np
import random
import string

def load_prn_file(instrument: str, folder: str = "data/raw") -> pd.DataFrame:
    """
    Load a .prn file for a given instrument and return a DataFrame with timestamp and OHLCV.

    Parameters:
        instrument (str): Instrument name, e.g., 'ALIOR'.
        folder (str): Path to the folder containing the .prn file.

    Returns:
        pd.DataFrame: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    file_path = f"{folder}/{instrument}.prn"

    df = pd.read_csv(
        file_path,
        header=None,
        names=["ticker", "col1", "date", "time", "open", "high", "low", "close", "volume", "col2"],
        dtype={"ticker": str, "date": str, "time": str}
    )

    df["timestamp"] = pd.to_datetime(df["date"] + df["time"], format="%Y%m%d%H%M%S")
    return df[["timestamp", "open", "high", "low", "close"]]

def fill_missing_intervals(
    df: pd.DataFrame,
    start_hour: str = "09:00:00",
    end_hour: str = "17:00:00",
    interval: str = "1min"
) -> pd.DataFrame:
    """
    Uzupełnia DataFrame OHLC brakującymi interwałami (np. minutami, sekundami) z NaN
    od start_hour do end_hour dla każdego dnia.

    Parameters:
        df (pd.DataFrame): DataFrame z kolumną 'timestamp'.
        start_hour (str): Początek dnia handlowego, np. "09:00:00".
        end_hour (str): Koniec dnia handlowego, np. "17:00:00".
        interval (str): Interwał czasowy (np. '1min', '5min', '1s').

    Returns:
        pd.DataFrame: DataFrame z kompletnymi interwałami i NaN w brakujących miejscach.
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    all_days = df.index.normalize().unique()

    full_index = []
    for day in all_days:
        start = pd.Timestamp(f"{day.date()} {start_hour}")
        end = pd.Timestamp(f"{day.date()} {end_hour}")
        full_index.extend(pd.date_range(start, end, freq=interval))

    full_index = pd.DatetimeIndex(full_index)
    df = df.reindex(full_index)

    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timestamp'}, inplace=True)

    return df

def aggregate_ohlc(df: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:
    """
    Agreguje tickowe dane do wybranego interwału czasowego (np. '1min', '5min', '1h').

    Parameters:
        df (pd.DataFrame): DataFrame z kolumną 'timestamp' oraz 'open', 'high', 'low', 'close', opcjonalnie 'volume'.
        interval (str): Interwał czasowy agregacji w formacie pandas, np. '1min', '5min', '1h'.

    Returns:
        pd.DataFrame: DataFrame zagregowany do wskazanego interwału OHLC(V).
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df.columns:
        ohlc_dict['volume'] = 'sum'

    df_agg = df.resample(interval).agg(ohlc_dict).dropna(how='all')

    df_agg.reset_index(inplace=True)
    return df_agg

def build_ts_dataframe(hour_df: pd.DataFrame, minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tworzy DataFrame z kolumnami:
    - 'timestamp' z hour_df
    - 'ts': lista cen 'close' z każdej minuty w danym okresie
    - 'ts_first_minute_open': cena open z pierwszej minuty (lub NaN)
    - 'mask_ts': lista binarna (1 - wartość istnieje, 0 - NaN)
    - 'ts_nan_count': liczba NaNów w szeregu
    - 'ts_length': długość szeregu
    - 'mask_ts_first_minute_open': 1 jeśli first_minute_open istnieje, inaczej 0
    - 'ts_min_max_norm': normalizacja szeregu do [0,1] względem jego własnego min/max
    - 'ts_low_high_norm': normalizacja względem low/high z hour_df (jeśli możliwe)

    Parameters:
        hour_df (pd.DataFrame): Dane godzinowe z kolumnami 'timestamp', 'low', 'high'
        minute_df (pd.DataFrame): Dane minutowe z kolumnami 'timestamp', 'open', 'close'

    Returns:
        pd.DataFrame
    """
    hour_df = hour_df.copy()
    minute_df = minute_df.copy()

    hour_df['timestamp'] = pd.to_datetime(hour_df['timestamp'])
    minute_df['timestamp'] = pd.to_datetime(minute_df['timestamp'])
    minute_df.set_index('timestamp', inplace=True)

    result = []

    for row in hour_df.itertuples(index=False):
        ts_min, ts_max = None, None
        start = row.timestamp
        end = start + pd.Timedelta(hours=1)
        minute_range = pd.date_range(start, end - pd.Timedelta(minutes=1), freq='1min')

        closes = minute_df.reindex(minute_range)['close']
        closes_list = closes.tolist()
        mask = closes.notna().astype(int).tolist()

        try: #try to get the first open price from the minute_df
            first_open = minute_df.at[start, 'open']
        except KeyError:
            first_open = float('nan')

        # ts_min_max_norm
        try:
            ts_min = min(x for x in closes_list if pd.notna(x))
            ts_max = max(x for x in closes_list if pd.notna(x))
            if ts_max != ts_min:
                ts_min_max_norm = [(x - ts_min) / (ts_max - ts_min) if pd.notna(x) else None for x in closes_list]
            else:
                ts_min_max_norm = [0.0 if pd.notna(x) else None for x in closes_list]  # stały szereg
        except ValueError:
            ts_min_max_norm = [None for _ in closes_list]

        # ts_low_high_norm
        try:
            low = row.low
            high = row.high
            if pd.notna(low) and pd.notna(high) and high != low:
                ts_low_high_norm = [(x - low) / (high - low) if pd.notna(x) else None for x in closes_list]
            else:
                ts_low_high_norm = [None for _ in closes_list]
        except AttributeError:
            ts_low_high_norm = [None for _ in closes_list]

        result.append({
            'timestamp': start,
            'ts': closes_list,
            'ts_min_max_norm': ts_min_max_norm,
            'ts_low_high_norm': ts_low_high_norm,
            'ts_first_minute_open': first_open,
            'mask_ts': mask,
            'mask_ts_first_minute_open': int(pd.notna(first_open)),
            'ts_nan_count': mask.count(0),
            'ts_length': len(mask),
            'ts_min': ts_min,
            'ts_max': ts_max
        })

    return pd.DataFrame(result)

def project_observations_to_minute_grid(
    observations: pd.DataFrame,
    interval_start: pd.Timestamp | str,
    value_column: str = "close",
    timestamp_column: str = "timestamp",
    seq_len: int = 60,
    duplicate_policy: str = "last"
) -> list[float]:
    """
    Project timestamped intrahour observations onto a fixed minute grid.

    Missing slots are returned as NaN. Observations outside
    [interval_start, interval_start + seq_len minutes) are ignored.

    Parameters:
        observations (pd.DataFrame): input observations with timestamp and value columns
        interval_start (pd.Timestamp | str): start of the reconstructed interval
        value_column (str): column containing observed values
        timestamp_column (str): column containing timestamps
        seq_len (int): number of minute slots in the output grid
        duplicate_policy (str): how to handle multiple observations in one slot;
            one of "first", "last", "mean"

    Returns:
        list[float]: length-seq_len sparse vector with observed values and NaN gaps
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if duplicate_policy not in {"first", "last", "mean"}:
        raise ValueError("duplicate_policy must be one of: first, last, mean")
    if timestamp_column not in observations.columns:
        raise KeyError(f"Column '{timestamp_column}' not found in observations.")
    if value_column not in observations.columns:
        raise KeyError(f"Column '{value_column}' not found in observations.")

    start = pd.to_datetime(interval_start)
    end = start + pd.Timedelta(minutes=seq_len)

    df = observations[[timestamp_column, value_column]].copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df[(df[timestamp_column] >= start) & (df[timestamp_column] < end)]

    grid = [float("nan")] * seq_len
    if df.empty:
        return grid

    offsets = ((df[timestamp_column] - start).dt.total_seconds() // 60).astype(int)
    df = df.assign(_slot=offsets)

    if duplicate_policy == "first":
        values = df.groupby("_slot", sort=False)[value_column].first()
    elif duplicate_policy == "last":
        values = df.groupby("_slot", sort=False)[value_column].last()
    else:
        values = df.groupby("_slot", sort=False)[value_column].mean()

    for slot, value in values.items():
        if 0 <= slot < seq_len and pd.notna(value):
            grid[int(slot)] = float(value)

    return grid

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

def align_on_common_timestamps(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zwraca dwa DataFrame'y zawierające tylko wspólne timestampy.

    Parameters:
        df1 (pd.DataFrame): Pierwszy DataFrame z kolumną 'timestamp'.
        df2 (pd.DataFrame): Drugi DataFrame z kolumną 'timestamp'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Oba DataFrame'y z zachowanymi tylko wspólnymi timestampami.
    """
    df1 = df1.copy()
    df2 = df2.copy()

    df1['timestamp'] = pd.to_datetime(df1['timestamp'])
    df2['timestamp'] = pd.to_datetime(df2['timestamp'])

    common = set(df1['timestamp']) & set(df2['timestamp'])

    df1_common = df1[df1['timestamp'].isin(common)].sort_values('timestamp').reset_index(drop=True)
    df2_common = df2[df2['timestamp'].isin(common)].sort_values('timestamp').reset_index(drop=True)

    return df1_common, df2_common

def add_rolling_correlations(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    column1: str = "ret_log",
    column2: str = "ret_log",
    windows: list[int] = [10, 20, 30, 60]
) -> pd.DataFrame:
    """
    Dodaje do df1 kolumny z korelacjami zwrotów logarytmicznych z df2
    dla różnych okien czasowych.

    Zakłada, że oba DataFrame'y mają kolumnę 'timestamp' i są dopasowane względem niej.

    Parameters:
        df1 (pd.DataFrame): pierwszy DataFrame (otrzyma kolumny z korelacją)
        df2 (pd.DataFrame): drugi DataFrame (porównywany)
        column1 (str): kolumna w df1 do użycia (domyślnie "ret_log")
        column2 (str): kolumna w df2 do użycia (domyślnie "ret_log")
        windows (list[int]): lista długości okien do korelacji

    Returns:
        pd.DataFrame: df1 z dodanymi kolumnami 'corr_{window}'
    """
    df1 = df1.copy()
    df2 = df2.copy()

    df1, df2 = align_on_common_timestamps(df1, df2)

    for w in windows:
        df1[f"corr_{w}"] = df1[column1].rolling(window=w).corr(df2[column2])

    return df1

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

def build_summary_dataframe(
    hour_instrument: pd.DataFrame,
    ts_instrument: pd.DataFrame,
    hour_instrument_metrics: pd.DataFrame,
    hour_index: pd.DataFrame,
    ts_index: pd.DataFrame,
    hour_index_metrics: pd.DataFrame,
    ticker: str | None = None
    ) -> pd.DataFrame:
    """
    Create summary dataframe merging instrument and index features.
    Assumes instrument has correlation columns, index does not.

    Parameters:
        ticker (str | None): Optional ticker name. If None, a random name is generated.

    Returns:
        pd.DataFrame: Merged summary dataframe.
    """
    df = hour_instrument.copy()
    df['ticker'] = ticker if ticker is not None else ''.join(random.choices(string.ascii_uppercase, k=5))

    # Instrument time series
    ts_cols = [
        'timestamp', 'ts', 'ts_min_max_norm', 'ts_low_high_norm',
        'ts_first_minute_open', 'mask_ts', 'mask_ts_first_minute_open',
        'ts_nan_count', 'ts_length', 'ts_min', 'ts_max'
    ]
    df = df.merge(ts_instrument[ts_cols], on='timestamp', how='left')

    # Instrument metrics
    df = df.merge(hour_instrument_metrics, on='timestamp', how='left')

    # Index OHLC
    df = df.merge(
        hour_index.add_prefix('index_'),
        left_on='timestamp', right_on='index_timestamp', how='left'
    ).drop(columns=['index_timestamp'])

    # Index time series
    index_ts_cols = [
        'timestamp', 'ts', 'ts_min_max_norm', 'ts_low_high_norm',
        'ts_first_minute_open', 'mask_ts', 'mask_ts_first_minute_open',
        'ts_nan_count', 'ts_length', 'ts_min', 'ts_max'
    ]
    df = df.merge(
        ts_index[index_ts_cols].add_prefix('index_'),
        left_on='timestamp', right_on='index_timestamp', how='left'
    ).drop(columns=['index_timestamp'])

    # Index metrics
    df = df.merge(
        hour_index_metrics.add_prefix('index_'),
        left_on='timestamp', right_on='index_timestamp', how='left'
    ).drop(columns=['index_timestamp'])

    return df

def append_to_master_dataframe(master_df: pd.DataFrame | None, df: pd.DataFrame) -> pd.DataFrame:
    """
    Append a single dataframe `df` to the accumulated `master_df`.
    If `master_df` is None or empty, return a copy of `df`.
    """
    if master_df is None or master_df.empty:
        return df.copy()
    return pd.concat([master_df, df], ignore_index=True)

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

    # Clip to avoid extreme log returns that cause price explosion
    max_abs_return = 0.1  # lub np. 0.05
    index_returns = np.clip(index_returns, -max_abs_return, max_abs_return)
    instr_returns = np.clip(instr_returns, -max_abs_return, max_abs_return)

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
        cum_log_returns = np.clip(cum_log_returns, -10, 10)
        prices = initial_price * np.exp(cum_log_returns)
    else:
        cum_growth_factors = np.cumprod(1 + returns)
        prices = initial_price * cum_growth_factors
    return prices

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
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
        cum_log_returns = np.clip(cum_log_returns, -10, 10)  # możesz dobrać limit
        prices = initial_price * np.exp(cum_log_returns)
    else:
        cum_growth_factors = np.cumprod(1 + returns)
        prices = initial_price * cum_growth_factors
    return prices
"""Shared validation for strategy market-data inputs."""

import numpy as np
import pandas as pd


def _validated_close(market_data: pd.DataFrame) -> pd.Series:
    if not isinstance(market_data, pd.DataFrame):
        raise ValueError("market_data must be a pandas DataFrame")
    if not isinstance(market_data.index, pd.DatetimeIndex):
        raise ValueError("market_data must use a DatetimeIndex")
    if market_data.index.hasnans:
        raise ValueError("market_data dates must be valid and must not be missing")
    if not market_data.index.is_monotonic_increasing:
        raise ValueError("market_data dates must be sorted in ascending order")
    if not market_data.index.is_unique:
        raise ValueError("market_data dates must be unique")
    if list(market_data.columns).count("Close") != 1:
        raise ValueError("market_data must contain a 'Close' column")

    close = pd.to_numeric(market_data["Close"], errors="coerce").astype("float64")
    if close.empty:
        raise ValueError("market_data must contain at least one observation")
    if close.isna().any():
        raise ValueError("Close prices must contain no missing or non-numeric values")
    if not np.isfinite(close.to_numpy()).all():
        raise ValueError("Close prices must contain only finite values")
    if (close <= 0).any():
        raise ValueError("Close prices must be strictly positive")
    return close

"""Validated return and drawdown calculations."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _series(values: Sequence[float] | pd.Series, name: str) -> pd.Series:
    """Convert observations to a numeric series and reject infinite values."""
    result = pd.Series(values, dtype="float64", copy=True)
    if result.empty:
        raise ValueError(f"{name} must contain at least one observation")
    if np.isinf(result).any():
        raise ValueError(f"{name} must not contain infinite values")
    return result


def _prices(values: Sequence[float] | pd.Series) -> pd.Series:
    """Validate finite, strictly positive prices for price-based metrics."""
    result = _series(values, "prices")
    if result.isna().any():
        raise ValueError("prices must not contain NaN values")
    if (result <= 0).any():
        raise ValueError("prices must be strictly positive")
    return result


def simple_returns(prices: Sequence[float] | pd.Series) -> pd.Series:
    """Calculate percentage price changes, with an explicit initial NaN."""
    return _prices(prices).pct_change(fill_method=None)


def log_returns(prices: Sequence[float] | pd.Series) -> pd.Series:
    """Calculate continuously compounded returns from strictly positive prices."""
    validated = _prices(prices)
    return np.log(validated / validated.shift(1))


def rolling_mean(
    values: Sequence[float] | pd.Series, window: int
) -> pd.Series:
    """Calculate a right-aligned rolling arithmetic mean."""
    if window < 1:
        raise ValueError("window must be at least 1")
    return _series(values, "values").rolling(window=window, min_periods=window).mean()


def rolling_volatility(
    values: Sequence[float] | pd.Series, window: int, ddof: int = 1
) -> pd.Series:
    """Calculate a right-aligned rolling sample standard deviation."""
    if window < 1:
        raise ValueError("window must be at least 1")
    if ddof < 0 or ddof >= window:
        raise ValueError("ddof must be non-negative and smaller than window")
    return _series(values, "values").rolling(
        window=window, min_periods=window
    ).std(ddof=ddof)


def maximum_drawdown(prices: Sequence[float] | pd.Series) -> float:
    """Return the most negative peak-to-trough percentage decline."""
    validated = _prices(prices)
    drawdowns = validated / validated.cummax() - 1.0
    return float(drawdowns.min())

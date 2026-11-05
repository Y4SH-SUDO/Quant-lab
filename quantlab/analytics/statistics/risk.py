"""Return-based risk statistics with explicit annualization conventions."""

from collections.abc import Sequence

import pandas as pd

from quantlab.analytics.statistics._utils import as_series, validate_window


def historical_volatility(
    returns: Sequence[float] | pd.Series,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> float:
    """Return annualized volatility from periodic returns."""
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    return float(
        as_series(returns, minimum=ddof + 1).std(ddof=ddof) * periods_per_year**0.5
    )


def rolling_volatility_annualized(
    returns: Sequence[float] | pd.Series,
    window: int,
    *,
    periods_per_year: int = 252,
    ddof: int = 1,
) -> pd.Series:
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    validate_window(window, ddof + 1)
    observations = as_series(returns, dropna=False)
    return (
        observations.rolling(window, min_periods=window).std(ddof=ddof)
        * periods_per_year**0.5
    )


rolling_volatility = rolling_volatility_annualized


def return_dispersion(returns: Sequence[float] | pd.Series, *, ddof: int = 1) -> float:
    """Return unannualized cross-sectional/time-series standard deviation."""
    return float(as_series(returns, minimum=ddof + 1).std(ddof=ddof))

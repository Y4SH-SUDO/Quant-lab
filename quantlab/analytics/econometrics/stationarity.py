"""Stationarity tests and transformations."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from quantlab.analytics.statistics._utils import as_series


@dataclass(frozen=True)
class ADFResult:
    statistic: float
    p_value: float
    used_lag: int
    observations: int
    critical_values: dict[str, float]
    regression: str
    stationary_at: float | None


def adf_test(
    values: Sequence[float] | pd.Series,
    *,
    regression: str = "c",
    max_lag: int | None = None,
    autolag: str | None = "AIC",
    alpha: float = 0.05,
) -> ADFResult:
    """Run ADF; rejection indicates evidence against a unit root only."""
    if regression not in {"c", "ct", "ctt", "n"}:
        raise ValueError("regression must be one of 'c', 'ct', 'ctt', or 'n'")
    if max_lag is not None and (not isinstance(max_lag, int) or max_lag < 0):
        raise ValueError("max_lag must be non-negative or None")
    if autolag not in {None, "AIC", "BIC", "t-stat"}:
        raise ValueError("autolag must be None, 'AIC', 'BIC', or 't-stat'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    observations = as_series(values, minimum=8)
    try:
        result = adfuller(
            observations, maxlag=max_lag, regression=regression, autolag=autolag
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"ADF test could not be fitted: {error}") from error
    critical = {str(key): float(value) for key, value in result[4].items()}
    level = alpha
    return ADFResult(
        float(result[0]),
        float(result[1]),
        int(result[2]),
        int(result[3]),
        critical,
        regression,
        level if result[1] < alpha else None,
    )


def difference(values: Sequence[float] | pd.Series, order: int = 1) -> pd.Series:
    if not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    result = pd.Series(values, dtype="float64", copy=True)
    if result.empty:
        raise ValueError("values must contain observations")
    for _ in range(order):
        result = result.diff().dropna()
    return result

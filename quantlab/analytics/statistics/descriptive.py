"""Descriptive statistics for finite numeric observations."""

from collections.abc import Sequence

import pandas as pd
from scipy import stats

from quantlab.analytics.statistics._utils import as_series


def mean(values: Sequence[float] | pd.Series) -> float:
    return float(as_series(values).mean())


def median(values: Sequence[float] | pd.Series) -> float:
    return float(as_series(values).median())


def minimum(values: Sequence[float] | pd.Series) -> float:
    return float(as_series(values).min())


def maximum(values: Sequence[float] | pd.Series) -> float:
    return float(as_series(values).max())


def data_range(values: Sequence[float] | pd.Series) -> float:
    observations = as_series(values)
    return float(observations.max() - observations.min())


def variance(values: Sequence[float] | pd.Series, *, ddof: int = 1) -> float:
    observations = as_series(values, minimum=ddof + 1)
    if ddof < 0 or ddof >= len(observations):
        raise ValueError("ddof must be non-negative and smaller than observations")
    return float(observations.var(ddof=ddof))


def standard_deviation(values: Sequence[float] | pd.Series, *, ddof: int = 1) -> float:
    return variance(values, ddof=ddof) ** 0.5


def skewness(values: Sequence[float] | pd.Series, *, bias: bool = False) -> float:
    return float(stats.skew(as_series(values, minimum=3), bias=bias))


def kurtosis(
    values: Sequence[float] | pd.Series,
    *,
    fisher: bool = True,
    bias: bool = False,
) -> float:
    """Return excess kurtosis by default; set ``fisher=False`` for Pearson."""
    return float(stats.kurtosis(as_series(values, minimum=4), fisher=fisher, bias=bias))


def describe(values: Sequence[float] | pd.Series) -> pd.Series:
    """Return the common descriptive statistics in one deterministic object."""
    observations = as_series(values, minimum=4)
    return pd.Series(
        {
            "mean": mean(observations),
            "median": median(observations),
            "minimum": minimum(observations),
            "maximum": maximum(observations),
            "range": data_range(observations),
            "variance_sample": variance(observations),
            "variance_population": variance(observations, ddof=0),
            "standard_deviation_sample": standard_deviation(observations),
            "standard_deviation_population": standard_deviation(observations, ddof=0),
            "skewness": skewness(observations),
            "kurtosis_excess": kurtosis(observations),
        }
    )

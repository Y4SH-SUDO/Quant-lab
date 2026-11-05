"""Covariance and correlation for aligned instrument return series."""

from collections.abc import Mapping, Sequence

import pandas as pd

from quantlab.analytics.statistics._utils import aligned_pair


def covariance(
    first: Sequence[float] | pd.Series,
    second: Sequence[float] | pd.Series,
    *,
    ddof: int = 1,
) -> float:
    left, right = aligned_pair(first, second, minimum=ddof + 1)
    return float(left.cov(right, ddof=ddof))


def correlation(
    first: Sequence[float] | pd.Series, second: Sequence[float] | pd.Series
) -> float:
    left, right = aligned_pair(first, second)
    value = left.corr(right)
    if pd.isna(value):
        raise ValueError("correlation is undefined for a constant series")
    return float(value)


def correlation_matrix(
    instruments: Mapping[str, Sequence[float] | pd.Series] | pd.DataFrame,
) -> pd.DataFrame:
    frame = (
        instruments.copy()
        if isinstance(instruments, pd.DataFrame)
        else pd.DataFrame(instruments)
    )
    if frame.shape[1] < 2:
        raise ValueError("at least two instruments are required")
    numeric = frame.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    if numeric.empty or not numeric.apply(lambda col: col.dropna().size >= 2).all():
        raise ValueError("each instrument must have at least two finite observations")
    if (
        not numeric.map(lambda value: pd.isna(value) or pd.api.types.is_number(value))
        .all()
        .all()
    ):
        raise ValueError("instrument data must be numeric")
    result = numeric.corr()
    if result.isna().any().any():
        raise ValueError("correlation is undefined for a constant instrument")
    return result


def rolling_correlation(
    first: Sequence[float] | pd.Series,
    second: Sequence[float] | pd.Series,
    window: int,
) -> pd.Series:
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2")
    left, right = aligned_pair(first, second, minimum=window)
    return left.rolling(window).corr(right)

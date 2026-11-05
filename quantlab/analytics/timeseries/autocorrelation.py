"""Autocorrelation and partial-autocorrelation diagnostics."""

from collections.abc import Sequence

import pandas as pd
from statsmodels.tsa.stattools import acf, pacf

from quantlab.analytics.statistics._utils import as_series


def autocorrelation(
    values: Sequence[float] | pd.Series,
    lags: int,
    *,
    alpha: float | None = 0.05,
) -> pd.DataFrame:
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 or None")
    result = acf(
        as_series(values, minimum=lags + 2), nlags=lags, alpha=alpha, fft=False
    )
    if alpha is None:
        return pd.DataFrame({"acf": result})
    values_, confidence = result
    return pd.DataFrame(
        {"acf": values_, "lower": confidence[:, 0], "upper": confidence[:, 1]}
    )


def partial_autocorrelation(
    values: Sequence[float] | pd.Series,
    lags: int,
    *,
    alpha: float | None = 0.05,
) -> pd.DataFrame:
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1 or None")
    result = pacf(as_series(values, minimum=lags + 2), nlags=lags, alpha=alpha)
    if alpha is None:
        return pd.DataFrame({"pacf": result})
    values_, confidence = result
    return pd.DataFrame(
        {"pacf": values_, "lower": confidence[:, 0], "upper": confidence[:, 1]}
    )

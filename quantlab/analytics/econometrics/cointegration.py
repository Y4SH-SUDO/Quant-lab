"""Engle-Granger cointegration testing and residual analysis."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from quantlab.analytics.econometrics.regression import OLSResult, ols_regression


@dataclass(frozen=True)
class CointegrationResult:
    statistic: float
    p_value: float
    critical_values: tuple[float, float, float]
    regression: OLSResult
    residuals: pd.Series
    alpha: float
    cointegrated_at: float | None


def engle_granger_test(
    first: Sequence[float] | pd.Series,
    second: Sequence[float] | pd.Series,
    *,
    trend: str = "c",
    max_lag: int | None = None,
    autolag: str | None = "aic",
    alpha: float = 0.05,
) -> CointegrationResult:
    """Test whether two aligned levels share a long-run equilibrium."""
    if trend not in {"c", "ct", "ctt", "n"}:
        raise ValueError("trend must be one of 'c', 'ct', 'ctt', or 'n'")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    regression = ols_regression(first, second)
    aligned = pd.concat(
        [pd.Series(first, dtype="float64"), pd.Series(second, dtype="float64")],
        axis=1,
    ).dropna()
    try:
        result = coint(
            aligned.iloc[:, 0],
            aligned.iloc[:, 1],
            trend=trend,
            maxlag=max_lag,
            autolag=autolag,
        )
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"cointegration test could not be fitted: {error}") from error
    p_value = float(result[1])
    critical_values = tuple(float(value) for value in result[2])
    return CointegrationResult(
        float(result[0]),
        p_value,
        (critical_values[0], critical_values[1], critical_values[2]),
        regression,
        regression.residuals,
        alpha,
        alpha if p_value < alpha else None,
    )

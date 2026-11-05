"""ARIMA modelling and forecasting wrappers."""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from quantlab.analytics.statistics._utils import as_series


@dataclass(frozen=True)
class ARIMAResult:
    model: object
    forecast: pd.Series
    residuals: pd.Series
    aic: float
    bic: float
    converged: bool
    warnings: tuple[str, ...]

    @property
    def fit_succeeded(self) -> bool:
        return self.converged


def fit_arima(
    values: Sequence[float] | pd.Series,
    order: tuple[int, int, int] = (1, 0, 0),
    *,
    forecast_steps: int = 0,
) -> ARIMAResult:
    if len(order) != 3 or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in order
    ):
        raise ValueError("order must contain three non-negative integers")
    if (
        not isinstance(forecast_steps, int)
        or isinstance(forecast_steps, bool)
        or forecast_steps < 0
    ):
        raise ValueError("forecast_steps must be a non-negative integer")
    observations = as_series(values, minimum=max(8, order[0] + order[2] + order[1] + 2))
    # ARIMA forecasts advance by observed session, not calendar day. Use integer
    # positions for irregular exchange dates and normalize for stable optimization.
    observations = pd.Series(observations.to_numpy(), dtype="float64")
    standard_deviation = float(observations.std())
    scale_factor = 1.0 / standard_deviation if standard_deviation > 0 else 1.0
    scaled_observations = observations * scale_factor
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fitted = ARIMA(scaled_observations, order=order).fit()
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"ARIMA model could not be fitted: {error}") from error
    for warning in caught:
        warnings.warn(str(warning.message), warning.category, stacklevel=2)
    convergence = getattr(fitted, "mle_retvals", {}).get("converged")
    converged = bool(convergence) if convergence is not None else True
    forecast = (
        fitted.forecast(forecast_steps) / scale_factor
        if forecast_steps
        else pd.Series(dtype="float64")
    )
    if forecast_steps and (
        len(forecast) != forecast_steps
        or not np.isfinite(np.asarray(forecast, dtype="float64")).all()
    ):
        raise ValueError("ARIMA forecast contains invalid numerical values")
    return ARIMAResult(
        fitted,
        forecast,
        fitted.resid / scale_factor,
        float(fitted.aic - 2 * len(observations) * np.log(scale_factor)),
        float(fitted.bic - 2 * len(observations) * np.log(scale_factor)),
        converged,
        tuple(str(warning.message) for warning in caught),
    )

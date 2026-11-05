"""GARCH volatility modelling."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from arch import arch_model

from quantlab.analytics.statistics._utils import as_series


@dataclass(frozen=True)
class GARCHResult:
    model: object
    conditional_volatility: pd.Series
    residuals: pd.Series
    aic: float
    bic: float
    forecast_variance: pd.Series
    forecast_volatility: pd.Series

    @property
    def conditional_variance_forecast(self) -> pd.Series:
        return self.forecast_variance

    @property
    def conditional_volatility_forecast(self) -> pd.Series:
        return self.forecast_volatility


def fit_garch(
    returns: Sequence[float] | pd.Series,
    *,
    p: int = 1,
    q: int = 1,
    mean: str = "constant",
    dist: str = "normal",
    forecast_steps: int = 0,
) -> GARCHResult:
    if not isinstance(p, int) or isinstance(p, bool) or p < 1:
        raise ValueError("p and q must be positive")
    if not isinstance(q, int) or isinstance(q, bool) or q < 1:
        raise ValueError("p and q must be positive")
    if mean not in {"constant", "zero", "AR"}:
        raise ValueError("mean must be 'constant', 'zero', or 'AR'")
    if (
        not isinstance(forecast_steps, int)
        or isinstance(forecast_steps, bool)
        or forecast_steps < 0
    ):
        raise ValueError("forecast_steps must be a non-negative integer")
    observations = as_series(returns, minimum=20) * 100
    try:
        fitted = arch_model(
            observations,
            mean=cast(Literal["constant", "zero", "AR"], mean),
            vol="GARCH",
            p=p,
            q=q,
            dist=cast(Literal["normal"], dist),
        ).fit(disp="off")
    except (ValueError, np.linalg.LinAlgError) as error:
        raise ValueError(f"GARCH model could not be fitted: {error}") from error
    if forecast_steps:
        try:
            forecast_frame = fitted.forecast(horizon=forecast_steps).variance.iloc[-1]
        except (ValueError, np.linalg.LinAlgError) as error:
            raise ValueError(
                f"GARCH forecast could not be calculated: {error}"
            ) from error
        forecast_variance = pd.Series(
            np.asarray(forecast_frame, dtype="float64") / 10000,
            index=pd.RangeIndex(1, forecast_steps + 1, name="horizon"),
            name="forecast_variance",
        )
        forecast_volatility = forecast_variance.pow(0.5).rename("forecast_volatility")
        if not (
            np.isfinite(forecast_variance.to_numpy()).all()
            and (forecast_variance >= 0).all()
            and np.isfinite(forecast_volatility.to_numpy()).all()
        ):
            raise ValueError("GARCH forecast contains invalid numerical values")
    else:
        forecast_variance = pd.Series(dtype="float64", name="forecast_variance")
        forecast_volatility = pd.Series(dtype="float64", name="forecast_volatility")
    return GARCHResult(
        fitted,
        fitted.conditional_volatility / 100,
        fitted.resid / 100,
        float(fitted.aic),
        float(fitted.bic),
        forecast_variance,
        forecast_volatility,
    )

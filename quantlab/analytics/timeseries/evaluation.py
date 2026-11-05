from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from quantlab.analytics.timeseries.arima import fit_arima
from quantlab.analytics.volatility.garch import fit_garch


def chronological_split(
    values: pd.Series, train_fraction: float = 0.8
) -> tuple[pd.Series, pd.Series]:
    """Split observed sessions chronologically without shuffling."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if not isinstance(values, pd.Series) or len(values) < 40:
        raise ValueError("values must be a Series with at least 40 observations")
    if not values.index.is_monotonic_increasing or values.index.has_duplicates:
        raise ValueError("values index must be sorted and unique")
    if not np.isfinite(values.to_numpy(dtype="float64")).all():
        raise ValueError("values must be finite")
    split = int(len(values) * train_fraction)
    if split < 20 or split >= len(values):
        raise ValueError("split must leave at least 20 train and 1 test observation")
    return values.iloc[:split].copy(), values.iloc[split:].copy()


def _mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def _rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def _qlike(proxy: np.ndarray, variance: np.ndarray) -> float:
    if (proxy < 0).any() or (variance <= 0).any():
        raise ValueError("QLIKE requires nonnegative proxy and positive variance")
    return float(
        np.mean(np.log(variance) + np.maximum(proxy, np.finfo(float).tiny) / variance)
    )


def evaluate_returns(returns: pd.Series) -> dict[str, Any]:
    """Evaluate fixed-origin multi-step mean and variance forecasts."""
    train, test = chronological_split(returns)
    actual = test.to_numpy(dtype="float64")
    arima = fit_arima(train, (1, 0, 1), forecast_steps=len(test))
    if not arima.converged:
        raise ValueError("ARIMA holdout fit did not converge")
    mean_forecast = arima.forecast.to_numpy(dtype="float64")
    garch = fit_garch(train, p=1, q=1, forecast_steps=len(test))
    variance = garch.forecast_variance.to_numpy(dtype="float64")
    zero = np.zeros(len(test))
    train_mean = np.full(len(test), train.mean())
    proxy = actual**2
    hist_variance = np.full(len(test), np.mean((train.to_numpy() - train.mean()) ** 2))
    lag = min(10, max(1, len(test) // 5))
    lb = acorr_ljungbox(actual - mean_forecast, lags=[lag], return_df=True)
    lb_sq = acorr_ljungbox(
        (actual - mean_forecast) ** 2 / variance, lags=[lag], return_df=True
    )
    return {
        "n_observations": len(returns),
        "n_train": len(train),
        "n_test": len(test),
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "test_start": str(test.index[0].date()),
        "test_end": str(test.index[-1].date()),
        "target": "one-session simple return from Yahoo Adj Close; decimal units",
        "split": (
            "chronological 80/20; fit at train end; fixed-origin "
            "multi-step test forecasts"
        ),
        "arima": {
            "order": [1, 0, 1],
            "converged": arima.converged,
            "benchmarks": {
                "zero": {"mae": _mae(actual, zero), "rmse": _rmse(actual, zero)},
                "training_mean": {
                    "mae": _mae(actual, train_mean),
                    "rmse": _rmse(actual, train_mean),
                },
            },
            "model": {
                "mae": _mae(actual, mean_forecast),
                "rmse": _rmse(actual, mean_forecast),
                "directional_accuracy": float(
                    np.mean((actual > 0) == (mean_forecast > 0))
                ),
            },
            "residual_diagnostics": {
                "ljung_box_lag": lag,
                "residual_autocorrelation_p": float(lb.lb_pvalue.iloc[0]),
            },
        },
        "garch": {
            "specification": "GARCH(1,1), constant mean, normal innovations",
            "target_proxy": (
                "squared one-session return; noisy proxy for latent variance"
            ),
            "variance_units": "decimal return squared per session; no annualization",
            "benchmarks": {
                "training_historical_variance": {
                    "rmse_squared_return": _rmse(proxy, hist_variance),
                    "qlike": _qlike(proxy, hist_variance),
                }
            },
            "model": {
                "rmse_squared_return": _rmse(proxy, variance),
                "qlike": _qlike(proxy, variance),
            },
            "standardized_error_diagnostics": {
                "squared_standardized_error_autocorrelation_p": float(
                    lb_sq.lb_pvalue.iloc[0]
                )
            },
        },
        "limitations": [
            "single holdout and fixed-origin forecasts; no rolling refit",
            "squared returns are a noisy variance proxy",
            "no significance tests or parameter uncertainty",
            "adjusted data completeness is unaudited",
            "results do not establish trading value",
        ],
    }

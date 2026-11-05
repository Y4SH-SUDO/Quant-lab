"""Deterministic coverage for M2 quantitative analytics."""

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.econometrics import (
    adf_test,
    difference,
    engle_granger_test,
    ols_regression,
)
from quantlab.analytics.statistics import (
    bonferroni_correction,
    correlation,
    correlation_matrix,
    covariance,
    describe,
    historical_volatility,
    one_sample_ttest,
    rolling_correlation,
    two_sample_ttest,
)
from quantlab.analytics.timeseries import (
    autocorrelation,
    fit_arima,
    partial_autocorrelation,
)
from quantlab.analytics.volatility import fit_garch


def test_descriptive_and_risk_statistics() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    summary = describe(values)
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["variance_sample"] == pytest.approx(5 / 3)
    assert historical_volatility(
        [0.01, 0.02, 0.03], periods_per_year=1
    ) == pytest.approx(0.01)


def test_alignment_and_missing_data() -> None:
    left = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    right = pd.Series([2.0, np.nan, 6.0], index=[1, 2, 3])
    assert correlation(left, right) == pytest.approx(1.0)
    assert correlation_matrix({"a": left, "b": right}).loc["a", "b"] == pytest.approx(
        1.0
    )
    assert rolling_correlation(left, right, 2).iloc[-1] == pytest.approx(1.0)


def test_hypothesis_stationarity_and_differencing() -> None:
    result = one_sample_ttest([1, 1, 1, 1], 0)
    assert result.reject_null
    with pytest.raises(ValueError):
        difference([1, 2], 0)
    pd.testing.assert_series_equal(
        difference([1.0, 4.0, 9.0, 16.0], 2),
        pd.Series([2.0, 2.0], index=[2, 3]),
    )
    with pytest.raises(ValueError):
        adf_test([1] * 20)
    assert two_sample_ttest([1, 2, 3], [4, 5, 6]).p_value < 0.05
    assert covariance([1, 2, 3], [2, 4, 6]) == pytest.approx(2)
    assert len(partial_autocorrelation(np.arange(20, dtype=float), 3)) == 4


def test_regression_cointegration_and_time_series_models() -> None:
    index = pd.date_range("2020-01-01", periods=80, freq="D")
    first = pd.Series(np.arange(80, dtype=float), index=index)
    second = 2 * first + 1
    regression = ols_regression(first, second)
    assert regression.slope == pytest.approx(0.5)
    assert engle_granger_test(first + 100, second + 100).p_value < 0.05
    assert len(autocorrelation(first, 3)) == 4
    arima = fit_arima(first, (1, 1, 0), forecast_steps=2)
    assert len(arima.forecast) == 2
    assert isinstance(arima.converged, bool)
    with pytest.raises(ValueError):
        fit_arima(first, (1, -1, 0))
    with pytest.raises(ValueError):
        fit_garch([0.0] * 10)


def test_arima_irregular_market_dates_use_session_steps() -> None:
    import warnings

    dates = pd.to_datetime(
        pd.bdate_range("2024-01-02", periods=120).delete([3, 17, 44, 79])
    )
    values = pd.Series(np.sin(np.arange(len(dates), dtype=float)) * 0.01, index=dates)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = fit_arima(values, (1, 0, 1), forecast_steps=2)
    assert not any("frequency information" in str(item.message) for item in caught)
    assert not any(
        "will be ignored when e.g. forecasting" in str(item.message)
        for item in caught
    )
    assert isinstance(result.model.data.row_labels, pd.RangeIndex)
    assert result.converged
    assert result.forecast.index.tolist() == [len(values), len(values) + 1]


def test_garch_fit_and_forecast_on_non_constant_returns() -> None:
    returns = pd.Series(
        0.01 * np.sin(np.arange(120, dtype=float)) + 0.002 * np.cos(np.arange(120))
    )
    result = fit_garch(returns, forecast_steps=4)
    assert len(result.conditional_volatility) == len(returns)
    assert len(result.forecast_variance) == 4
    assert len(result.forecast_volatility) == 4
    assert np.isfinite(result.forecast_variance).all()
    assert (result.forecast_variance >= 0).all()
    with pytest.raises(ValueError):
        fit_garch(returns, forecast_steps=-1)


def test_multiple_testing_preserves_unadjusted_p_values() -> None:
    result = bonferroni_correction([0.001, 0.02, 0.5])
    assert result.unadjusted_p_values.tolist() == [0.001, 0.02, 0.5]
    assert result.adjusted_p_values.tolist() == pytest.approx([0.003, 0.06, 1.0])
    assert result.reject.tolist() == [True, False, False]

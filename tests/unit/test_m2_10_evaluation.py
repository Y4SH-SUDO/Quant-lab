import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.timeseries.evaluation import (
    _mae,
    _qlike,
    _rmse,
    chronological_split,
)


def test_split_boundary_is_chronological_and_disjoint() -> None:
    values = pd.Series(np.arange(50.0), index=pd.date_range("2020-01-01", periods=50))
    train, test = chronological_split(values, 0.8)
    assert len(train) == 40 and len(test) == 10
    assert train.index[-1] < test.index[0]
    assert train.iloc[-1] == 39 and test.iloc[0] == 40


def test_forecast_metrics_and_benchmark_baseline() -> None:
    actual = np.array([1.0, -1.0])
    assert _mae(actual, np.zeros(2)) == 1
    assert _rmse(actual, np.zeros(2)) == 1
    assert _rmse(actual, actual) == 0
    assert _qlike(np.array([1.0]), np.array([1.0])) == pytest.approx(1)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.1])
def test_split_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError):
        chronological_split(pd.Series(np.arange(50.0)), fraction)


def test_split_rejects_unsorted_and_nonfinite_data() -> None:
    dates = pd.date_range("2020-01-01", periods=50)
    with pytest.raises(ValueError, match="sorted and unique"):
        chronological_split(pd.Series(np.arange(50.0), index=dates[::-1]))
    values = pd.Series(np.arange(50.0))
    values.iloc[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        chronological_split(values)


def test_qlike_rejects_zero_variance() -> None:
    with pytest.raises(ValueError, match="positive variance"):
        _qlike(np.array([1.0]), np.array([0.0]))

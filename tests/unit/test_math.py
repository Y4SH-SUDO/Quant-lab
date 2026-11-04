"""Tests for mathematical foundation utilities."""

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.statistics.returns import (
    log_returns,
    maximum_drawdown,
    rolling_mean,
    rolling_volatility,
    simple_returns,
)


def test_returns_and_drawdown() -> None:
    prices = pd.Series([100.0, 110.0, 99.0, 120.0])
    pd.testing.assert_series_equal(
        simple_returns(prices),
        pd.Series([np.nan, 0.1, -0.1, 120 / 99 - 1]),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        log_returns(prices),
        np.log(prices / prices.shift(1)),
        check_names=False,
    )
    assert maximum_drawdown(prices) == pytest.approx(-0.1)


def test_rolling_calculations() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(
        rolling_mean(values, 2),
        pd.Series([np.nan, 1.5, 2.5, 3.5]),
        check_names=False,
    )
    assert rolling_volatility(values, 2).iloc[-1] == pytest.approx(
        np.sqrt(0.5)
    )


@pytest.mark.parametrize("values", [[100.0, np.nan], [100.0, 0.0], [100.0, -1.0]])
def test_price_validation(values: list[float]) -> None:
    with pytest.raises(ValueError):
        simple_returns(values)

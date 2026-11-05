"""M1 Parquet to M2 analytics integration coverage."""

import numpy as np
import pandas as pd

from quantlab.analytics.econometrics import adf_test, ols_regression
from quantlab.analytics.statistics import correlation, describe, historical_volatility
from quantlab.analytics.statistics.returns import simple_returns
from quantlab.data.cleaning.ohlcv import normalize_ohlcv
from quantlab.data.storage.parquet import ParquetMarketStore
from quantlab.data.validation.ohlcv import validate_ohlcv


def test_stored_market_series_feed_m2_analytics(tmp_path) -> None:
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    close = 100 + np.arange(60, dtype=float) + np.sin(np.arange(60))
    frame = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close * 0.99,
            "Volume": np.full(60, 1000),
        },
        index=dates,
    )
    normalized = normalize_ohlcv(frame, "TEST")
    store = ParquetMarketStore(tmp_path)
    store.write("TEST", normalized, validate_ohlcv(normalized, "TEST"))

    stored = store.read("TEST")
    prices = stored["Close"]
    returns = simple_returns(prices).dropna()
    assert describe(prices)["mean"] > 100
    assert historical_volatility(returns, periods_per_year=252) > 0
    assert correlation(returns, returns) == 1
    adf = adf_test(returns)
    assert 0 < adf.observations <= len(returns)
    regression = ols_regression(
        prices.iloc[1:], prices.iloc[:-1].set_axis(prices.index[1:])
    )
    assert regression.observations == len(returns)

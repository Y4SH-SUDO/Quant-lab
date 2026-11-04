"""Provider interface for daily historical equity data."""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DailyHistoryProvider(ABC):
    """Fetch daily bars; implementations can be replaced with deterministic fakes."""

    price_adjustment = "unadjusted_ohlc"

    @abstractmethod
    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch bars for [start, end), returning provider-shaped OHLCV data.

        Providers must return unadjusted OHLC prices. An ``Adj Close`` column
        may contain the provider's adjusted close series for comparison.
        """

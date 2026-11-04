"""Daily market-data provider interfaces and adapters."""

from quantlab.data.providers.base import DailyHistoryProvider
from quantlab.data.providers.yfinance_provider import YFinanceDailyProvider

__all__ = ["DailyHistoryProvider", "YFinanceDailyProvider"]

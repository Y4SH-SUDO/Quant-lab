"""Yahoo Finance adapter for daily NSE history."""

from datetime import date

import pandas as pd
import yfinance as yf

from quantlab.data.providers.base import DailyHistoryProvider


class YFinanceDailyProvider(DailyHistoryProvider):
    """Fetch daily unadjusted OHLC history using Yahoo Finance's NSE suffix."""

    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        ticker = symbol.strip().upper()
        if not ticker.endswith(".NS") and not ticker.startswith("^"):
            ticker += ".NS"
        data = yf.download(
            ticker,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            group_by="column",
            multi_level_index=False,
        )
        if data is None or data.empty:
            raise ValueError(f"provider returned no daily data for {ticker}")
        return data

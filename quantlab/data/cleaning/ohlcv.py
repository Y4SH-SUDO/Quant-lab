"""Normalization and deterministic cleaning of daily OHLCV data."""

import pandas as pd

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def normalize_ohlcv(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Return canonical date-indexed OHLCV rows sorted by date.

    Duplicate dates are retained so validation can report them instead of
    silently discarding provider data.
    """
    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [str(column[0]) for column in data.columns]
    data.columns = [str(column).strip().title() for column in data.columns]
    if "Adjclose" in data.columns:
        data = data.rename(columns={"Adjclose": "Adj Close"})
    if "Date" in data.columns:
        data = data.set_index("Date")
    if data.index.name is None:
        data.index.name = "Date"
    data.index = pd.to_datetime(data.index, utc=True).tz_convert(None).normalize()
    data.index.name = "Date"
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"missing OHLCV columns: {', '.join(sorted(missing))}")
    result = data.loc[:, REQUIRED_COLUMNS].apply(pd.to_numeric, errors="coerce")
    result = result.sort_index()
    result.insert(0, "Symbol", symbol.strip().upper())
    return result

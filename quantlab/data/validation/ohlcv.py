"""OHLCV validation and dataset quality summaries."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quantlab.data.cleaning.ohlcv import REQUIRED_COLUMNS


@dataclass(frozen=True)
class DataQualityReport:
    symbol: str
    row_count: int
    start_date: str | None
    end_date: str | None
    missing_values: int
    non_finite_values: int
    duplicate_dates: int
    invalid_ohlc_rows: int
    negative_volume_rows: int
    quality_ok: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_ohlcv(frame: pd.DataFrame, symbol: str | None = None) -> DataQualityReport:
    """Report missing values, duplicate dates and impossible OHLCV rows."""
    missing_columns = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"missing OHLCV columns: {', '.join(sorted(missing_columns))}")
    values = frame.loc[:, REQUIRED_COLUMNS]
    invalid = (values["High"] < values[["Open", "Close", "Low"]].max(axis=1)) | (
        values["Low"] > values[["Open", "Close", "High"]].min(axis=1)
    )
    missing = int(values.isna().sum().sum())
    non_finite = int((~np.isfinite(values.select_dtypes(include="number"))).sum().sum())
    duplicates = int(frame.index.duplicated(keep=False).sum())
    negative_volume = int((values["Volume"] < 0).fillna(False).sum())
    non_positive_prices = (
        values[["Open", "High", "Low", "Close", "Adj Close"]] <= 0
    ).any(axis=1)
    invalid_count = int((invalid | non_positive_prices).fillna(False).sum())
    return DataQualityReport(
        symbol=symbol
        or (str(frame["Symbol"].iloc[0]) if "Symbol" in frame and len(frame) else ""),
        row_count=len(frame),
        start_date=frame.index.min().date().isoformat() if len(frame) else None,
        end_date=frame.index.max().date().isoformat() if len(frame) else None,
        missing_values=missing,
        non_finite_values=non_finite,
        duplicate_dates=duplicates,
        invalid_ohlc_rows=invalid_count,
        negative_volume_rows=negative_volume,
        quality_ok=not (
            missing or non_finite or duplicates or invalid_count or negative_volume
        ),
    )

"""Explicit selection of adjusted and unadjusted stored price series."""

import pandas as pd


def price_series(frame: pd.DataFrame, *, adjusted: bool = False) -> pd.Series:
    """Select a close series without replacing either source column."""
    column = "Adj Close" if adjusted else "Close"
    if column not in frame.columns:
        raise ValueError(f"frame must contain the '{column}' column")
    result = pd.to_numeric(frame[column], errors="coerce")
    if result.dropna().empty:
        raise ValueError(f"'{column}' must contain at least one numeric observation")
    return result.rename(column)

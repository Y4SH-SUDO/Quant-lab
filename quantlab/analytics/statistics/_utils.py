"""Shared validation helpers for quantitative analytics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd


def as_series(
    values: Sequence[float] | pd.Series,
    name: str = "values",
    *,
    dropna: bool = True,
    minimum: int = 1,
) -> pd.Series:
    """Return finite numeric observations with an optional NaN policy."""
    result = pd.Series(values, dtype="float64", copy=True)
    if dropna:
        result = result.dropna()
    if result.empty or len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} observations")
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def validate_window(window: int, observations: int = 1) -> None:
    if not isinstance(window, int) or isinstance(window, bool) or window < observations:
        raise ValueError(f"window must be an integer of at least {observations}")


def aligned_pair(
    first: Sequence[float] | pd.Series,
    second: Sequence[float] | pd.Series,
    *,
    minimum: int = 2,
) -> tuple[pd.Series, pd.Series]:
    """Align two series on their indexes and remove incomplete observations."""
    left = pd.Series(first, dtype="float64", copy=True)
    right = pd.Series(second, dtype="float64", copy=True)
    aligned = pd.concat([left.rename("first"), right.rename("second")], axis=1).dropna()
    if len(aligned) < minimum:
        raise ValueError(f"aligned inputs must contain at least {minimum} observations")
    if not np.isfinite(aligned.to_numpy()).all():
        raise ValueError("aligned inputs must contain only finite values")
    return aligned["first"], aligned["second"]

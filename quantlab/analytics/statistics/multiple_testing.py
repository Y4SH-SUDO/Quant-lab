"""Optional corrections for multiple hypothesis tests."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultipleTestingResult:
    """Unadjusted and adjusted p-values for a family of hypotheses."""

    unadjusted_p_values: pd.Series
    adjusted_p_values: pd.Series
    reject: pd.Series
    alpha: float


def _validate(p_values: Sequence[float] | pd.Series, alpha: float) -> pd.Series:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    values = pd.Series(p_values, dtype="float64", copy=True)
    if values.empty:
        raise ValueError("p_values must contain at least one observation")
    if not np.isfinite(values.to_numpy()).all() or (
        (values < 0) | (values > 1)
    ).any():
        raise ValueError("p_values must be finite and between 0 and 1")
    return values


def bonferroni_correction(
    p_values: Sequence[float] | pd.Series, *, alpha: float = 0.05
) -> MultipleTestingResult:
    """Apply Bonferroni correction and retain original p-values."""
    values = _validate(p_values, alpha)
    adjusted = values.mul(len(values)).clip(upper=1.0).rename("adjusted_p_value")
    return MultipleTestingResult(
        values.rename("unadjusted_p_value"), adjusted, adjusted < alpha, alpha
    )


def benjamini_hochberg_correction(
    p_values: Sequence[float] | pd.Series, *, alpha: float = 0.05
) -> MultipleTestingResult:
    """Apply the Benjamini-Hochberg false-discovery-rate correction."""
    values = _validate(p_values, alpha)
    order = np.argsort(values.to_numpy(), kind="stable")
    ranked = values.to_numpy()[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted_array = np.empty(len(values), dtype="float64")
    adjusted_array[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = pd.Series(adjusted_array, index=values.index, name="adjusted_p_value")
    return MultipleTestingResult(
        values.rename("unadjusted_p_value"), adjusted, adjusted < alpha, alpha
    )


bonferroni = bonferroni_correction
benjamini_hochberg = benjamini_hochberg_correction


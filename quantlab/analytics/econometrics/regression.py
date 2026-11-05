"""Ordinary least-squares regression for aligned time series."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from quantlab.analytics.statistics._utils import aligned_pair


@dataclass(frozen=True)
class OLSResult:
    intercept: float
    slope: float
    r_squared: float
    residuals: pd.Series
    standard_errors: pd.Series
    t_values: pd.Series
    p_values: pd.Series
    observations: int


def ols_regression(
    dependent: Sequence[float] | pd.Series,
    independent: Sequence[float] | pd.Series,
) -> OLSResult:
    y, x = aligned_pair(dependent, independent, minimum=3)
    if x.nunique() < 2:
        raise ValueError("independent series must not be constant")
    fitted = sm.OLS(y, sm.add_constant(x, has_constant="add")).fit()
    return OLSResult(
        float(fitted.params.iloc[0]),
        float(fitted.params.iloc[1]),
        float(fitted.rsquared),
        fitted.resid,
        fitted.bse,
        fitted.tvalues,
        fitted.pvalues,
        int(fitted.nobs),
    )

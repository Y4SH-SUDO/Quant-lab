"""Parametric hypothesis tests with explicit hypotheses and significance levels."""

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd
from scipy import stats

from quantlab.analytics.statistics._utils import as_series


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float
    degrees_of_freedom: float
    alpha: float
    null_hypothesis: str
    alternative_hypothesis: str

    @property
    def reject_null(self) -> bool:
        return self.p_value < self.alpha


def _alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")


def one_sample_ttest(
    sample: Sequence[float] | pd.Series,
    population_mean: float = 0.0,
    *,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    _alpha(alpha)
    observations = as_series(sample, minimum=2)
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    result = stats.ttest_1samp(observations, population_mean, alternative=alternative)
    return TestResult(
        float(result.statistic),
        float(result.pvalue),
        float(len(observations) - 1),
        alpha,
        f"mean = {population_mean}",
        f"mean {alternative} {population_mean}",
    )


def two_sample_ttest(
    first: Sequence[float] | pd.Series,
    second: Sequence[float] | pd.Series,
    *,
    equal_var: bool = False,
    alternative: str = "two-sided",
    alpha: float = 0.05,
) -> TestResult:
    _alpha(alpha)
    left, right = as_series(first, minimum=2), as_series(second, minimum=2)
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")
    result = stats.ttest_ind(left, right, equal_var=equal_var, alternative=alternative)
    if equal_var:
        degrees = len(left) + len(right) - 2
    else:
        left_var, right_var = left.var(), right.var()
        numerator = (left_var / len(left) + right_var / len(right)) ** 2
        denominator = (left_var / len(left)) ** 2 / (len(left) - 1) + (
            right_var / len(right)
        ) ** 2 / (len(right) - 1)
        degrees = numerator / denominator
    return TestResult(
        float(result.statistic),
        float(result.pvalue),
        float(degrees),
        alpha,
        "means are equal",
        f"first mean {alternative} second mean",
    )

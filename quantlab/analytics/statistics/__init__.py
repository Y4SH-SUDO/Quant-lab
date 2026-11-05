"""Statistical analysis modules."""

from quantlab.analytics.statistics.covariance import (
    correlation,
    correlation_matrix,
    covariance,
    rolling_correlation,
)
from quantlab.analytics.statistics.descriptive import (
    data_range,
    describe,
    kurtosis,
    maximum,
    mean,
    median,
    minimum,
    skewness,
    standard_deviation,
    variance,
)
from quantlab.analytics.statistics.hypothesis import (
    TestResult,
    one_sample_ttest,
    two_sample_ttest,
)
from quantlab.analytics.statistics.multiple_testing import (
    MultipleTestingResult,
    benjamini_hochberg,
    benjamini_hochberg_correction,
    bonferroni,
    bonferroni_correction,
)
from quantlab.analytics.statistics.risk import (
    historical_volatility,
    return_dispersion,
    rolling_volatility,
)

__all__ = [
    "TestResult",
    "MultipleTestingResult",
    "benjamini_hochberg",
    "benjamini_hochberg_correction",
    "bonferroni",
    "bonferroni_correction",
    "correlation",
    "correlation_matrix",
    "covariance",
    "data_range",
    "describe",
    "historical_volatility",
    "kurtosis",
    "maximum",
    "mean",
    "median",
    "minimum",
    "one_sample_ttest",
    "return_dispersion",
    "rolling_correlation",
    "rolling_volatility",
    "skewness",
    "standard_deviation",
    "two_sample_ttest",
    "variance",
]

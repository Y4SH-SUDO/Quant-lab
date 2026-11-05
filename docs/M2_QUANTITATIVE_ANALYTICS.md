# M2 Quantitative Analytics

M2 consumes finite numeric `pandas.Series` values, or aligned columns loaded
from the M1 Parquet store. It does not retrieve data and it does not place
orders. For example:

```python
from pathlib import Path
from quantlab.data.storage.parquet import ParquetMarketStore
from quantlab.analytics.statistics import historical_volatility
from quantlab.analytics.statistics.returns import simple_returns

prices = ParquetMarketStore(Path("data/processed")).read("RELIANCE")["Close"]
returns = simple_returns(prices).dropna()
annual_volatility = historical_volatility(returns, periods_per_year=252)
```

The modules provide descriptive statistics, annualized risk statistics,
date-aligned covariance and correlation, one- and two-sample t-tests, ADF
stationarity and differencing, ACF/PACF diagnostics, OLS and Engle-Granger
cointegration, ARIMA forecasting, and GARCH conditional volatility.

`fit_arima` reports `converged`, `fit_succeeded`, and any fit warnings while
still surfacing those warnings to callers. `fit_garch(..., forecast_steps=N)`
returns `forecast_variance` and `forecast_volatility` indexed by forecast
horizon; both are expressed in the same return units as the input.

`bonferroni_correction` and `benjamini_hochberg_correction` return both
unadjusted and adjusted p-values, along with the rejection decisions. These
are optional inferential corrections and do not change the original values.

M1 stores unadjusted `Open`, `High`, `Low`, and `Close` fields and retains
Yahoo Finance's `Adj Close` separately. Use
`quantlab.data.cleaning.price_series(frame, adjusted=True)` to select the
adjusted series explicitly. Yahoo Finance's adapter requests
`auto_adjust=False`, does not download corporate-action rows, and cannot
guarantee complete split/dividend history; adjusted data must therefore be
reviewed before research use.

Inputs with insufficient observations, non-finite values, invalid parameters,
constant regressors, or model-fitting failures raise `ValueError`. ADF
rejection is evidence against a unit root, not proof of every condition of
stationarity. Statistical significance is not trading profitability: costs,
execution, stability, and out-of-sample validation still matter. ARIMA and
GARCH forecasts do not guarantee profitable predictions.

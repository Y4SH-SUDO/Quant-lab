# M2.10 — Model validation and forecast evaluation

Source: saved M2.9 Yahoo Parquet files; no refetch. Targets are one-session simple returns from Adj Close in decimal units. Each series uses a chronological 80/20 split. ARIMA(1,0,1) and GARCH(1,1) are fit on training data, then issue fixed-origin forecasts over the full test horizon. Later test observations are not incorporated into forecasts.

GARCH evaluation uses squared test returns as a noisy proxy for latent variance. Variance units are decimal return squared per session, without annualization. QLIKE and squared-return RMSE are reported. No statistical significance or trading profitability claim is made.

| Instrument | Test n | ARIMA RMSE | Zero RMSE | Mean RMSE | Direction | GARCH QLIKE | Historical QLIKE | GARCH RMSE | Historical RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RELIANCE | 536 | 0.0141124 | 0.0140672 | 0.0141121 | 0.493 | -7.47243 | -7.43941 | 0.000654208 | 0.00065852 |
| TCS | 536 | 0.0160465 | 0.0159873 | 0.0160485 | 0.444 | -7.26441 | -7.26519 | 0.000612835 | 0.000612702 |
| INFY | 536 | 0.0178845 | 0.0178379 | 0.0178844 | 0.476 | -7.0473 | -7.04448 | 0.000662072 | 0.00066237 |
| NIFTY50 | 535 | 0.00840253 | 0.00838202 | 0.00840263 | 0.488 | -8.5213 | -8.4709 | 0.000144761 | 0.000148392 |

Detailed metrics, residual diagnostics, and date boundaries are in `evaluation.json`.

## Limitations

This is one historical holdout and is sensitive to regime choice. Repeated tuning on it would make it no longer an untouched evaluation. Diagnostics are descriptive. Repeated walk-forward evaluation and uncertainty-aware comparisons are needed before M3 relies on these forecasts. Convergence and in-sample fit do not demonstrate forecasting skill.

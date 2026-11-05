# M2.9 — Real Market Data Validation

**Run:** 2016-01-01 through 2026-11-05 (Yahoo end date 2026-11-06, exclusive)  
**Provider:** Existing Yahoo Finance adapter (`yfinance`), `auto_adjust=False`  
**Reproduce:** `python -m scripts.validate_real_market_data --start 2016-01-01 --end 2026-11-06`

## Data checks

Real daily bars were retrieved for RELIANCE.NS, TCS.NS, INFY.NS, and Yahoo's
NIFTY 50 symbol `^NSEI`. The provider passes caret-prefixed index symbols
through unchanged. All four saved Parquet datasets had zero reported missing
OHLCV values, duplicate dates, invalid OHLC/price rows, and negative-volume
rows.

| Instrument | Rows | Date range | Missing OHLCV values | Duplicates | Invalid OHLC/price rows | Negative volume |
|---|---:|---|---:|---:|---:|---:|
| RELIANCE | 2,679 | 2016-01-01–2026-11-05 | 0 | 0 | 0 | 0 |
| TCS | 2,679 | 2016-01-01–2026-11-05 | 0 | 0 | 0 | 0 |
| INFY | 2,679 | 2016-01-01–2026-11-05 | 0 | 0 | 0 | 0 |
| NIFTY50 | 2,672 | 2016-01-04–2026-11-05 | 0 | 0 | 0 | 0 |

The three stock series share 2,679 observed dates; NIFTY50 shares 2,672 of
them. The seven stock dates without an index observation were checked against
the NSE Capital Market holiday circulars/calendar:

| Date | NSE calendar/session context | Data interpretation |
|---|---|---|
| 2016-01-01 (Friday) | Not listed as a Capital Market holiday in 2016 | Stock observations precede the NIFTY50 dataset's first date (2016-01-04); outside index coverage, not an index gap within its coverage |
| 2016-08-12 (Friday) | Not listed as a 2016 Capital Market holiday | Index provider observation absent on a regular weekday; requires source-level verification |
| 2018-01-01 (Monday) | Not listed as a 2018 Capital Market holiday | Index provider observation absent on a regular weekday; requires source-level verification |
| 2019-01-01 (Tuesday) | Not listed as a 2019 Capital Market holiday | Index provider observation absent on a regular weekday; requires source-level verification |
| 2019-10-27 (Sunday) | Diwali-Laxmi Pujan; NSE announced a Muhurat trading session | Weekend date had a special trading session; index observation absent, so weekend alone does not explain the gap |
| 2020-11-14 (Saturday) | NSE announced a Diwali Muhurat live trading session | Weekend date had a special trading session; index observation absent |
| 2026-01-15 (Thursday) | NSE's 2026 calendar declares a Capital Market holiday for Municipal Corporation Election in Maharashtra | Stock rows exist on an official closure date; treat these provider rows as a calendar conflict requiring primary-record checks, not automatically as valid sessions or a proven data error |

This review is deliberately limited to those seven observed stock/index
mismatches. It does not reconstruct every NSE session in the eleven-year
period, nor prove that Yahoo's remaining observations are complete. NSE
notes equities ordinarily trade Monday–Friday except declared holidays, and
its circulars explicitly call out special Muhurat sessions on weekends.
Official references: [NSE 2016 holiday circular NSE/CMTR/31297](https://vardhamancapital.co.in/downloads/vcapital_download/Trading%20Holidays%20for%20the%20Year%202016.pdf), [NSE 2018 holiday circular NSE/CMTR/36475](https://nsearchives.nseindia.com/content/circulars/CMTR36475.pdf), [NSE 2019 holiday circular NSE/CMTR/39612](https://nsearchives.nseindia.com/content/circulars/CMTR39612.pdf), [NSE 2020 Muhurat-session circular NSE/CMTR/46230](https://nsearchives.nseindia.com/content/circulars/CMTR46230.pdf), and [NSE Market Timings & Holidays (2026 calendar)](https://www.nseindia.com/resources/exchange-communication-holidays?article_id=432919950.0).

## Adjusted close and return choice

Yahoo documents `Adj Close` as a historical close adjusted for applicable
splits and dividend distributions using its stated CRSP-style multipliers.
With `auto_adjust=False`, yfinance preserves separate `Close` and `Adj Close`
columns; the explicit `price_series(adjusted=True)` selector therefore uses
Yahoo's adjusted close without changing the source OHLC columns. For
split/dividend-adjusted, total-return-style price returns, this is an
appropriate convenience series. It is not the right field for raw traded-price
returns or execution-price/backtest fills, which should use unadjusted `Close`
(and explicitly model corporate actions/cash flows as needed).

Yahoo's description establishes the intended adjustment convention, not the
completeness or correctness of Yahoo's event inputs for these Indian securities.
This run set `actions=False`, did not independently reconcile dividends,
splits, rights/bonus issues or other corporate actions against issuer/NSE
records, and did not verify point-in-time adjustment vintages. Thus no claim is
made that the adjusted series is complete or suitable as official corporate-
action data. Source: [Yahoo Help — What is the adjusted close?](https://in.help.yahoo.com/kb/adjusted-close-sln28256.html); the pinned adapter behavior is documented in [yfinance download parameters](https://github.com/ranaroussi/yfinance/blob/0.2.59/yfinance/multi.py).

## M2 analytics and ARIMA follow-up

The pipeline explicitly selected Yahoo `Adj Close` and used 2,672 common
prices / 2,671 returns. It ran descriptive return statistics, annualized
volatility, aligned correlation and covariance, t-tests, ADF tests on adjusted
levels and returns, second differences, ACF/PACF, OLS against NIFTY50,
Engle–Granger cointegration, ARIMA and GARCH forecasts, and multiple-testing
corrections.

Annualized sample volatility (252 sessions) was 27.0% for RELIANCE, 24.3% for
TCS, 27.0% for INFY, and 16.1% for NIFTY50. RELIANCE/NIFTY50 return
correlation was 0.648. These are in-sample descriptive results, not evidence of
predictive performance or profitability.

ARIMA(1,0,1) was fit to 2,671 RELIANCE common-date adjusted-price returns.
The wrapper uses integer session positions, since exchange dates have no fixed
calendar frequency, and normalizes the input to unit standard deviation for
optimizer conditioning. It rescales forecasts and residuals back to return
units and corrects AIC/BIC for the likelihood scale. The unscaled optimizer
reported zero iterations and `converged=false`; the normalized fit converged in
13 iterations with no warnings. Its original-unit AIC/BIC are -14,180.157 /
-14,156.596. The five-step forecast now comes from the converged fit. It has not
been validated out of sample or with a full residual diagnostic study, so
convergence alone is not evidence of predictive validity. GARCH(1,1) fit and
produced a five-step volatility forecast.

## Reproducibility and limitations

- `validation.json` records quality counts, observed-date alignment and the
  seven-date NSE review, analytics, warnings, selected price field, symbols,
  and requested dates.
- Four `*_daily.parquet` files preserve the normalized bars from the original
  Yahoo run. The calendar/ARIMA follow-up used these files; it did not refetch
  market data or overwrite them.
- `scripts/validate_real_market_data.py` refetches the date range and
  recreates the outputs. Its scoped NSE annotations classify the seven known
  observed differences; they are not a full machine-readable session calendar.
- Yahoo history may be revised or unavailable. Official session comparison is
  limited to the seven observed differences. Corporate-action completeness,
  point-in-time adjustments, and forecast validity remain unverified.

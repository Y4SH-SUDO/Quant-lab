# Development Log

## 2026-11-04 — Milestone M0

Implemented the initial Python 3.12 project foundation, pinned dependency
configuration, package layout, FastAPI health endpoint, mathematical utilities,
documentation, and focused tests. Verification is run as part of the M0
completion process.

Verification completed:

- `pip check`: passed.
- `pytest`: 17 passed, with one upstream Starlette/AnyIO deprecation warning.
- `ruff check .`: passed.
- `mypy`: passed for 25 source files.
- Direct imports and `GET /health`: passed.
- Required `.gitignore` rules: present. This workspace has no Git metadata, so
  `git check-ignore` was not available.

### Outstanding

- Market-data providers, cleaning pipelines, and storage are not implemented.
- Strategies, backtesting, risk, portfolio, execution, and dashboard features
  are not implemented.
- No broker integration, live order execution, or historical data is included.

## 2026-11-04 — Milestone M1

Implemented a daily historical NSE data provider interface and Yahoo Finance
adapter, configurable instrument catalogue, canonical OHLCV normalization,
quality validation/reporting, and local Parquet persistence with JSON metadata.
The pipeline validates each fetched instrument before writing. Tests use
deterministic synthetic daily bars and a fake provider; they do not make network
requests. No live/intraday feed, strategy, backtest, or trading capability was
added. Corporate-action research and provider data quality still require
independent review.

Verification:

- `.\.venv\Scripts\python.exe -m pytest`: 24 passed.
- `.\.venv\Scripts\python.exe -m ruff check .`: passed.
- `.\.venv\Scripts\python.exe -m mypy`: passed for 32 source files.
- Pytest reports one existing Starlette/AnyIO deprecation warning.
- Yahoo Finance requests explicitly set `auto_adjust=False`; direct verification
  of the requested 2026 RELIANCE dates was unavailable because DNS resolution
  for Yahoo Finance failed in this environment.

Final M1 acceptance checks also confirmed Parquet round-trip preservation and
metadata fields for the unadjusted OHLC policy. Duplicate dates remain visible
to validation and are not silently dropped during normalization.

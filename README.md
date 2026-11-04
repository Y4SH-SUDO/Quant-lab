# QUANT LAB

QUANT LAB is a modular foundation for quantitative research and algorithmic
paper-trading experiments focused initially on Indian equity markets (NSE/BSE).
Milestone M0 provides project structure, configuration, a health endpoint, and
tested mathematical utilities. M1 adds daily historical NSE equity data
retrieval, OHLCV normalization and validation, local Parquet storage, an
instrument catalogue, and quality metadata. No live/intraday feeds, strategies,
backtests, broker integration, or order execution are included.

## Daily market data (M1)

Configure NSE instruments in `data/instruments.json` or another JSON catalogue
using `InstrumentCatalogue`.
`DailyDataPipeline` accepts any `DailyHistoryProvider`; the included
`YFinanceDailyProvider` maps symbols to Yahoo Finance `.NS` tickers and fetches
daily bars with `auto_adjust=False`. The stored `Open`, `High`, `Low`, and
`Close` columns are therefore unadjusted prices; Yahoo Finance's `Adj Close`
is retained as a separate comparison series. Corporate-action rows are not
downloaded by the M1 provider and are not represented in the stored dataset.
The pipeline normalizes and validates bars before storing them under
`data/processed/daily/symbol=SYMBOL/` as Parquet plus a JSON quality report.
Provider calls are network-dependent; tests use synthetic bars and never access
the network. Data availability and corporate-action correctness must be
independently reviewed before research use.

## Technology stack

Python 3.12, NumPy, pandas, SciPy, Polars, statsmodels, scikit-learn, arch,
yfinance, PyArrow, FastAPI, Pydantic, pytest, Ruff, mypy, and JupyterLab.

## Installation

From PowerShell:

```powershell
& "C:\Users\yashk\Project\quant_lab\.venv\Scripts\python.exe" -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

Activation is optional. If PowerShell permits user-scoped scripts, use
`.\.venv\Scripts\Activate.ps1`; otherwise run commands through
`.\.venv\Scripts\python.exe` as shown above. No execution policy change is
required.

## Development commands

```powershell
& ".\.venv\Scripts\python.exe" -m pytest
& ".\.venv\Scripts\python.exe" -m ruff check .
& ".\.venv\Scripts\python.exe" -m mypy
& ".\.venv\Scripts\python.exe" -m uvicorn backend.main:app --reload
```

The backend health check is available at `http://127.0.0.1:8000/health`.

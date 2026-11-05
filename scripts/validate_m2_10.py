"""Evaluate M2 models on saved M2.9 Parquet data; no network access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.analytics.timeseries.evaluation import evaluate_returns
from quantlab.data.cleaning.prices import price_series

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "research" / "validation" / "m2_9"
OUTPUT = ROOT / "research" / "validation" / "m2_10"
FILES = {
    "RELIANCE": "reliance_daily.parquet",
    "TCS": "tcs_daily.parquet",
    "INFY": "infy_daily.parquet",
    "NIFTY50": "nifty50_daily.parquet",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name, filename in FILES.items():
        bars = pd.read_parquet(INPUT / filename)
        prices = price_series(bars, adjusted=True).dropna()
        results[name] = evaluate_returns(prices.pct_change().dropna())
    payload = {
        "design": {
            "source": "saved M2.9 Parquet; no refetch",
            "instruments": list(FILES),
            "train_fraction": 0.8,
            "annualization": "none; variance and volatility metrics are per session",
        },
        "results": results,
    }
    (OUTPUT / "evaluation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# M2.10 — Model validation and forecast evaluation",
        "",
        (
            "Source: saved M2.9 Yahoo Parquet files; no refetch. Targets are "
            "one-session simple returns from Adj Close in decimal units. Each "
            "series uses a chronological 80/20 split. ARIMA(1,0,1) and "
            "GARCH(1,1) are fit on training data, then issue fixed-origin "
            "forecasts over the full test horizon. Later test observations are "
            "not incorporated into forecasts."
        ),
        "",
        (
            "GARCH evaluation uses squared test returns as a noisy proxy for "
            "latent variance. Variance units are decimal return squared per "
            "session, without annualization. QLIKE and squared-return RMSE are "
            "reported. No statistical significance or trading profitability "
            "claim is made."
        ),
        "",
        "| Instrument | Test n | ARIMA RMSE | Zero RMSE | Mean RMSE | Direction | "
        "GARCH QLIKE | Historical QLIKE | GARCH RMSE | Historical RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in results.items():
        arima, garch = result["arima"], result["garch"]
        historical = garch["benchmarks"]["training_historical_variance"]
        lines.append(
            f"| {name} | {result['n_test']} | {arima['model']['rmse']:.6g} | "
            f"{arima['benchmarks']['zero']['rmse']:.6g} | "
            f"{arima['benchmarks']['training_mean']['rmse']:.6g} | "
            f"{arima['model']['directional_accuracy']:.3f} | "
            f"{garch['model']['qlike']:.6g} | {historical['qlike']:.6g} | "
            f"{garch['model']['rmse_squared_return']:.6g} | "
            f"{historical['rmse_squared_return']:.6g} |"
        )
    lines += [
        "",
        "Detailed metrics, residual diagnostics, and date boundaries are in "
        "`evaluation.json`.",
        "",
        "## Limitations",
        "",
        (
            "This is one historical holdout and is sensitive to regime choice. "
            "Repeated tuning on it would make it no longer an untouched "
            "evaluation. Diagnostics are descriptive. Repeated walk-forward "
            "evaluation and uncertainty-aware comparisons are needed before "
            "M3 relies on these forecasts. Convergence and in-sample fit do "
            "not demonstrate forecasting skill."
        ),
        "",
    ]
    (OUTPUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

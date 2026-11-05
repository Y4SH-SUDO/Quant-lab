"""Fetch and validate a real NSE sample, then run the public M2 analytics API.

Example:
    python -m scripts.validate_real_market_data --start 2016-01-01 --end 2026-11-06

Yahoo's end date is exclusive. The script stores the provider-shaped daily bars,
quality summaries, and analytics beside this script's validation outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.analytics import (
    adf_test,
    autocorrelation,
    benjamini_hochberg_correction,
    bonferroni_correction,
    covariance,
    describe,
    difference,
    engle_granger_test,
    fit_arima,
    fit_garch,
    historical_volatility,
    ols_regression,
    one_sample_ttest,
    partial_autocorrelation,
    two_sample_ttest,
)
from quantlab.analytics.statistics.covariance import correlation_matrix
from quantlab.data.cleaning.ohlcv import normalize_ohlcv
from quantlab.data.cleaning.prices import price_series
from quantlab.data.providers.yfinance_provider import YFinanceDailyProvider
from quantlab.data.validation.ohlcv import validate_ohlcv

SYMBOLS = {"RELIANCE": "RELIANCE", "TCS": "TCS", "INFY": "INFY", "NIFTY50": "^NSEI"}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "research" / "validation" / "m2_9"

# Manual review of the only stock/index date differences in the saved M2.9 run.
# These are source-calendar annotations, not a synthetic complete exchange calendar.
NSE_DATE_REVIEW = {
    "2016-01-01": {
        "nse_status": "regular weekday; not listed as a 2016 Capital Market holiday",
        "special_session": False,
        "source": "NSE/CMTR/31297 (2016 trading holiday circular)",
    },
    "2016-08-12": {
        "nse_status": "regular weekday; not listed as a 2016 Capital Market holiday",
        "special_session": False,
        "source": "NSE/CMTR/31297 (2016 trading holiday circular)",
    },
    "2018-01-01": {
        "nse_status": "regular weekday; not listed as a 2018 Capital Market holiday",
        "special_session": False,
        "source": "NSE/CMTR/36475 (2018 trading holiday circular)",
    },
    "2019-01-01": {
        "nse_status": "regular weekday; not listed as a 2019 Capital Market holiday",
        "special_session": False,
        "source": "NSE/CMTR/39612 (2019 trading holiday circular)",
    },
    "2019-10-27": {
        "nse_status": "Sunday; Diwali-Laxmi Pujan",
        "special_session": True,
        "source": (
            "NSE/CMTR/39612 (2019 trading holiday circular; Muhurat session noted)"
        ),
    },
    "2020-11-14": {
        "nse_status": "Saturday; Diwali Muhurat trading session",
        "special_session": True,
        "source": "NSE/CMTR/46230 (2020 Muhurat session circular)",
    },
    "2026-01-15": {
        "nse_status": (
            "official Capital Market holiday: Municipal Corporation Election "
            "in Maharashtra"
        ),
        "special_session": False,
        "source": "NSE Market Timings & Holidays, 2026 Capital Market calendar",
    },
}


def _date_review(date_sets: dict[str, set[pd.Timestamp]]) -> dict[str, Any]:
    union = set.union(*date_sets.values())
    index_dates = date_sets["NIFTY50"]
    rows: list[dict[str, Any]] = []
    for day in sorted(union - index_dates):
        iso_day = day.date().isoformat()
        annotation = NSE_DATE_REVIEW.get(iso_day, {})
        present = sorted(symbol for symbol, dates in date_sets.items() if day in dates)
        rows.append({
            "date": iso_day,
            "weekday": day.day_name(),
            "weekend": day.dayofweek >= 5,
            "observed_in": present,
            "missing_provider_observation": sorted(set(date_sets) - set(present)),
            "outside_index_coverage": day < min(index_dates) or day > max(index_dates),
            "nse_status": annotation.get("nse_status", "not reviewed"),
            "special_session": annotation.get("special_session", False),
            "official_source": annotation.get("source"),
        })
    return {
        "scope": (
            "Observed stock/index date differences only; this is not a full "
            "exchange-session reconstruction."
        ),
        "dates": rows,
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, pd.Timestamp | date):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def run(start: date, end: date) -> dict[str, Any]:
    if start >= end:
        raise ValueError("start must be earlier than end")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    provider = YFinanceDailyProvider()
    frames: dict[str, pd.DataFrame] = {}
    quality: dict[str, dict[str, Any]] = {}

    for symbol, provider_symbol in SYMBOLS.items():
        raw = provider.fetch_daily(provider_symbol, start, end)
        frame = normalize_ohlcv(raw, symbol)
        frames[symbol] = frame
        report = validate_ohlcv(frame, symbol)
        quality[symbol] = report.to_dict()
        frame.to_parquet(OUTPUT_DIR / f"{symbol.lower()}_daily.parquet")

    date_sets = {symbol: set(frame.index) for symbol, frame in frames.items()}
    date_union = set.union(*date_sets.values())
    date_intersection = set.intersection(*date_sets.values())
    alignment = {
        "union_trading_dates": len(date_union),
        "common_dates_all_instruments": len(date_intersection),
        "per_instrument": {
            symbol: {
                "dates": len(dates),
                "dates_missing_from_union": sorted(
                    day.date().isoformat() for day in date_union - dates
                ),
            }
            for symbol, dates in date_sets.items()
        },
        "note": (
            "Union is observed dates across these four instruments, "
            "not an official exchange calendar."
        ),
        "nse_date_review": _date_review(date_sets),
    }

    adjusted = {
        symbol: price_series(frame, adjusted=True).rename(symbol)
        for symbol, frame in frames.items()
    }
    prices = pd.concat(adjusted.values(), axis=1).sort_index()
    aligned_prices = prices.dropna()
    returns = aligned_prices.pct_change(fill_method=None).dropna()
    analysis: dict[str, Any] = {
        "price_selector": "Adj Close via price_series(adjusted=True)",
        "aligned_observations": len(aligned_prices),
        "return_observations": len(returns),
        "return_descriptives": {
            symbol: describe(returns[symbol]).to_dict() for symbol in returns
        },
        "annualized_historical_volatility_252": {
            symbol: historical_volatility(returns[symbol], periods_per_year=252)
            for symbol in returns
        },
        "correlation_matrix": correlation_matrix(returns).to_dict(),
        "covariance_reliance_nifty50": covariance(
            returns["RELIANCE"], returns["NIFTY50"]
        ),
        "first_vs_second_half_reliance_ttest": _jsonable(
            two_sample_ttest(
                returns["RELIANCE"].iloc[: len(returns) // 2],
                returns["RELIANCE"].iloc[len(returns) // 2 :],
            )
        ),
        "adf_adjusted_price": {
            symbol: _jsonable(adf_test(aligned_prices[symbol]))
            for symbol in aligned_prices
        },
        "adf_returns": {
            symbol: _jsonable(adf_test(returns[symbol])) for symbol in returns
        },
        "second_difference_reliance_log_price_preview": difference(
            aligned_prices["RELIANCE"].apply("log"), order=2
        )
        .head(5)
        .to_dict(),
        "reliance_return_acf_10": autocorrelation(returns["RELIANCE"], 10).to_dict(),
        "reliance_return_pacf_10": partial_autocorrelation(
            returns["RELIANCE"], 10
        ).to_dict(),
        "reliance_vs_nifty_return_ols": _jsonable(
            ols_regression(returns["RELIANCE"], returns["NIFTY50"])
        ),
        "reliance_nifty_level_cointegration": _jsonable(
            engle_granger_test(aligned_prices["RELIANCE"], aligned_prices["NIFTY50"])
        ),
    }
    analysis["one_sample_return_tests_against_zero"] = {
        symbol: _jsonable(one_sample_ttest(returns[symbol], population_mean=0.0))
        for symbol in returns
    }
    p_values = pd.Series(
        {
            symbol: result["p_value"]
            for symbol, result in analysis[
                "one_sample_return_tests_against_zero"
            ].items()
        }
    )
    analysis["multiple_testing_bonferroni"] = _jsonable(bonferroni_correction(p_values))
    analysis["multiple_testing_benjamini_hochberg"] = _jsonable(
        benjamini_hochberg_correction(p_values)
    )
    arima = fit_arima(returns["RELIANCE"], order=(1, 0, 1), forecast_steps=5)
    analysis["reliance_arima_returns"] = _jsonable(
        {
            "forecast": arima.forecast,
            "aic": arima.aic,
            "bic": arima.bic,
            "converged": arima.converged,
            "warnings": arima.warnings,
        }
    )
    garch = fit_garch(returns["RELIANCE"], p=1, q=1, forecast_steps=5)
    analysis["reliance_garch_returns"] = _jsonable(
        {
            "conditional_volatility_tail": garch.conditional_volatility.tail(10),
            "aic": garch.aic,
            "bic": garch.bic,
            "forecast_variance": garch.forecast_variance,
            "forecast_volatility": garch.forecast_volatility,
        }
    )

    result = {
        "provider": "Yahoo Finance via existing yfinance adapter",
        "provider_adjustment": (
            "auto_adjust=False; unadjusted OHLC and Adj Close retained"
        ),
        "requested_start_inclusive": start.isoformat(),
        "requested_end_exclusive": end.isoformat(),
        "symbols": SYMBOLS,
        "quality": quality,
        "date_alignment": alignment,
        "analytics": analysis,
        "limitations": [
            (
                "Yahoo Finance availability and adjusted prices are not "
                "independently audited."
            ),
            (
                "Corporate-action rows are not downloaded; adjusted close is used "
                "explicitly for return analytics."
            ),
            (
                "Observed-date alignment does not establish completeness against "
                "the official NSE trading calendar."
            ),
            (
                "Statistical outputs are descriptive validation, not evidence of "
                "trading profitability."
            ),
        ],
    }
    (OUTPUT_DIR / "validation.json").write_text(
        json.dumps(_jsonable(result), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2016, 1, 1))
    parser.add_argument(
        "--end", type=date.fromisoformat, default=date.today() + timedelta(days=1)
    )
    arguments = parser.parse_args()
    result = run(arguments.start, arguments.end)
    print(
        json.dumps(
            {
                "output_dir": str(OUTPUT_DIR),
                "quality": result["quality"],
                "date_alignment": result["date_alignment"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

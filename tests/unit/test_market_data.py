"""Deterministic M1 market-data tests; no provider network calls."""

from datetime import date

import pandas as pd
import pytest

from quantlab.data.catalogue import Instrument, InstrumentCatalogue
from quantlab.data.cleaning.ohlcv import normalize_ohlcv
from quantlab.data.pipeline import DailyDataPipeline
from quantlab.data.providers.base import DailyHistoryProvider
from quantlab.data.providers.yfinance_provider import YFinanceDailyProvider
from quantlab.data.storage.parquet import ParquetMarketStore
from quantlab.data.validation.ohlcv import validate_ohlcv
from tests.fixtures.daily_ohlcv import daily_bars


class FakeProvider(DailyHistoryProvider):
    def fetch_daily(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        assert symbol == "RELIANCE.NS"
        return daily_bars()


def test_normalization_sorts_dates_and_adds_symbol() -> None:
    source = daily_bars().iloc[::-1]
    normalized = normalize_ohlcv(source, "reliance")
    assert normalized.index.is_monotonic_increasing
    assert normalized["Symbol"].unique().tolist() == ["RELIANCE"]
    assert list(normalized.columns) == [
        "Symbol", "Open", "High", "Low", "Close", "Adj Close", "Volume"
    ]


def test_quality_report_detects_invalid_rows() -> None:
    bars = normalize_ohlcv(daily_bars(), "RELIANCE")
    bars.loc[bars.index[0], "High"] = 90
    bars.loc[bars.index[1], "Volume"] = -1
    report = validate_ohlcv(bars)
    assert report.invalid_ohlc_rows == 1
    assert report.negative_volume_rows == 1
    assert not report.quality_ok


def test_quality_report_detects_non_finite_and_non_positive_prices() -> None:
    bars = normalize_ohlcv(daily_bars(), "RELIANCE")
    bars.loc[bars.index[0], "Close"] = 0
    bars["Volume"] = bars["Volume"].astype(float)
    bars.loc[bars.index[1], "Volume"] = float("inf")
    report = validate_ohlcv(bars)
    assert report.non_finite_values == 1
    assert report.invalid_ohlc_rows == 1
    assert not report.quality_ok


def test_normalization_retains_duplicate_dates_for_reporting() -> None:
    source = pd.concat([daily_bars(), daily_bars().iloc[[0]]])
    normalized = normalize_ohlcv(source, "RELIANCE")
    report = validate_ohlcv(normalized)
    assert len(normalized) == 4
    assert report.duplicate_dates == 2
    assert not report.quality_ok


def test_pipeline_writes_parquet_and_metadata(tmp_path) -> None:
    store = ParquetMarketStore(tmp_path)
    pipeline = DailyDataPipeline(FakeProvider(), store)
    catalogue = InstrumentCatalogue(
        [Instrument("reliance", name="Reliance Industries")]
    )
    reports = pipeline.update(catalogue, date(2024, 1, 1), date(2024, 2, 1))
    assert reports["RELIANCE"].quality_ok
    pd.testing.assert_frame_equal(
        store.read("reliance"),
        normalize_ohlcv(daily_bars(), "RELIANCE"),
        check_freq=False,
    )
    assert (tmp_path / "daily" / "symbol=RELIANCE" / "history.json").exists()
    metadata = (
        tmp_path / "daily" / "symbol=RELIANCE" / "history.json"
    ).read_text(encoding="utf-8")
    assert '"price_adjustment": "unadjusted_ohlc"' in metadata
    assert '"adjusted_close_column": "Adj Close"' in metadata
    assert '"corporate_actions": "not_downloaded"' in metadata
    store.write_catalogue(catalogue)
    restored = InstrumentCatalogue.from_json(tmp_path / "catalogue.json")
    assert restored.get("RELIANCE").provider_symbol == "RELIANCE.NS"


def test_catalogue_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="unique"):
        InstrumentCatalogue([Instrument("TCS"), Instrument("tcs")])


def test_yfinance_provider_maps_nse_symbol_and_adjustment(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(*args: object, **kwargs: object) -> pd.DataFrame:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return daily_bars()

    monkeypatch.setattr(
        "quantlab.data.providers.yfinance_provider.yf.download", fake_download
    )
    result = YFinanceDailyProvider().fetch_daily(
        "reliance", date(2024, 1, 1), date(2024, 2, 1)
    )
    assert not result.empty
    assert captured["args"] == ("RELIANCE.NS",)
    assert captured["kwargs"]["auto_adjust"] is False
    assert captured["kwargs"]["actions"] is False
    assert captured["kwargs"]["multi_level_index"] is False

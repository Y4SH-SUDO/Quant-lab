"""Orchestration for fetching, normalizing, validating and storing daily bars."""

from datetime import date

from quantlab.data.catalogue import InstrumentCatalogue
from quantlab.data.cleaning.ohlcv import normalize_ohlcv
from quantlab.data.providers.base import DailyHistoryProvider
from quantlab.data.storage.parquet import ParquetMarketStore
from quantlab.data.validation.ohlcv import DataQualityReport, validate_ohlcv


class DailyDataPipeline:
    """Run daily history updates for selected catalogue instruments."""

    def __init__(
        self, provider: DailyHistoryProvider, store: ParquetMarketStore
    ) -> None:
        self.provider = provider
        self.store = store

    def update(
        self, catalogue: InstrumentCatalogue, start: date, end: date
    ) -> dict[str, DataQualityReport]:
        if start >= end:
            raise ValueError("start must be earlier than end")
        reports: dict[str, DataQualityReport] = {}
        for instrument in catalogue.active():
            assert instrument.provider_symbol is not None
            raw = self.provider.fetch_daily(instrument.provider_symbol, start, end)
            normalized = normalize_ohlcv(raw, instrument.symbol)
            report = validate_ohlcv(normalized, instrument.symbol)
            if not report.quality_ok:
                raise ValueError(
                    f"quality validation failed for {instrument.symbol}: {report}"
                )
            self.store.write(instrument.symbol, normalized, report)
            reports[instrument.symbol] = report
        return reports

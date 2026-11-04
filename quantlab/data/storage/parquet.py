"""Local Parquet persistence and catalogue metadata."""

import json
from pathlib import Path

import pandas as pd

from quantlab.data.catalogue import InstrumentCatalogue
from quantlab.data.validation.ohlcv import DataQualityReport


class ParquetMarketStore:
    """Store daily history and metadata under a configurable root."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self, symbol: str, frame: pd.DataFrame, report: DataQualityReport
    ) -> Path:
        destination = (
            self.root / "daily" / f"symbol={symbol.upper()}" / "history.parquet"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=True)
        metadata = {
            "symbol": symbol.upper(),
            "frequency": "1d",
            "price_adjustment": "unadjusted_ohlc",
            "adjusted_close_column": "Adj Close",
            "corporate_actions": "not_downloaded",
            "quality": report.to_dict(),
        }
        serialized = json.dumps(metadata, indent=2) + "\n"
        destination.with_suffix(".json").write_text(serialized, encoding="utf-8")
        return destination

    def read(self, symbol: str) -> pd.DataFrame:
        source = self.root / "daily" / f"symbol={symbol.upper()}" / "history.parquet"
        return pd.read_parquet(source)

    def write_catalogue(self, catalogue: InstrumentCatalogue) -> Path:
        target = self.root / "catalogue.json"
        catalogue.to_json(target)
        return target

from collections.abc import Iterator
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class Instrument:
    """One instrument available to the daily historical data pipeline."""

    symbol: str
    name: str = ""
    exchange: str = "NSE"
    currency: str = "INR"
    provider_symbol: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        if not symbol:
            raise ValueError("symbol must not be empty")
        if self.exchange.upper() != "NSE":
            raise ValueError("only NSE instruments are supported")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "exchange", "NSE")
        if self.provider_symbol is None:
            object.__setattr__(self, "provider_symbol", f"{symbol}.NS")


class InstrumentCatalogue:
    """In-memory catalogue with JSON load/save support."""

    def __init__(self, instruments: list[Instrument]) -> None:
        symbols = [item.symbol for item in instruments]
        if len(symbols) != len(set(symbols)):
            raise ValueError("instrument symbols must be unique")
        self._items = {item.symbol: item for item in instruments}

    def __iter__(self) -> Iterator[Instrument]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def get(self, symbol: str) -> Instrument:
        return self._items[symbol.strip().upper()]

    def active(self) -> list[Instrument]:
        return [item for item in self._items.values() if item.active]

    @classmethod
    def from_json(cls, path: Path) -> "InstrumentCatalogue":
        rows = json.loads(path.read_text(encoding="utf-8"))
        return cls([Instrument(**row) for row in rows])

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps([asdict(item) for item in self], indent=2) + "\n"
        path.write_text(serialized, encoding="utf-8")

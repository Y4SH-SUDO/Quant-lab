"""Reusable interfaces for deterministic trading strategies."""

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """Interface implemented by strategies that generate date-aligned signals."""

    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Generate signals aligned with the input market-data index."""

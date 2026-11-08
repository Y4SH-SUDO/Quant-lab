"""Long-only price momentum strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy
from quantlab.strategies.validation import _validated_close


@dataclass(frozen=True)
class MomentumStrategy(Strategy):
    """Generate long-only signals from an n-observation price momentum."""

    lookback: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.lookback, int)
            or isinstance(self.lookback, bool)
            or self.lookback < 1
        ):
            raise ValueError("lookback must be a positive integer")

    def calculate_momentum(self, market_data: pd.DataFrame) -> pd.Series:
        """Return percentage momentum, aligned with the input dates."""
        close = _validated_close(market_data)
        momentum = (close / close.shift(self.lookback) - 1.0) * 100.0
        return momentum.rename("momentum")

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Return 1 for positive momentum, 0 otherwise, and NaN during warm-up."""
        momentum = self.calculate_momentum(market_data)
        signals = pd.Series(
            np.nan, index=momentum.index, name="signal", dtype="float64"
        )
        available = momentum.notna()
        signals.loc[available] = (momentum.loc[available] > 0).astype("float64")
        return signals

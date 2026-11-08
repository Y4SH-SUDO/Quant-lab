"""Long-only simple moving-average crossover strategy."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.analytics.statistics.returns import rolling_mean
from quantlab.strategies.base import Strategy
from quantlab.strategies.validation import _validated_close


def _validate_window(window: int, name: str) -> None:
    if not isinstance(window, int) or isinstance(window, bool) or window < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class SMACrossoverStrategy(Strategy):
    """Generate long-only signals from fast and slow closing-price SMAs.

    A signal uses the closing price at its own date. Under a normal
    next-session execution assumption, it must therefore be acted on no
    earlier than the following session.
    """

    fast_window: int
    slow_window: int

    def __post_init__(self) -> None:
        _validate_window(self.fast_window, "fast_window")
        _validate_window(self.slow_window, "slow_window")
        if self.fast_window >= self.slow_window:
            raise ValueError("fast_window must be smaller than slow_window")

    def moving_averages(
        self, market_data: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series]:
        """Return right-aligned fast and slow SMAs for validated close prices."""
        close = _validated_close(market_data)
        fast = rolling_mean(close, self.fast_window).rename("fast_sma")
        slow = rolling_mean(close, self.slow_window).rename("slow_sma")
        return fast, slow

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Return 1 for long, 0 for flat, and NaN during SMA warm-up."""
        fast, slow = self.moving_averages(market_data)
        signals = pd.Series(np.nan, index=fast.index, name="signal", dtype="float64")
        available = fast.notna() & slow.notna()
        signals.loc[available] = (fast.loc[available] > slow.loc[available]).astype(
            "float64"
        )
        return signals

"""Long-only rolling Z-score mean-reversion strategy."""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.analytics.statistics.returns import rolling_mean, rolling_volatility
from quantlab.strategies.base import Strategy
from quantlab.strategies.validation import _validated_close


@dataclass(frozen=True)
class MeanReversionStrategy(Strategy):
    """Generate long signals when price is sufficiently below its rolling mean.

    The rolling standard deviation uses the sample convention ``ddof=1``.
    A zero standard deviation produces an unavailable (NaN) Z-score and signal.
    """

    window: int
    entry_threshold: float = 2.0
    exit_threshold: float = 0.0
    ddof: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window, int)
            or isinstance(self.window, bool)
            or self.window < 1
        ):
            raise ValueError("window must be a positive integer")
        if not isinstance(self.ddof, int) or isinstance(self.ddof, bool):
            raise ValueError("ddof must be an integer")
        if self.ddof < 0:
            raise ValueError("ddof must be non-negative")
        if (
            not isinstance(self.entry_threshold, int | float)
            or isinstance(self.entry_threshold, bool)
            or not math.isfinite(float(self.entry_threshold))
            or self.entry_threshold <= 0
        ):
            raise ValueError("entry_threshold must be positive and finite")
        if (
            not isinstance(self.exit_threshold, int | float)
            or isinstance(self.exit_threshold, bool)
            or not math.isfinite(float(self.exit_threshold))
        ):
            raise ValueError("exit_threshold must be finite")
        if self.exit_threshold >= self.entry_threshold:
            raise ValueError("exit_threshold must be smaller than entry_threshold")

    def rolling_statistics(
        self, market_data: pd.DataFrame
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Return rolling mean, sample standard deviation, and Z-score."""
        close = _validated_close(market_data)
        mean = rolling_mean(close, self.window).rename("rolling_mean")
        if self.ddof >= self.window:
            standard_deviation = pd.Series(
                np.nan, index=close.index, name="rolling_std", dtype="float64"
            )
        else:
            standard_deviation = rolling_volatility(
                close, self.window, ddof=self.ddof
            ).rename("rolling_std")
        z_score = ((close - mean) / standard_deviation).where(
            standard_deviation != 0
        )
        return mean, standard_deviation, z_score.rename("z_score")

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Return long/flat state with NaN during warm-up or zero dispersion."""
        _, _, z_score = self.rolling_statistics(market_data)
        signals = pd.Series(np.nan, index=z_score.index, name="signal", dtype="float64")
        position = 0.0
        for date, value in z_score.items():
            if pd.isna(value):
                continue
            if value <= -self.entry_threshold:
                position = 1.0
            elif value >= -self.exit_threshold:
                position = 0.0
            signals.loc[date] = position
        return signals

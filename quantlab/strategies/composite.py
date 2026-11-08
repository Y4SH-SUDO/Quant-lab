"""Weighted composite strategy signals."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy


@dataclass(frozen=True)
class CompositeStrategy(Strategy):
    """Combine component target positions with a weighted voting threshold."""

    strategies: Sequence[Strategy]
    weights: Sequence[float] | None = None
    threshold: float = 0.5

    def __post_init__(self) -> None:
        components = tuple(self.strategies)
        if len(components) < 2:
            raise ValueError("at least two strategies are required")
        if not all(isinstance(strategy, Strategy) for strategy in components):
            raise ValueError("all components must implement Strategy")
        if self.weights is None:
            validated_weights = tuple(1.0 for _ in components)
        else:
            validated_weights = tuple(self.weights)
            if len(validated_weights) != len(components):
                raise ValueError("weights must match the number of strategies")
            if any(
                not isinstance(weight, int | float)
                or isinstance(weight, bool)
                or not math.isfinite(float(weight))
                or weight <= 0
                for weight in validated_weights
            ):
                raise ValueError("weights must be finite and strictly positive")
        if (
            not isinstance(self.threshold, int | float)
            or isinstance(self.threshold, bool)
            or not math.isfinite(float(self.threshold))
            or not 0 <= self.threshold <= 1
        ):
            raise ValueError("threshold must be finite and between 0 and 1")
        object.__setattr__(self, "strategies", components)
        object.__setattr__(self, "weights", validated_weights)

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        """Return weighted long/flat votes, or NaN if any component is unavailable."""
        component_signals = [
            strategy.generate_signals(market_data) for strategy in self.strategies
        ]
        if any(
            not isinstance(signal, pd.Series)
            or not signal.index.equals(market_data.index)
            for signal in component_signals
        ):
            raise ValueError("component signals must be aligned with market_data")
        for signal in component_signals:
            if not pd.api.types.is_numeric_dtype(signal):
                raise ValueError("component signals must be numeric")
            values = signal.to_numpy(dtype="float64", na_value=np.nan)
            if not np.isfinite(values[~np.isnan(values)]).all() or not np.isin(
                values[~np.isnan(values)], [0.0, 1.0]
            ).all():
                raise ValueError("component signals must contain only 0, 1, or NaN")
        frame = pd.concat(component_signals, axis=1)
        available = frame.notna().all(axis=1)
        weights = np.asarray(self.weights, dtype="float64")
        score = frame.mul(weights, axis=1).sum(axis=1, min_count=len(weights))
        score = score / weights.sum()
        result = pd.Series(
            np.nan, index=market_data.index, name="signal", dtype="float64"
        )
        result.loc[available] = (score.loc[available] >= self.threshold).astype(
            "float64"
        )
        return result

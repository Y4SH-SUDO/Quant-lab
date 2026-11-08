"""Deterministic tests for M3 signal-generating strategies."""

import numpy as np
import pandas as pd
import pytest

from quantlab.strategies import (
    CompositeStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    SMACrossoverStrategy,
    Strategy,
)


def market_data(prices: list[float], dates: list[str] | None = None) -> pd.DataFrame:
    index = pd.to_datetime(
        dates or pd.date_range("2024-01-01", periods=len(prices), freq="D")
    )
    return pd.DataFrame({"Close": prices}, index=index)


def test_sma_values_and_hand_calculated_signals() -> None:
    data = market_data([3, 2, 4, 6, 5])
    strategy = SMACrossoverStrategy(fast_window=2, slow_window=3)

    fast, slow = strategy.moving_averages(data)
    pd.testing.assert_series_equal(
        fast, pd.Series([np.nan, 2.5, 3.0, 5.0, 5.5], index=data.index, name="fast_sma")
    )
    pd.testing.assert_series_equal(
        slow,
        pd.Series(
            [np.nan, np.nan, 3.0, 4.0, 5.0], index=data.index, name="slow_sma"
        ),
    )
    pd.testing.assert_series_equal(
        strategy.generate_signals(data),
        pd.Series(
            [np.nan, np.nan, 0.0, 1.0, 1.0],
            index=data.index,
            name="signal",
        ),
    )


def test_long_and_flat_signals_and_warmup() -> None:
    data = market_data([5, 4, 3, 2, 1, 4, 5])
    signals = SMACrossoverStrategy(2, 3).generate_signals(data)
    assert signals.iloc[:2].isna().all()
    assert signals.iloc[2:].tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_signal_dates_are_aligned_with_original_input() -> None:
    data = market_data([1, 2, 3], ["2024-03-01", "2024-03-04", "2024-03-07"])
    signals = SMACrossoverStrategy(1, 2).generate_signals(data)
    assert signals.index.equals(data.index)


@pytest.mark.parametrize(
    ("fast", "slow"),
    [(0, 3), (-1, 3), (1, 0), (1, -2), (1.5, 3), (True, 3), (1, True), (3, 3), (4, 3)],
)
def test_invalid_windows(fast: int, slow: int) -> None:
    with pytest.raises(ValueError):
        SMACrossoverStrategy(fast, slow)


def test_strategy_implements_reusable_interface() -> None:
    assert isinstance(SMACrossoverStrategy(2, 3), Strategy)


def test_missing_close_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="Close"):
        SMACrossoverStrategy(2, 3).generate_signals(
            pd.DataFrame(
                {"Open": [1.0, 2.0]},
                index=pd.date_range("2024-01-01", periods=2),
            )
        )


@pytest.mark.parametrize("bad_price", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_missing_infinite_and_non_positive_prices_are_rejected(
    bad_price: float,
) -> None:
    with pytest.raises(ValueError):
        SMACrossoverStrategy(1, 2).generate_signals(
            market_data([1.0, bad_price, 2.0])
        )


@pytest.mark.parametrize(
    "dates",
    [
        ["2024-01-02", "2024-01-01", "2024-01-03"],
        ["2024-01-01", "2024-01-01", "2024-01-02"],
        ["2024-01-01", "not-a-date", "2024-01-03"],
    ],
)
def test_invalid_dates_are_rejected(dates: list[str]) -> None:
    with pytest.raises(ValueError):
        SMACrossoverStrategy(1, 2).generate_signals(market_data([1, 2, 3], dates))


def test_repeated_runs_are_deterministic() -> None:
    data = market_data([2, 4, 1, 5, 3])
    strategy = SMACrossoverStrategy(2, 3)
    pd.testing.assert_series_equal(
        strategy.generate_signals(data), strategy.generate_signals(data)
    )


def test_future_price_perturbation_does_not_change_prior_signals() -> None:
    original = market_data([2, 4, 3, 5, 4, 6])
    perturbed = original.copy()
    perturbed.iloc[-1, 0] = 1000.0
    strategy = SMACrossoverStrategy(2, 3)
    original_signals = strategy.generate_signals(original)
    perturbed_signals = strategy.generate_signals(perturbed)
    pd.testing.assert_series_equal(
        original_signals.iloc[:-1], perturbed_signals.iloc[:-1]
    )


def test_momentum_hand_calculation_and_signals() -> None:
    data = market_data([100.0, 110.0, 99.0, 121.0, 121.0])
    strategy = MomentumStrategy(lookback=2)

    pd.testing.assert_series_equal(
        strategy.calculate_momentum(data),
        pd.Series(
            [np.nan, np.nan, -1.0, 10.0, 22.2222222222],
            index=data.index,
            name="momentum",
        ),
        check_exact=False,
        rtol=1e-10,
    )
    pd.testing.assert_series_equal(
        strategy.generate_signals(data),
        pd.Series(
            [np.nan, np.nan, 0.0, 1.0, 1.0],
            index=data.index,
            name="signal",
        ),
    )


def test_momentum_zero_is_flat_and_warmup_uses_lookback() -> None:
    data = market_data([10.0, 12.0, 10.0, 12.0, 10.0])
    signals = MomentumStrategy(2).generate_signals(data)
    assert signals.iloc[:2].isna().all()
    assert signals.iloc[2:].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("lookback", [0, -1, 1.5, True, False])
def test_invalid_momentum_lookback_is_rejected(lookback: int) -> None:
    with pytest.raises(ValueError, match="lookback"):
        MomentumStrategy(lookback)


def test_momentum_implements_strategy_interface_and_preserves_dates() -> None:
    data = market_data([10.0, 11.0], ["2024-03-01", "2024-03-04"])
    strategy = MomentumStrategy(1)
    assert isinstance(strategy, Strategy)
    assert strategy.generate_signals(data).index.equals(data.index)


@pytest.mark.parametrize(
    "dates",
    [
        ["2024-01-02", "2024-01-01", "2024-01-03"],
        ["2024-01-01", "2024-01-01", "2024-01-02"],
        ["2024-01-01", "not-a-date", "2024-01-03"],
    ],
)
def test_momentum_rejects_invalid_dates(dates: list[str]) -> None:
    with pytest.raises(ValueError):
        MomentumStrategy(1).generate_signals(market_data([1.0, 2.0, 3.0], dates))


def test_momentum_future_price_perturbation_does_not_change_prior_signals() -> None:
    original = market_data([10.0, 12.0, 11.0, 13.0, 12.0])
    perturbed = original.copy()
    perturbed.iloc[-1, 0] = 1000.0
    strategy = MomentumStrategy(2)
    pd.testing.assert_series_equal(
        strategy.generate_signals(original).iloc[:-1],
        strategy.generate_signals(perturbed).iloc[:-1],
    )


def test_mean_reversion_indicators_use_sample_standard_deviation() -> None:
    data = market_data([10.0, 12.0, 14.0, 12.0])
    strategy = MeanReversionStrategy(window=3)
    mean, standard_deviation, z_score = strategy.rolling_statistics(data)
    expected_std = 2.0
    assert mean.iloc[2] == pytest.approx(12.0)
    assert standard_deviation.iloc[2] == pytest.approx(expected_std)
    assert z_score.iloc[2] == pytest.approx(2 / expected_std)
    assert mean.iloc[:2].isna().all()
    assert z_score.iloc[:2].isna().all()


def test_mean_reversion_entry_exit_and_position_retention() -> None:
    data = market_data([10.0, 11.0, 12.0, 8.0, 9.0, 10.0])
    strategy = MeanReversionStrategy(window=3, entry_threshold=1.0)
    signals = strategy.generate_signals(data)
    assert signals.iloc[:2].isna().all()
    assert signals.iloc[2:].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_mean_reversion_zero_standard_deviation_is_unavailable() -> None:
    data = market_data([10.0, 10.0, 10.0, 10.0])
    strategy = MeanReversionStrategy(window=3)
    _, standard_deviation, z_score = strategy.rolling_statistics(data)
    assert (standard_deviation.iloc[2:] == 0).all()
    assert z_score.isna().all()
    assert strategy.generate_signals(data).isna().all()


@pytest.mark.parametrize(
    "arguments",
    [
        {"window": 0},
        {"window": 3, "entry_threshold": 0},
        {"window": 3, "entry_threshold": np.inf},
        {"window": 3, "exit_threshold": np.inf},
        {"window": 3, "entry_threshold": 1, "exit_threshold": 1},
    ],
)
def test_mean_reversion_rejects_invalid_configuration(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        MeanReversionStrategy(**arguments)


class FixedStrategy(Strategy):
    def __init__(self, signals: list[object]) -> None:
        self.signals = signals

    def generate_signals(self, market_data: pd.DataFrame) -> pd.Series:
        return pd.Series(self.signals, index=market_data.index)


def test_composite_equal_and_unequal_weight_votes() -> None:
    data = market_data([1.0, 2.0, 3.0])
    first = FixedStrategy([1.0, 0.0, 1.0])
    second = FixedStrategy([0.0, 1.0, 1.0])
    assert CompositeStrategy([first, second]).generate_signals(data).tolist() == [
        1.0,
        1.0,
        1.0,
    ]
    assert CompositeStrategy([first, second], [2.0, 1.0]).generate_signals(
        data
    ).tolist() == [1.0, 0.0, 1.0]


def test_composite_threshold_boundary_and_missing_signals() -> None:
    data = market_data([1.0, 2.0, 3.0])
    first = FixedStrategy([1.0, np.nan, 0.0])
    second = FixedStrategy([0.0, 1.0, 1.0])
    result = CompositeStrategy([first, second], threshold=0.5).generate_signals(data)
    assert result.iloc[0] == 1.0
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 1.0


@pytest.mark.parametrize("invalid_signal", [[0.4, 1.0], [0.0, 0.7], [np.inf, 0.0]])
def test_composite_rejects_non_binary_or_infinite_signals(
    invalid_signal: list[object],
) -> None:
    data = market_data([1.0, 2.0])
    with pytest.raises(ValueError, match="component signals"):
        CompositeStrategy(
            [FixedStrategy(invalid_signal), FixedStrategy([0.0, 1.0])]
        ).generate_signals(data)


def test_composite_rejects_non_numeric_signals() -> None:
    data = market_data([1.0, 2.0])
    with pytest.raises(ValueError, match="numeric"):
        CompositeStrategy(
            [FixedStrategy(["long", 0.0]), FixedStrategy([0.0, 1.0])]
        ).generate_signals(data)


def test_composite_accepts_binary_signals_and_propagates_nan() -> None:
    data = market_data([1.0, 2.0, 3.0])
    result = CompositeStrategy(
        [FixedStrategy([1.0, 0.0, np.nan]), FixedStrategy([1.0, 1.0, 1.0])]
    ).generate_signals(data)
    assert result.iloc[0] == 1.0
    assert result.iloc[1] == 1.0
    assert pd.isna(result.iloc[2])


@pytest.mark.parametrize(
    "arguments",
    [
        {"strategies": [FixedStrategy([1.0])]},
        {"strategies": [FixedStrategy([1.0]), object()]},
        {"strategies": [FixedStrategy([1.0]), FixedStrategy([1.0])], "weights": [1.0]},
        {
            "strategies": [FixedStrategy([1.0]), FixedStrategy([1.0])],
            "weights": [0.0, 1.0],
        },
        {
            "strategies": [FixedStrategy([1.0]), FixedStrategy([1.0])],
            "threshold": 1.1,
        },
    ],
)
def test_composite_rejects_invalid_configuration(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        CompositeStrategy(**arguments)


def test_composite_preserves_alignment_and_does_not_mutate_inputs() -> None:
    data = market_data([1.0, 2.0])
    first = FixedStrategy([1.0, 0.0])
    second = FixedStrategy([0.0, 1.0])
    original = data.copy()
    result = CompositeStrategy([first, second]).generate_signals(data)
    assert result.index.equals(data.index)
    pd.testing.assert_frame_equal(data, original)

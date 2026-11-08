# M3 Trading Strategy Development

M3 currently provides signal generation only. It does not execute trades,
backtest a strategy, allocate a portfolio, size positions, or establish
profitability.

## SMA crossover

`SMACrossoverStrategy` is a long-only strategy using unadjusted `Close`
prices. For fast window `nf` and slow window `ns`, where `nf < ns`:

```
F(t) = (1 / nf) * sum(P(t-i), i=0..nf-1)
S(t) = (1 / ns) * sum(P(t-i), i=0..ns-1)
q(t) = 1 when F(t) > S(t)
q(t) = 0 when F(t) <= S(t)
```

`1` means long and `0` means flat. Both averages are right-aligned and require
their complete windows, so the first `ns - 1` signals are `NaN` warm-up
values. Signals retain the original date index and never use future prices.

The input must be a date-indexed, sorted, unique `DataFrame` containing finite,
strictly positive `Close` prices. A signal uses the current session's close;
under a normal next-session execution assumption it must not be treated as
executable at that same close.

The strategy uses unadjusted closes, so splits and distributions can create
price discontinuities that affect signals. Corporate-action-aware research
should explicitly select and validate an adjusted series before strategy use.
Signal generation alone says nothing about profitability: execution costs,
slippage, liquidity, risk, and out-of-sample performance require later
milestones.

## Momentum

`MomentumStrategy` is a long-only strategy using the percentage change in the
unadjusted `Close` price over a configurable positive integer lookback `n`:

```
M(t) = ((P(t) / P(t-n)) - 1) * 100
q(t) = 1 when M(t) > 0
q(t) = 0 when M(t) <= 0
```

The first `n` signals are `NaN` because no prior comparison price exists.
Momentum signals use only the current and historical closes, retain the input
dates, and should be executed no earlier than the next session under a normal
next-session assumption.

## Mean reversion

`MeanReversionStrategy` compares the current close with a rolling mean using a
sample standard deviation (`ddof=1` by default):

```
Z(t) = (P(t) - mean(t)) / std(t)
```

The first `window - 1` observations are warm-up values. A long position is
entered when `Z(t) <= -entry_threshold`, exited when `Z(t) >= -exit_threshold`,
and otherwise retains its previous position. It begins flat when the
indicator first becomes available. A zero standard deviation has no
well-defined Z-score, so both the Z-score and signal are `NaN` rather than an
infinite or fabricated signal.

Unlike trend-following SMA crossover and momentum strategies, mean reversion
expects sufficiently negative deviations from a recent norm to move back
toward that norm. It is sensitive to regime changes, window choice, thresholds,
and unadjusted corporate-action discontinuities.

## Composite strategy

`CompositeStrategy` combines at least two strategies through positive weights:

```
score(t) = sum(weight_i * signal_i(t)) / sum(weight_i)
```

The composite is long when the score is at least the threshold (default `0.5`)
and flat below it. Any unavailable component signal produces an unavailable
composite signal. This is a deterministic weighted voting mechanism, not a
machine-learning model and not a guarantee of improved performance.

All strategies describe a desired target position. A signal based on a
session's closing price should be acted on no earlier than the following
session under a normal next-session execution assumption. None of these
strategies executes trades or establishes profitability.

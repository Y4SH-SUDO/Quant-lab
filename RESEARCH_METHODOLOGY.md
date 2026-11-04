# Research Methodology

All future research must follow these principles:

1. **No look-ahead bias:** features, labels, and decisions may use only data
   available at the decision timestamp.
2. **No survivorship bias:** universes must account for delisted and changed
   securities where the data source permits.
3. **Out-of-sample validation:** separate development, validation, and
   out-of-sample periods; do not tune on the final evaluation period.
4. **Realistic transaction costs:** include brokerage, taxes, fees, slippage,
   liquidity limits, and market-impact assumptions when applicable.
5. **Reproducible experiments:** record inputs, code version, parameters,
   random seeds, and outputs.
6. **Statistical evaluation:** assess uncertainty, robustness, and multiple
   comparisons rather than relying on anecdotal winning examples.

These are research requirements, not claims that the corresponding engines are
implemented in M0.

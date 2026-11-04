"""Small, deterministic daily bars for provider-independent tests."""

import pandas as pd


def daily_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "High": [102.0, 103.0, 104.0],
            "Low": [99.0, 100.0, 101.0],
            "Close": [101.0, 102.0, 103.0],
            "Adj Close": [101.0, 102.0, 103.0],
            "Volume": [1000, 1100, 1200],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
    )

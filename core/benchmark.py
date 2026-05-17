"""Default benchmark strategies the user's strategy is compared against.

Each benchmark is just a position generator with the same contract as a
generated strategy, so it runs through the *same* engine and pays the *same*
costs — a fair, apples-to-apples comparison rather than a costless ideal.
"""
from __future__ import annotations

import pandas as pd

from .indicators import sma


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Fully invested the entire period (the classic benchmark)."""
    return pd.Series(1.0, index=df.index)


def sma_200_trend(df: pd.DataFrame) -> pd.Series:
    """Long while close is above its 200-day SMA, else flat (trend filter)."""
    ma = sma(df["close"], 200)
    return (df["close"] > ma).astype("float64").fillna(0.0)


def half_invested(df: pd.DataFrame) -> pd.Series:
    """Constant 50% exposure — a simple lower-risk passive baseline."""
    return pd.Series(0.5, index=df.index)


BENCHMARKS: dict[str, callable] = {
    "Buy & Hold": buy_and_hold,
    "SMA 200 Trend": sma_200_trend,
    "50% Invested": half_invested,
}

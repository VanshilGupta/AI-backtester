"""Vectorized technical-indicator helpers exposed to generated strategy code.

Every function takes and returns pandas Series aligned to the input index, so the
generated `generate_signals` function can compose them without lookahead bugs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average over `n` periods."""
    return series.rolling(int(n), min_periods=int(n)).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with span `n`."""
    return series.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def wma(series: pd.Series, n: int) -> pd.Series:
    """Linearly weighted moving average over `n` periods."""
    n = int(n)
    weights = np.arange(1, n + 1)
    return series.rolling(n, min_periods=n).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def stdev(series: pd.Series, n: int) -> pd.Series:
    """Rolling standard deviation over `n` periods."""
    return series.rolling(int(n), min_periods=int(n)).std()


def zscore(series: pd.Series, n: int) -> pd.Series:
    """Rolling z-score of `series` over `n` periods."""
    n = int(n)
    mean = series.rolling(n, min_periods=n).mean()
    sd = series.rolling(n, min_periods=n).std()
    return (series - mean) / sd


def roc(series: pd.Series, n: int) -> pd.Series:
    """Rate of change (fractional) over `n` periods."""
    return series.pct_change(int(n))


def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing), range 0-100."""
    n = int(n)
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD. Returns (macd_line, signal_line, histogram)."""
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = macd_line.ewm(span=int(signal), adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger(
    series: pd.Series, n: int = 20, k: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands. Returns (middle, upper, lower)."""
    mid = sma(series, n)
    sd = stdev(series, n)
    return mid, mid + k * sd, mid - k * sd


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Average True Range (Wilder's smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / int(n), adjust=False, min_periods=int(n)).mean()


def rolling_high(series: pd.Series, n: int) -> pd.Series:
    """Highest value over the trailing `n` periods (inclusive)."""
    return series.rolling(int(n), min_periods=int(n)).max()


def rolling_low(series: pd.Series, n: int) -> pd.Series:
    """Lowest value over the trailing `n` periods (inclusive)."""
    return series.rolling(int(n), min_periods=int(n)).min()


def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where `a` crosses from below to above `b`."""
    a_prev, b_prev = a.shift(1), b.shift(1)
    return (a > b) & (a_prev <= b_prev)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True on bars where `a` crosses from above to below `b`."""
    a_prev, b_prev = a.shift(1), b.shift(1)
    return (a < b) & (a_prev >= b_prev)


def slope(series: pd.Series, n: int) -> pd.Series:
    """Per-bar linear-regression slope over a trailing window of `n`."""
    n = int(n)
    x = np.arange(n)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return float(((x - x_mean) * (y - y.mean())).sum() / denom)

    return series.rolling(n, min_periods=n).apply(_slope, raw=True)


# Names injected into the generated-strategy execution namespace.
INDICATOR_NAMESPACE = {
    "sma": sma,
    "ema": ema,
    "wma": wma,
    "stdev": stdev,
    "zscore": zscore,
    "roc": roc,
    "rsi": rsi,
    "macd": macd,
    "bollinger": bollinger,
    "atr": atr,
    "rolling_high": rolling_high,
    "rolling_low": rolling_low,
    "crossover": crossover,
    "crossunder": crossunder,
    "slope": slope,
}

# Human-readable signatures, embedded in the (cached) system prompt.
INDICATOR_DOCS = """\
sma(series, n)                      -> Series   Simple moving average
ema(series, n)                      -> Series   Exponential moving average
wma(series, n)                      -> Series   Linearly weighted moving average
stdev(series, n)                    -> Series   Rolling standard deviation
zscore(series, n)                   -> Series   Rolling z-score
roc(series, n)                      -> Series   Rate of change (fractional)
rsi(series, n=14)                   -> Series   Relative Strength Index (0-100)
macd(series, fast=12, slow=26, signal=9) -> (macd_line, signal_line, hist)
bollinger(series, n=20, k=2.0)      -> (mid, upper, lower)
atr(high, low, close, n=14)         -> Series   Average True Range
rolling_high(series, n)             -> Series   Trailing max
rolling_low(series, n)              -> Series   Trailing min
crossover(a, b)                     -> Series[bool]  a crosses above b
crossunder(a, b)                    -> Series[bool]  a crosses below b
slope(series, n)                    -> Series   Trailing linear-regression slope
"""

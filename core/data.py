"""OHLCV loading: from an uploaded CSV, or fetched live by ticker + timeline."""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

_COLUMN_ALIASES = {
    "date": ["date", "datetime", "timestamp", "time", "dt"],
    "open": ["open", "o", "open_price"],
    "high": ["high", "h", "high_price"],
    "low": ["low", "l", "low_price"],
    "close": ["close", "c", "close_price", "adj close", "adj_close", "adjclose", "price"],
    "volume": ["volume", "vol", "v", "qty"],
}

MIN_ROWS = 30


def _resolve(columns: list[str], wanted: str) -> str | None:
    lower = {str(c).lower().strip(): c for c in columns}
    for alias in _COLUMN_ALIASES[wanted]:
        if alias in lower:
            return lower[alias]
    return None


def _finalise(out: pd.DataFrame) -> pd.DataFrame:
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out = out.sort_values("date").drop_duplicates(subset="date")
    out = out.set_index("date")
    out["volume"] = out["volume"].fillna(0.0)
    if len(out) < MIN_ROWS:
        raise ValueError(
            f"Only {len(out)} valid rows after cleaning; need at least "
            f"{MIN_ROWS} to backtest."
        )
    return out


def load_ohlcv(source: str | io.BytesIO | io.StringIO) -> pd.DataFrame:
    """Read a CSV and return a clean DataFrame indexed by date with OHLCV columns.

    `volume` is synthesised as 0 if absent. Raises ValueError on missing essentials.
    """
    df = pd.read_csv(source)
    cols = list(df.columns)

    date_col = _resolve(cols, "date")
    if date_col is None:
        raise ValueError("Could not find a date/timestamp column in the CSV.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=False)
    for field in ("open", "high", "low", "close"):
        src = _resolve(cols, field)
        if src is None:
            raise ValueError(f"Missing required '{field}' column in the CSV.")
        out[field] = pd.to_numeric(df[src], errors="coerce")

    vol_col = _resolve(cols, "volume")
    out["volume"] = pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0
    return _finalise(out)


def fetch_ohlcv(
    ticker: str,
    start: str | date | None = None,
    end: str | date | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV for `ticker` over a timeline via Yahoo Finance.

    Ticker conventions: US symbols plain (AAPL, SPY); NSE/BSE with a suffix
    (RELIANCE.NS, INFY.NS, 500325.BO); indices with a caret (^NSEI, ^GSPC);
    crypto as PAIR-USD (BTC-USD). `interval` is one of yfinance's intervals
    (1d, 1h, 1wk, ...). Raises ValueError with an actionable message on failure.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError(
            "Ticker fetch needs the 'yfinance' package. Install with: "
            "python -m pip install yfinance"
        ) from exc

    sym = ticker.strip()
    if not sym:
        raise ValueError("Enter a ticker symbol.")

    try:
        raw = yf.download(
            sym,
            start=str(start) if start else None,
            end=str(end) if end else None,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # network / yfinance internals
        raise ValueError(f"Download failed for '{sym}': {exc}") from exc

    if raw is None or raw.empty:
        raise ValueError(
            f"No data returned for '{sym}'. Check the symbol (NSE needs a "
            f".NS suffix, e.g. RELIANCE.NS) and the date range."
        )

    # yfinance returns a MultiIndex column frame for single tickers too.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw.index, errors="coerce")
    for field in ("open", "high", "low", "close"):
        if field not in raw.columns:
            raise ValueError(f"Provider response missing '{field}' for '{sym}'.")
        out[field] = pd.to_numeric(raw[field].to_numpy().ravel(), errors="coerce")
    out["volume"] = (
        pd.to_numeric(raw["volume"].to_numpy().ravel(), errors="coerce")
        if "volume" in raw.columns
        else 0.0
    )
    return _finalise(out)

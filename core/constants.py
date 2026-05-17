"""Single source of truth for trading costs, charges and engine defaults.

Everything a user is likely to want to tweak lives here. The values are also
surfaced (and editable) in the app sidebar — these are just the defaults the
sidebar is seeded with.

Cost model
----------
All charges are expressed in **basis points (bps) of traded notional**, where
1 bps = 0.01% = 0.0001. The backtest charges costs on *turnover*: the fraction
of capital whose position changed on a bar. A full entry then a full exit is
two units of turnover, so a per-side bps value is charged twice over a round
trip — which is the correct, conservative treatment.

The defaults below are a realistic Indian-equity *delivery* profile (NSE, 2024-25
schedule). They are deliberately on the conservative side: in backtesting it is
far safer to over-estimate frictions than to discover them live. Change them
freely for intraday, F&O, US equities, crypto, etc.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostModel:
    """Per-side transaction costs, in basis points of traded notional.

    `gst_pct` is a percentage (18 = 18%) applied to the statutory-fee subset
    (brokerage + exchange + SEBI), mirroring how Indian brokers bill GST.
    """

    brokerage_bps: float = 3.0        # broker fee per side
    stt_bps: float = 10.0             # securities transaction tax (delivery ~0.1%/side)
    exchange_txn_bps: float = 0.32    # exchange transaction charge (NSE ~0.00322%)
    sebi_bps: float = 0.01            # SEBI turnover fee (~0.0001%)
    stamp_duty_bps: float = 1.5       # stamp duty (buy side ~0.015%)
    gst_pct: float = 18.0             # GST on (brokerage + exchange + SEBI)
    slippage_bps: float = 2.0         # modelled execution slippage per side

    def gst_bps(self) -> float:
        base = self.brokerage_bps + self.exchange_txn_bps + self.sebi_bps
        return base * self.gst_pct / 100.0

    def effective_bps_per_turnover(self) -> float:
        """Total cost charged on one unit of turnover (one side), in bps."""
        return (
            self.brokerage_bps
            + self.stt_bps
            + self.exchange_txn_bps
            + self.sebi_bps
            + self.stamp_duty_bps
            + self.gst_bps()
            + self.slippage_bps
        )

    def cost_rate(self) -> float:
        """Effective per-turnover cost as a fraction (what the engine multiplies)."""
        return self.effective_bps_per_turnover() / 1e4

    def round_trip_pct(self) -> float:
        """Approx cost of a full entry+exit, as a percent of notional."""
        return 2.0 * self.effective_bps_per_turnover() / 100.0

    def breakdown(self) -> dict[str, float]:
        """Per-side cost components in bps, for transparent display."""
        return {
            "Brokerage": self.brokerage_bps,
            "STT / CTT": self.stt_bps,
            "Exchange txn": self.exchange_txn_bps,
            "SEBI": self.sebi_bps,
            "Stamp duty": self.stamp_duty_bps,
            "GST": round(self.gst_bps(), 4),
            "Slippage": self.slippage_bps,
            "Total / side": round(self.effective_bps_per_turnover(), 4),
        }


@dataclass
class EngineDefaults:
    initial_capital: float = 100_000.0
    execution_lag: int = 1            # bars between signal and fill (1 => no lookahead)
    risk_free_rate: float = 0.06      # annualised, for Sharpe/Sortino excess return
    cost_model: CostModel = field(default_factory=CostModel)


DEFAULTS = EngineDefaults()

# Default benchmark the strategy is judged against. Keys map to
# core.benchmark.BENCHMARKS.
DEFAULT_BENCHMARK = "Buy & Hold"

# Ready-made starters surfaced in the UI. Each is
# (label, prompt, data_hint, code). `code` is a working `generate_signals`
# implementation so the "Paste my own code" path can auto-fill it. The code
# only uses the built-in indicator helpers + pd/np and passes the sandbox.
EXAMPLE_STRATEGIES: list[tuple[str, str, str, str]] = [
    (
        "SMA golden cross (trend following)",
        "Go long when the 50-day SMA crosses above the 200-day SMA and stay "
        "long until the 50-day SMA crosses back below the 200-day SMA. "
        "Long only, fully invested when in the trade.",
        "Daily OHLCV of a trending index/large-cap, 8+ years "
        "(e.g. NIFTY 50, RELIANCE.NS, SPY, AAPL).",
        '''def generate_signals(df):
    fast = sma(df['close'], 50)
    slow = sma(df['close'], 200)
    pos = pd.Series(0.0, index=df.index)
    pos[fast > slow] = 1.0
    return pos.fillna(0.0)''',
    ),
    (
        "RSI(2) mean reversion",
        "On daily bars, go long when RSI(2) closes below 10 while price is "
        "above its 200-day SMA (long-term uptrend filter); exit when RSI(2) "
        "closes above 60 or price closes below the 200-day SMA. Long only.",
        "Daily OHLCV of a liquid large-cap or index ETF, 10+ years "
        "(e.g. SPY, QQQ, NIFTYBEES.NS).",
        '''def generate_signals(df):
    r = rsi(df['close'], 2)
    uptrend = df['close'] > sma(df['close'], 200)
    enter = (r < 10) & uptrend
    leave = (r > 60) | (~uptrend)
    pos = pd.Series(np.nan, index=df.index)
    pos[enter] = 1.0
    pos[leave] = 0.0
    return pos.ffill().fillna(0.0).clip(0.0, 1.0)''',
    ),
    (
        "Bollinger Band breakout",
        "Go long when close breaks above the upper Bollinger Band (20, 2.0) "
        "and exit when close falls back below the 20-period middle band. "
        "Long only; size 1.0 when in the trade.",
        "Daily or hourly OHLCV of a volatile, trending instrument, 5+ years "
        "(e.g. BTC-USD, TSLA, ADANIENT.NS).",
        '''def generate_signals(df):
    mid, upper, lower = bollinger(df['close'], 20, 2.0)
    enter = df['close'] > upper
    leave = df['close'] < mid
    pos = pd.Series(np.nan, index=df.index)
    pos[enter] = 1.0
    pos[leave] = 0.0
    return pos.ffill().fillna(0.0).clip(0.0, 1.0)''',
    ),
    (
        "Dual-momentum 12-1",
        "Compute 12-month price momentum skipping the most recent month. "
        "Go fully long when that momentum is positive, otherwise stay flat. "
        "Long only, evaluated on daily data.",
        "Daily OHLCV of a broad index over 15+ years "
        "(e.g. ^GSPC / SPY, ^NSEI / NIFTYBEES.NS).",
        '''def generate_signals(df):
    c = df['close']
    # price 1 month ago vs 12 months ago (~21 / ~252 trading days)
    momentum = c.shift(21) / c.shift(252) - 1.0
    pos = pd.Series(0.0, index=df.index)
    pos[momentum > 0] = 1.0
    return pos.fillna(0.0)''',
    ),
]

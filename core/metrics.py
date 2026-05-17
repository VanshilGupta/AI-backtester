"""Performance metrics derived from a BacktestResult."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .constants import DEFAULTS
from .engine import BacktestResult


@dataclass
class Metrics:
    total_return: float
    cagr: float
    benchmark_cagr: float
    excess_cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    exposure: float
    best_trade: float
    worst_trade: float

    def as_dict(self) -> dict:
        return asdict(self)


def _years(result: BacktestResult) -> float:
    idx = result.equity.index
    days = (idx[-1] - idx[0]).days
    return max(days / 365.25, 1e-9)


def compute_metrics(
    result: BacktestResult,
    risk_free_rate: float = DEFAULTS.risk_free_rate,
) -> Metrics:
    """Annualised performance metrics. Sharpe/Sortino use returns in excess of
    the per-period risk-free rate (best practice, not the rf=0 shortcut)."""
    equity = result.equity
    net = result.net_returns
    years = _years(result)
    ppy = result.periods_per_year
    ann_factor = np.sqrt(ppy)
    rf_period = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0
    excess = net - rf_period

    start, end = float(equity.iloc[0]), float(equity.iloc[-1])
    total_return = end / start - 1.0
    cagr = (end / start) ** (1.0 / years) - 1.0 if end > 0 else -1.0

    bench = result.benchmark_equity
    bench_cagr = (
        (float(bench.iloc[-1]) / float(bench.iloc[0])) ** (1.0 / years) - 1.0
        if float(bench.iloc[-1]) > 0
        else -1.0
    )

    std = float(net.std())
    volatility = std * ann_factor
    mean_excess = float(excess.mean())
    # Sharpe denominator is the stdev of excess returns; subtracting the
    # constant rf_period doesn't change the stdev, so std(net) == std(excess).
    sharpe = (mean_excess / std * ann_factor) if std > 0 else 0.0

    # Sortino uses the *downside deviation*: RMS of the shortfall below the
    # target (here the risk-free rate, since `excess` is already net of it),
    # averaged over ALL periods — not the sample stdev of only-negative
    # returns. This is the textbook definition (Sortino & Price, 1994).
    shortfall = np.minimum(excess.to_numpy(dtype="float64"), 0.0)
    downside_dev = float(np.sqrt(np.mean(shortfall ** 2)))
    sortino = (
        mean_excess / downside_dev * ann_factor if downside_dev > 0 else 0.0
    )

    max_dd = float(result.drawdown.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else 0.0

    trades = result.trades
    if len(trades):
        rets = trades["return"]
        wins = rets[rets > 0]
        losses = rets[rets < 0]
        win_rate = len(wins) / len(rets)
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        profit_factor = (
            gross_profit / gross_loss
            if gross_loss > 0
            else (float("inf") if gross_profit > 0 else 0.0)
        )
        avg_trade = float(rets.mean())
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        best_trade = float(rets.max())
        worst_trade = float(rets.min())
    else:
        win_rate = profit_factor = avg_trade = avg_win = avg_loss = 0.0
        best_trade = worst_trade = 0.0

    exposure = float((result.positions != 0).mean())

    return Metrics(
        total_return=total_return,
        cagr=cagr,
        benchmark_cagr=bench_cagr,
        excess_cagr=cagr - bench_cagr,
        volatility=volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        calmar=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=int(len(trades)),
        avg_trade_return=avg_trade,
        avg_win=avg_win,
        avg_loss=avg_loss,
        exposure=exposure,
        best_trade=best_trade,
        worst_trade=worst_trade,
    )

"""Vectorized backtest engine.

Execution model:
  * `generate_signals` returns the TARGET position for each bar using data up to
    that bar's close.
  * The engine applies a one-bar execution lag (you can only act on the *next*
    bar), so there is no lookahead even if the strategy reads the current close.
  * Costs come from the configurable `CostModel` (brokerage, STT, exchange,
    SEBI, stamp duty, GST, slippage) and are charged on turnover whenever the
    held position changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .constants import DEFAULTS, CostModel


@dataclass
class BacktestConfig:
    initial_capital: float = DEFAULTS.initial_capital
    execution_lag: int = DEFAULTS.execution_lag      # bars between signal and fill
    cost_model: CostModel = field(default_factory=CostModel)


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    drawdown: pd.Series
    positions: pd.Series          # actually-held position after lag
    trades: pd.DataFrame
    periods_per_year: float
    total_costs: float            # cumulative cost paid, in currency
    config: BacktestConfig


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        return 252.0
    years = span_days / 365.25
    return float(np.clip(len(index) / years, 1.0, 365.0 * 24))


def _extract_trades(
    held: pd.Series, net_returns: pd.Series, price: pd.Series
) -> pd.DataFrame:
    """Group consecutive bars with the same non-zero position into trades."""
    trades = []
    n = len(held)
    i = 0
    pos_vals = held.to_numpy()
    while i < n:
        p = pos_vals[i]
        if p == 0.0:
            i += 1
            continue
        j = i
        while j + 1 < n and pos_vals[j + 1] == p:
            j += 1
        seg = net_returns.iloc[i : j + 1]
        trades.append(
            {
                "entry_time": held.index[i],
                "exit_time": held.index[j],
                "direction": "long" if p > 0 else "short",
                "size": abs(float(p)),
                "bars_held": j - i + 1,
                "return": float((1.0 + seg).prod() - 1.0),
                "entry_price": float(price.iloc[i]),
                "exit_price": float(price.iloc[j]),
            }
        )
        i = j + 1
    return pd.DataFrame(trades)


def run_backtest(
    df: pd.DataFrame,
    target_positions: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    cfg = config or BacktestConfig()
    close = df["close"].astype("float64")

    target = target_positions.reindex(df.index).fillna(0.0).clip(-1.0, 1.0)
    held = target.shift(cfg.execution_lag).fillna(0.0)

    asset_returns = close.pct_change().fillna(0.0)
    gross = held * asset_returns

    turnover = held.diff().abs()
    turnover.iloc[0] = abs(held.iloc[0])
    cost_rate = cfg.cost_model.cost_rate()
    costs = turnover * cost_rate

    net = gross - costs
    equity = cfg.initial_capital * (1.0 + net).cumprod()
    benchmark = cfg.initial_capital * (1.0 + asset_returns).cumprod()

    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0

    trades = _extract_trades(held, net, close)
    total_costs = float((costs * equity.shift(1).fillna(cfg.initial_capital)).sum())

    return BacktestResult(
        equity=equity,
        benchmark_equity=benchmark,
        net_returns=net,
        gross_returns=gross,
        drawdown=drawdown,
        positions=held,
        trades=trades,
        periods_per_year=_infer_periods_per_year(df.index),
        total_costs=total_costs,
        config=cfg,
    )

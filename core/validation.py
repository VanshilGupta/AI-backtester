"""Robustness & statistical-significance tools — the 'is the edge real?' layer.

  * train_test_split  -> in-sample vs out-of-sample metrics (default 70/30).
  * sharpe_confidence -> Probabilistic Sharpe Ratio: P(true Sharpe > 0),
    adjusted for sample size, skew and fat tails (Bailey & Lopez de Prado).
  * monte_carlo       -> block-bootstrap confidence bands on Sharpe / CAGR /
    max drawdown, plus probability of a profitable path.

All functions are pure (no I/O) and reuse the existing engine/metrics so the
numbers are consistent with the headline backtest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestConfig, BacktestResult, _infer_periods_per_year
from .metrics import Metrics, compute_metrics


# --- Out-of-sample (train/test) ---------------------------------------------
@dataclass
class SplitMetrics:
    train_frac: float
    split_date: pd.Timestamp
    is_metrics: Metrics       # in-sample (train)
    oos_metrics: Metrics      # out-of-sample (test)
    degradation: float        # oos_sharpe / is_sharpe (1.0 = holds up)
    holds_up: bool            # OOS keeps a meaningful share of the IS edge


def _sub_result(result: BacktestResult, lo: int, hi: int) -> BacktestResult:
    """Build a self-consistent BacktestResult for the bar window [lo, hi).

    Equity is recompounded from the sliced net returns so start == capital;
    indicators are causal, so slicing performance is a fair OOS read."""
    cap = result.config.initial_capital
    net = result.net_returns.iloc[lo:hi]
    gross = result.gross_returns.iloc[lo:hi]
    bench_net = result.benchmark_equity.pct_change().iloc[lo:hi].fillna(0.0)

    equity = cap * (1.0 + net).cumprod()
    bench_equity = cap * (1.0 + bench_net).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    positions = result.positions.iloc[lo:hi]

    idx = result.equity.index
    window = (idx[lo], idx[hi - 1])
    trades = result.trades
    if len(trades):
        mask = (trades["entry_time"] >= window[0]) & (
            trades["entry_time"] <= window[1]
        )
        trades = trades[mask].reset_index(drop=True)

    return BacktestResult(
        equity=equity,
        benchmark_equity=bench_equity,
        net_returns=net,
        gross_returns=gross,
        drawdown=drawdown,
        positions=positions,
        trades=trades,
        periods_per_year=_infer_periods_per_year(equity.index),
        total_costs=0.0,
        config=result.config,
    )


def train_test_split(
    result: BacktestResult, risk_free_rate: float, train_frac: float = 0.70
) -> SplitMetrics:
    n = len(result.equity)
    k = int(round(n * train_frac))
    k = max(2, min(k, n - 2))  # keep both windows non-trivial

    is_m = compute_metrics(_sub_result(result, 0, k), risk_free_rate)
    oos_m = compute_metrics(_sub_result(result, k, n), risk_free_rate)

    if is_m.sharpe > 0:
        degradation = oos_m.sharpe / is_m.sharpe
    else:
        degradation = 0.0
    holds_up = oos_m.sharpe > 0 and degradation >= 0.5

    return SplitMetrics(
        train_frac=train_frac,
        split_date=result.equity.index[k],
        is_metrics=is_m,
        oos_metrics=oos_m,
        degradation=degradation,
        holds_up=holds_up,
    )


# --- Probabilistic Sharpe Ratio ---------------------------------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class SharpeConfidence:
    psr: float                # P(true Sharpe > benchmark), 0-1
    sr_benchmark_annual: float
    n_obs: int


def sharpe_confidence(
    result: BacktestResult,
    risk_free_rate: float,
    sr_benchmark_annual: float = 0.0,
) -> SharpeConfidence:
    """Probabilistic Sharpe Ratio: the probability the *true* Sharpe exceeds a
    benchmark Sharpe, given the observed sample, its skew and its kurtosis.

    A high Sharpe on a short, fat-tailed sample is not trustworthy — PSR makes
    that explicit as a single 0-100% confidence number."""
    r = result.net_returns.to_numpy(dtype="float64")
    n = r.size
    ppy = result.periods_per_year
    rf_period = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0
    excess = r - rf_period

    if n < 8 or excess.std() == 0:
        return SharpeConfidence(0.5, sr_benchmark_annual, n)

    mean = excess.mean()
    m2 = np.mean((excess - mean) ** 2)
    sd = math.sqrt(m2)
    sr = mean / sd  # per-period Sharpe estimate
    skew = float(np.mean((excess - mean) ** 3) / m2 ** 1.5)
    kurt = float(np.mean((excess - mean) ** 4) / m2 ** 2)  # non-excess

    sr_star = sr_benchmark_annual / math.sqrt(ppy)  # de-annualise the bar
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2
    if denom <= 0:
        return SharpeConfidence(0.5, sr_benchmark_annual, n)

    z = (sr - sr_star) * math.sqrt(n - 1) / math.sqrt(denom)
    return SharpeConfidence(_norm_cdf(z), sr_benchmark_annual, n)


# --- Monte Carlo (block bootstrap) ------------------------------------------
@dataclass
class MonteCarlo:
    n_sims: int
    block: int
    sharpe: tuple[float, float, float]   # p5, p50, p95
    cagr: tuple[float, float, float]
    max_drawdown: tuple[float, float, float]
    prob_profit: float                   # P(final equity > start)
    prob_beat_benchmark: float           # P(CAGR > benchmark CAGR)


def monte_carlo(
    result: BacktestResult,
    risk_free_rate: float,
    n_sims: int = 500,
    block: int | None = None,
    seed: int = 12345,
) -> MonteCarlo:
    """Resample the realised return path in contiguous blocks (preserving
    short-term autocorrelation) to get a distribution of outcomes rather than
    a single point estimate."""
    net = result.net_returns.to_numpy(dtype="float64")
    n = net.size
    ppy = result.periods_per_year
    block = block or max(5, int(round(ppy / 52)))  # ~1 week of bars
    block = min(block, n)

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(n_sims, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_sims, -1)[:, :n]
    sims = net[idx]  # (n_sims, n)

    rf_period = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0
    mean = sims.mean(axis=1)
    std = sims.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = np.where(std > 0, (mean - rf_period) / std * math.sqrt(ppy), 0.0)

    eq = np.cumprod(1.0 + sims, axis=1)
    total = eq[:, -1] - 1.0
    base = np.maximum(1.0 + total, 1e-9)
    cagr = base ** (ppy / n) - 1.0
    peak = np.maximum.accumulate(eq, axis=1)
    max_dd = (eq / peak - 1.0).min(axis=1)

    bench_cagr = (
        (float(result.benchmark_equity.iloc[-1])
         / float(result.benchmark_equity.iloc[0]))
        ** (ppy / n) - 1.0
    )

    def pct(a):
        return tuple(float(x) for x in np.percentile(a, [5, 50, 95]))

    return MonteCarlo(
        n_sims=n_sims,
        block=block,
        sharpe=pct(sharpe),
        cagr=pct(cagr),
        max_drawdown=pct(max_dd),
        prob_profit=float((total > 0).mean()),
        prob_beat_benchmark=float((cagr > bench_cagr).mean()),
    )

"""Post-backtest analytics.

These functions go beyond the headline metrics so you can actually judge
whether an edge is real: return distribution, rolling risk-adjusted return,
drawdown episodes, calendar returns, trade-level statistics, and a fair
benchmark comparison (alpha/beta/information ratio/capture).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import BacktestResult


# --- Calendar / time-series views -------------------------------------------
def monthly_returns_table(net_returns: pd.Series) -> pd.DataFrame:
    """Year x Month return matrix (%), with a 'Year' total column."""
    r = net_returns.copy()
    r.index = pd.to_datetime(r.index)
    monthly = (1.0 + r).resample("ME").prod() - 1.0
    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.values,
        }
    )
    pivot = frame.pivot_table(index="year", columns="month", values="ret")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    yearly = (1.0 + r).groupby(r.index.year).prod() - 1.0
    pivot["Year"] = yearly
    return (pivot * 100.0).round(2)


def rolling_sharpe(
    net_returns: pd.Series, periods_per_year: float, window: int | None = None
) -> pd.Series:
    """Rolling annualised Sharpe — exposes whether the edge is persistent or
    driven by one lucky stretch."""
    win = window or int(max(round(periods_per_year / 2), 20))
    mean = net_returns.rolling(win).mean()
    std = net_returns.rolling(win).std()
    return (mean / std * np.sqrt(periods_per_year)).replace(
        [np.inf, -np.inf], np.nan
    )


def drawdown_table(equity: pd.Series, top: int = 5) -> pd.DataFrame:
    """The `top` deepest drawdown episodes: peak, trough, recovery, depth, length."""
    eq = equity.astype("float64")
    running_max = eq.cummax()
    dd = eq / running_max - 1.0

    episodes = []
    in_dd = False
    peak_idx = eq.index[0]
    trough_idx = eq.index[0]
    trough_val = 0.0

    for ts, d in dd.items():
        if not in_dd and d < 0:
            in_dd = True
            peak_idx = running_max.loc[:ts][running_max.loc[:ts] == running_max.loc[ts]].index[0]
            trough_idx, trough_val = ts, d
        elif in_dd:
            if d < trough_val:
                trough_idx, trough_val = ts, d
            if d >= 0:  # recovered
                episodes.append((peak_idx, trough_idx, ts, trough_val))
                in_dd = False
    if in_dd:
        episodes.append((peak_idx, trough_idx, None, trough_val))

    rows = []
    for peak, trough, recovery, depth in episodes:
        length = (
            (recovery - peak).days if recovery is not None else None
        )
        rows.append(
            {
                "Peak": pd.Timestamp(peak).date(),
                "Trough": pd.Timestamp(trough).date(),
                "Recovered": pd.Timestamp(recovery).date()
                if recovery is not None
                else "ongoing",
                "Depth %": round(depth * 100.0, 2),
                "Length (days)": length,
            }
        )
    out = pd.DataFrame(rows).sort_values("Depth %").head(top)
    return out.reset_index(drop=True)


# --- Distribution & trade statistics ----------------------------------------
@dataclass
class TradeStats:
    expectancy: float          # mean return per trade
    payoff_ratio: float        # avg win / |avg loss|
    profit_factor: float
    max_win_streak: int
    max_loss_streak: int
    avg_bars_held: float
    t_stat: float              # significance of mean trade return vs 0
    kelly_fraction: float      # suggested fractional sizing (capped at 1)


def _streaks(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def trade_stats(result: BacktestResult) -> TradeStats | None:
    trades = result.trades
    if not len(trades):
        return None
    r = trades["return"].to_numpy(dtype="float64")
    wins = r[r > 0]
    losses = r[r < 0]
    avg_win = wins.mean() if len(wins) else 0.0
    avg_loss = losses.mean() if len(losses) else 0.0
    win_rate = len(wins) / len(r)
    payoff = (avg_win / abs(avg_loss)) if avg_loss < 0 else float("inf")
    gp, gl = wins.sum(), -losses.sum()
    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
    t_stat = (
        float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
        if len(r) > 1 and r.std(ddof=1) > 0
        else 0.0
    )
    if payoff in (float("inf"),) or avg_loss == 0:
        kelly = win_rate
    else:
        kelly = win_rate - (1.0 - win_rate) / payoff
    return TradeStats(
        expectancy=float(r.mean()),
        payoff_ratio=float(payoff),
        profit_factor=float(pf),
        max_win_streak=_streaks(r > 0),
        max_loss_streak=_streaks(r < 0),
        avg_bars_held=float(trades["bars_held"].mean()),
        t_stat=t_stat,
        kelly_fraction=float(np.clip(kelly, 0.0, 1.0)),
    )


def return_distribution(net_returns: pd.Series) -> dict[str, float]:
    r = net_returns[net_returns != 0.0]
    if not len(r):
        return {}
    arr = r.to_numpy(dtype="float64")
    mean, std = arr.mean(), arr.std()
    skew = float(((arr - mean) ** 3).mean() / std**3) if std > 0 else 0.0
    kurt = float(((arr - mean) ** 4).mean() / std**4 - 3.0) if std > 0 else 0.0
    return {
        "mean_%": round(mean * 100, 4),
        "std_%": round(std * 100, 4),
        "skew": round(skew, 3),
        "excess_kurtosis": round(kurt, 3),
        "p05_%": round(float(np.percentile(arr, 5)) * 100, 4),
        "p95_%": round(float(np.percentile(arr, 95)) * 100, 4),
        "var_95_%": round(float(np.percentile(arr, 5)) * 100, 4),
        "cvar_95_%": round(float(arr[arr <= np.percentile(arr, 5)].mean()) * 100, 4),
    }


# --- Fair benchmark comparison ----------------------------------------------
@dataclass
class BenchmarkComparison:
    benchmark_name: str
    alpha_annual: float        # CAPM alpha, annualised
    beta: float
    correlation: float
    tracking_error: float      # annualised stdev of active return
    information_ratio: float   # active return / tracking error
    up_capture: float
    down_capture: float
    pct_periods_outperformed: float


def benchmark_comparison(
    strategy: BacktestResult,
    benchmark: BacktestResult,
    benchmark_name: str,
    risk_free_rate: float,
) -> BenchmarkComparison:
    s = strategy.net_returns.reindex(benchmark.net_returns.index).fillna(0.0)
    b = benchmark.net_returns
    ppy = strategy.periods_per_year
    rf = (1.0 + risk_free_rate) ** (1.0 / ppy) - 1.0

    # CAPM: beta = cov(strategy, benchmark) / var(benchmark); the constant rf
    # cancels in both. Jensen's alpha is the per-period intercept, annualised
    # arithmetically (alpha * periods/yr) — the conventional convention.
    var_b = float(b.var())
    beta = float(((s - rf).cov(b - rf)) / var_b) if var_b > 0 else 0.0
    alpha_period = float((s - rf).mean() - beta * (b - rf).mean())
    alpha_annual = alpha_period * ppy

    active = s - b
    te = float(active.std() * np.sqrt(ppy))
    ir = float(active.mean() / active.std() * np.sqrt(ppy)) if active.std() > 0 else 0.0
    corr = float(s.corr(b)) if s.std() > 0 and b.std() > 0 else 0.0

    up = b > 0
    down = b < 0
    up_cap = (
        float((1 + s[up]).prod() ** (1 / max(up.sum(), 1)) - 1)
        / float((1 + b[up]).prod() ** (1 / max(up.sum(), 1)) - 1)
        if up.sum() and (1 + b[up]).prod() != 1
        else 0.0
    )
    down_cap = (
        float((1 + s[down]).prod() ** (1 / max(down.sum(), 1)) - 1)
        / float((1 + b[down]).prod() ** (1 / max(down.sum(), 1)) - 1)
        if down.sum() and (1 + b[down]).prod() != 1
        else 0.0
    )
    return BenchmarkComparison(
        benchmark_name=benchmark_name,
        alpha_annual=alpha_annual,
        beta=beta,
        correlation=corr,
        tracking_error=te,
        information_ratio=ir,
        up_capture=up_cap,
        down_capture=down_cap,
        pct_periods_outperformed=float((active > 0).mean()),
    )


# --- Plain-English read ------------------------------------------------------
def quality_report(
    metrics,
    tstats: TradeStats | None,
    comp: BenchmarkComparison | None,
) -> list[tuple[str, str]]:
    """Return a list of (status, message) where status is good|warn|bad.

    This is a heuristic sanity layer on top of the deterministic 0-10 score —
    it tells the user, in words, *why* a strategy is or isn't trustworthy.
    """
    out: list[tuple[str, str]] = []

    if metrics.sharpe >= 1.0:
        out.append(("good", f"Sharpe {metrics.sharpe:.2f} (risk-adjusted return is solid)."))
    elif metrics.sharpe >= 0.5:
        out.append(("warn", f"Sharpe {metrics.sharpe:.2f} is mediocre - weak risk-adjusted edge."))
    else:
        out.append(("bad", f"Sharpe {metrics.sharpe:.2f} - no meaningful risk-adjusted edge."))

    if comp is not None:
        if comp.alpha_annual > 0.0 and comp.information_ratio > 0.5:
            out.append((
                "good",
                f"Beats {comp.benchmark_name}: alpha {comp.alpha_annual:+.1%}/yr, "
                f"IR {comp.information_ratio:.2f}.",
            ))
        elif comp.alpha_annual > 0.0:
            out.append((
                "warn",
                f"Slight edge over {comp.benchmark_name} (alpha {comp.alpha_annual:+.1%}) "
                f"but inconsistent (IR {comp.information_ratio:.2f}).",
            ))
        else:
            out.append((
                "bad",
                f"Does not beat {comp.benchmark_name} on a risk-adjusted basis "
                f"(alpha {comp.alpha_annual:+.1%}/yr).",
            ))

    if tstats is not None:
        if abs(tstats.t_stat) >= 2.0:
            out.append((
                "good",
                f"Per-trade edge is statistically significant (t = {tstats.t_stat:.2f}).",
            ))
        else:
            out.append((
                "warn",
                f"Per-trade edge is not statistically significant (t = {tstats.t_stat:.2f}; "
                f"want |t| >= 2). Likely noise.",
            ))

    if metrics.num_trades < 30:
        out.append((
            "warn",
            f"Only {metrics.num_trades} trades - small sample, treat results with caution.",
        ))

    if metrics.max_drawdown < -0.35:
        out.append((
            "bad",
            f"Max drawdown {metrics.max_drawdown:.0%} would be very hard to hold live.",
        ))

    if metrics.profit_factor == float("inf"):
        out.append((
            "bad",
            "No losing trades — almost certainly lookahead bias or overfitting.",
        ))
    return out


# --- Copy-paste report for an external LLM ----------------------------------
def _fmt_metrics(m) -> str:
    d = m.as_dict()
    pf = d["profit_factor"]
    return (
        f"  total_return   {d['total_return']:+.2%}\n"
        f"  cagr           {d['cagr']:+.2%}   (benchmark {d['benchmark_cagr']:+.2%})\n"
        f"  excess_cagr    {d['excess_cagr']:+.2%}\n"
        f"  volatility     {d['volatility']:.2%}\n"
        f"  sharpe         {d['sharpe']:.2f}\n"
        f"  sortino        {d['sortino']:.2f}\n"
        f"  max_drawdown   {d['max_drawdown']:.2%}\n"
        f"  calmar         {d['calmar']:.2f}\n"
        f"  win_rate       {d['win_rate']:.1%}\n"
        f"  profit_factor  {'inf' if pf == float('inf') else f'{pf:.2f}'}\n"
        f"  num_trades     {d['num_trades']}\n"
        f"  exposure       {d['exposure']:.1%}\n"
        f"  avg_win        {d['avg_win']:+.2%}   avg_loss {d['avg_loss']:+.2%}\n"
        f"  best/worst     {d['best_trade']:+.2%} / {d['worst_trade']:+.2%}"
    )


def format_llm_report(
    *,
    spec,
    config,
    source_desc: str,
    df: pd.DataFrame,
    metrics,
    bench_metrics,
    benchmark_name: str,
    verdict,
    comp: "BenchmarkComparison",
    tstats: "TradeStats | None",
    dd_table: pd.DataFrame,
    monthly_table: pd.DataFrame,
    quality: list[tuple[str, str]],
) -> str:
    """A compact, copy-pasteable text dump of one backtest run, designed to be
    handed to an LLM with: "analyse this and suggest improvements".

    Deliberately excludes the full equity/return series (too large, low value)
    and keeps the strategy code, settings, metrics, benchmark, verdict and
    diagnostics — everything needed to reason about the edge."""
    cm = config.cost_model
    parts: list[str] = []
    parts.append("=== BACKTEST RUN — for strategy analysis & improvement ===")
    parts.append(
        "You are a quantitative strategy reviewer. Analyse the results below: "
        "is the edge real or likely overfit/regime-luck? What are the specific "
        "weaknesses, and how would you concretely improve the rules? Be candid "
        "and specific.\n"
    )

    parts.append("--- STRATEGY ---")
    parts.append(f"name: {spec.name}")
    parts.append(f"direction: {spec.direction}   regime: {spec.market_regime}")
    parts.append(f"description: {spec.description}")
    parts.append(f"rationale: {spec.rationale}")
    parts.append("code:\n```python\n" + spec.code.strip() + "\n```")

    parts.append("\n--- DATA & SETTINGS ---")
    parts.append(f"data_source: {source_desc}")
    parts.append(
        f"period: {df.index[0].date()} to {df.index[-1].date()} "
        f"({len(df):,} bars)"
    )
    parts.append(
        f"initial_capital: {config.initial_capital:,.0f}   "
        f"execution_lag: {config.execution_lag} bar(s)"
    )
    parts.append(
        f"costs: {cm.effective_bps_per_turnover():.2f} bps/side "
        f"(round trip ~{cm.round_trip_pct():.3f}% of notional); "
        f"breakdown bps -> " + ", ".join(
            f"{k} {v}" for k, v in cm.breakdown().items()
        )
    )

    parts.append("\n--- STRATEGY METRICS ---")
    parts.append(_fmt_metrics(metrics))
    parts.append(f"\n--- BENCHMARK ({benchmark_name}) METRICS ---")
    parts.append(_fmt_metrics(bench_metrics))

    parts.append(f"\n--- VS BENCHMARK ({benchmark_name}, same costs) ---")
    parts.append(
        f"  alpha_annual   {comp.alpha_annual:+.2%}\n"
        f"  beta           {comp.beta:.2f}\n"
        f"  correlation    {comp.correlation:.2f}\n"
        f"  info_ratio     {comp.information_ratio:.2f}\n"
        f"  tracking_err   {comp.tracking_error:.2%}\n"
        f"  up_capture     {comp.up_capture:.0%}   "
        f"down_capture {comp.down_capture:.0%}\n"
        f"  periods_outperf {comp.pct_periods_outperformed:.0%}"
    )

    if tstats is not None:
        po = tstats.payoff_ratio
        parts.append("\n--- TRADE STATS ---")
        parts.append(
            f"  expectancy/trade {tstats.expectancy:+.3%}\n"
            f"  payoff_ratio   {'inf' if po == float('inf') else f'{po:.2f}'}\n"
            f"  profit_factor  {tstats.profit_factor:.2f}\n"
            f"  t_stat         {tstats.t_stat:.2f}  (|t|>=2 = significant)\n"
            f"  kelly_fraction {tstats.kelly_fraction:.0%}\n"
            f"  max_win/loss_streak {tstats.max_win_streak}/{tstats.max_loss_streak}"
            f"   avg_bars_held {tstats.avg_bars_held:.1f}"
        )

    parts.append("\n--- DETERMINISTIC VERDICT ---")
    parts.append(
        f"  score {verdict.score}/10  (raw {verdict.raw_score} x "
        f"confidence {verdict.confidence})  -> {verdict.recommendation}"
    )
    parts.append(
        "  components (score/10 x weight): "
        + ", ".join(
            f"{k} {v['score']}x{v['weight']}"
            for k, v in verdict.components.items()
        )
    )
    if verdict.risk_flags:
        parts.append("  risk_flags:")
        for f in verdict.risk_flags:
            parts.append(f"    - {f}")

    parts.append("\n--- QUALITY READ ---")
    for status, msg in quality:
        parts.append(f"  [{status}] {msg}")

    if not dd_table.empty:
        parts.append("\n--- WORST DRAWDOWNS ---")
        parts.append(dd_table.to_string(index=False))

    if not monthly_table.empty:
        parts.append("\n--- YEARLY RETURNS (%) ---")
        yearly = monthly_table["Year"]
        parts.append(
            "  " + "  ".join(f"{idx}: {val:+.1f}" for idx, val in yearly.items())
        )

    parts.append(
        "\n=== END — suggest concrete, testable rule changes (entries, exits, "
        "filters, sizing, regime) and call out any overfitting risk. ==="
    )
    return "\n".join(parts)

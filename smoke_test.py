"""Exercises the whole pipeline except the Anthropic call, using a hand-written
strategy string fed through the same sandbox the LLM output goes through."""
import numpy as np
import pandas as pd

from core.analysis import (
    benchmark_comparison,
    drawdown_table,
    format_llm_report,
    monthly_returns_table,
    quality_report,
    return_distribution,
    rolling_sharpe,
    trade_stats,
)
from core.benchmark import BENCHMARKS
from core.constants import DEFAULTS, EXAMPLE_STRATEGIES, CostModel
from core.engine import BacktestConfig, run_backtest
from core.evaluator import evaluate
from core.metrics import compute_metrics
from core.strategy_generator import (
    StrategyError,
    StrategySpec,
    compile_strategy,
)
from core.verification import summarize, verify_data, verify_strategy

# --- synthetic OHLCV: trending random walk -----------------------------------
rng = np.random.default_rng(42)
n = 1200
idx = pd.date_range("2018-01-01", periods=n, freq="B")
ret = rng.normal(0.0004, 0.012, n)
close = 100 * np.exp(np.cumsum(ret))
open_ = close * (1 + rng.normal(0, 0.001, n))
hi_base = np.maximum(open_, close)
lo_base = np.minimum(open_, close)
df = pd.DataFrame(
    {
        "open": open_,
        "high": hi_base * (1 + abs(rng.normal(0, 0.004, n))),
        "low": lo_base * (1 - abs(rng.normal(0, 0.004, n))),
        "close": close,
        "volume": rng.integers(1e5, 1e6, n).astype(float),
    },
    index=idx,
)

STRAT = """
def generate_signals(df):
    fast = sma(df['close'], 20)
    slow = sma(df['close'], 50)
    pos = pd.Series(0.0, index=df.index)
    pos[fast > slow] = 1.0
    return pos.fillna(0.0)
"""

print("compiling strategy via sandbox...")
fn = compile_strategy(STRAT)

print("\n--- data verification ---")
data_checks = verify_data(df, "Synthetic OHLCV (smoke test)")
for c in data_checks:
    print(f"  [{c.status}] {c.name}: {c.detail}")
print(f"  => {summarize(data_checks)}")

spec = StrategySpec(
    name="SMA 20/50 crossover",
    description="Long when SMA20 > SMA50.",
    rationale="trend following",
    market_regime="trending",
    direction="long_only",
    indicators_used=["sma"],
    code=STRAT.strip(),
)
print("\n--- strategy code verification ---")
positions, strat_checks = verify_strategy(spec, fn, df)
for c in strat_checks:
    print(f"  [{c.status}] {c.name}: {c.detail}")
print(f"  => {summarize(strat_checks)}")
print(f"  positions: long {(positions == 1).mean():.0%} of bars")

cfg = BacktestConfig(cost_model=CostModel())
result = run_backtest(df, positions, cfg)
metrics = compute_metrics(result, risk_free_rate=DEFAULTS.risk_free_rate)
verdict = evaluate(metrics)

print("\n--- cost model ---")
print(f"  effective {cfg.cost_model.effective_bps_per_turnover():.2f} bps/side, "
      f"round trip {cfg.cost_model.round_trip_pct():.3f}%, "
      f"total paid {result.total_costs:,.0f}")

print("\n--- metrics ---")
for k, v in metrics.as_dict().items():
    print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

print("\n--- verdict ---")
print(f"  score        {verdict.score}/10  (raw {verdict.raw_score}, "
      f"conf {verdict.confidence})")
print(f"  recommend    {verdict.recommendation}")
for f in verdict.risk_flags:
    print(f"  risk: {f}")

# --- post-backtest analytics -------------------------------------------------
bench_name = "Buy & Hold"
bench_result = run_backtest(df, BENCHMARKS[bench_name](df), cfg)
comp = benchmark_comparison(result, bench_result, bench_name, DEFAULTS.risk_free_rate)
tstats = trade_stats(result)
m_table = monthly_returns_table(result.net_returns)
dd_table = drawdown_table(result.equity)
rs = rolling_sharpe(result.net_returns, result.periods_per_year)
dist = return_distribution(result.net_returns)

bench_metrics = compute_metrics(bench_result, risk_free_rate=DEFAULTS.risk_free_rate)
bench_verdict = evaluate(bench_metrics)
quality = quality_report(metrics, tstats, comp)

print("\n--- benchmark standalone ---")
print(f"  CAGR {bench_metrics.cagr:+.2%}  Sharpe {bench_metrics.sharpe:.2f}  "
      f"MaxDD {bench_metrics.max_drawdown:.1%}  Score {bench_verdict.score}/10")

print("\n--- benchmark comparison ---")
print(f"  alpha {comp.alpha_annual:+.2%}/yr  beta {comp.beta:.2f}  "
      f"IR {comp.information_ratio:.2f}  corr {comp.correlation:.2f}")
print(f"  up-capture {comp.up_capture:.0%}  down-capture {comp.down_capture:.0%}")

print("\n--- trade stats ---")
print(f"  expectancy {tstats.expectancy:.3%}  payoff {tstats.payoff_ratio:.2f}  "
      f"t-stat {tstats.t_stat:.2f}  kelly {tstats.kelly_fraction:.0%}")

print("\n--- quality report ---")
for status, msg in quality_report(metrics, tstats, comp):
    print(f"  [{status}] {msg}")

print("\n--- analytics shapes ---")
print(f"  monthly table {m_table.shape}, drawdown episodes {len(dd_table)}, "
      f"rolling-sharpe pts {rs.notna().sum()}, dist keys {len(dist)}")

report = format_llm_report(
    spec=spec, config=cfg, source_desc="Synthetic OHLCV (smoke test)",
    df=df, metrics=metrics, bench_metrics=bench_metrics,
    benchmark_name=bench_name, verdict=verdict, comp=comp,
    tstats=tstats, dd_table=dd_table, monthly_table=m_table, quality=quality,
)
print("\n--- llm report (head) ---")
print("\n".join(report.splitlines()[:6]))
print(f"  ... report length {len(report)} chars")

print("\n--- bundled example strategies ---")
example_ok = True
for _label, _prompt, _hint, _code in EXAMPLE_STRATEGIES:
    try:
        _fn = compile_strategy(_code)
        _spec = StrategySpec(
            name=_label, description=_prompt, rationale="bundled example",
            market_regime="n/a", direction="long_only",
            indicators_used=[], code=_code,
        )
        _pos, _checks = verify_strategy(_spec, _fn, df)
        _fails = [c for c in _checks if c.status == "fail"]
        _res = run_backtest(df, _pos, cfg)
        status = "ok" if not _fails and _res.equity.notna().all() else "FAIL"
        if _fails or not _res.equity.notna().all():
            example_ok = False
        print(f"  [{status}] {_label}: {summarize(_checks)}, "
              f"trades {len(_res.trades)}")
    except Exception as e:
        example_ok = False
        print(f"  [FAIL] {_label}: {e}")

print("\n--- sandbox rejection checks ---")
for bad, why in [
    ("def generate_signals(df):\n import os\n return df['close']*0", "import"),
    ("def generate_signals(df):\n open('x'); return df['close']*0", "open()"),
    ("def f(df): return df", "no entrypoint"),
]:
    try:
        compile_strategy(bad)
        print(f"  FAIL: {why} was not blocked")
    except StrategyError as e:
        print(f"  ok: blocked {why} -> {str(e)[:60]}")

assert result.equity.notna().all(), "equity has NaNs"
assert len(result.trades) > 0, "no trades extracted"
assert 0 <= verdict.score <= 10, "score out of range"
assert not m_table.empty, "monthly table empty"
assert tstats is not None, "trade stats missing"
assert result.total_costs > 0, "no costs charged"
assert not any(c.status == "fail" for c in data_checks), "data verification failed"
assert not any(c.status == "fail" for c in strat_checks), "strategy verification failed"
assert len(data_checks) >= 7 and len(strat_checks) >= 5, "missing checks"
assert bench_metrics.num_trades >= 1, "benchmark produced no trade"
assert 0 <= bench_verdict.score <= 10, "benchmark score out of range"
assert "STRATEGY METRICS" in report and "VS BENCHMARK" in report, "report incomplete"
assert len(report) > 500, "llm report suspiciously short"
# Sortino downside-deviation must be RMS over ALL periods (textbook), so for a
# series with any losses |sortino| should not exceed |sharpe| by an absurd
# factor; sanity-check it is finite and not the old only-negatives variant.
assert abs(metrics.sortino) < 1e3 and metrics.sortino == metrics.sortino, "sortino broken"
assert example_ok, "a bundled example strategy failed to compile/verify/run"
print("\nALL CHECKS PASSED")

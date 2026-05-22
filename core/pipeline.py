"""One backtest, end to end.

`run_full` is the single source of truth for turning a StrategySpec + data into
a fully analysed result: compile -> verify -> (optional risk overlay) ->
backtest -> benchmark -> metrics -> robustness (OOS / PSR / Monte Carlo) ->
diagnostics. Both the initial run and the improvement loop call it, so they can
never drift apart.

The LLM verdict narrative is intentionally *not* here — it needs API
credentials and is best-effort, so the UI layer attaches it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .analysis import (
    BenchmarkComparison,
    TradeStats,
    benchmark_comparison,
    drawdown_table,
    monthly_returns_table,
    quality_report,
    trade_stats,
)
from .benchmark import BENCHMARKS
from .engine import BacktestConfig, BacktestResult, run_backtest
from .engine import _infer_periods_per_year
from .evaluator import Verdict, evaluate
from .metrics import Metrics, compute_metrics
from .overlays import OverlayConfig, apply_overlays
from .strategy_generator import StrategySpec, compile_strategy
from .validation import (
    MonteCarlo,
    SharpeConfidence,
    SplitMetrics,
    monte_carlo,
    sharpe_confidence,
    train_test_split,
)
from .verification import Check, verify_strategy


@dataclass
class RunResult:
    spec: StrategySpec
    strat_checks: list[Check]
    ok: bool                            # False if a verification check FAILED
    positions: pd.Series | None = None
    overlay_notes: list[str] | None = None
    result: BacktestResult | None = None
    bench_result: BacktestResult | None = None
    bench_name: str | None = None
    metrics: Metrics | None = None
    bench_metrics: Metrics | None = None
    verdict: Verdict | None = None
    bench_verdict: Verdict | None = None
    comp: BenchmarkComparison | None = None
    tstats: TradeStats | None = None
    monthly_table: pd.DataFrame | None = None
    dd_table: pd.DataFrame | None = None
    quality: list[tuple[str, str]] | None = None
    split: SplitMetrics | None = None
    sharpe_conf: SharpeConfidence | None = None
    mc: MonteCarlo | None = None


def run_full(
    df: pd.DataFrame,
    spec: StrategySpec,
    config: BacktestConfig,
    *,
    benchmark_name: str,
    risk_free_rate: float,
    overlay_cfg: OverlayConfig | None = None,
    train_frac: float = 0.70,
) -> RunResult:
    """Compile, verify and fully analyse `spec` on `df`. Raises StrategyError
    only if the code won't compile; verification *failures* come back as
    `ok=False` so the caller can show the checklist."""
    overlay_cfg = overlay_cfg or OverlayConfig()
    fn = compile_strategy(spec.code)  # raises StrategyError on bad code
    positions, checks = verify_strategy(spec, fn, df)
    if any(c.status == "fail" for c in checks):
        return RunResult(spec=spec, strat_checks=checks, ok=False)

    ppy = _infer_periods_per_year(df.index)
    positions, overlay_notes = apply_overlays(positions, df, overlay_cfg, ppy)

    result = run_backtest(df, positions, config)
    bench_result = run_backtest(df, BENCHMARKS[benchmark_name](df), config)

    metrics = compute_metrics(result, risk_free_rate=risk_free_rate)
    bench_metrics = compute_metrics(bench_result, risk_free_rate=risk_free_rate)
    verdict = evaluate(metrics)
    bench_verdict = evaluate(bench_metrics)

    comp = benchmark_comparison(result, bench_result, benchmark_name, risk_free_rate)
    tstats = trade_stats(result)
    quality = quality_report(metrics, tstats, comp)

    return RunResult(
        spec=spec,
        strat_checks=checks,
        ok=True,
        positions=positions,
        overlay_notes=overlay_notes,
        result=result,
        bench_result=bench_result,
        bench_name=benchmark_name,
        metrics=metrics,
        bench_metrics=bench_metrics,
        verdict=verdict,
        bench_verdict=bench_verdict,
        comp=comp,
        tstats=tstats,
        monthly_table=monthly_returns_table(result.net_returns),
        dd_table=drawdown_table(result.equity),
        quality=quality,
        split=train_test_split(result, risk_free_rate, train_frac),
        sharpe_conf=sharpe_confidence(result, risk_free_rate),
        mc=monte_carlo(result, risk_free_rate),
    )

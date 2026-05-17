"""Turn metrics into a transparent 0-10 verdict plus an optional LLM assessment.

The numeric score is fully deterministic and explainable (every component is
shown). The LLM step only adds a qualitative narrative + risk read; it never
changes the number.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .llm import ANTHROPIC, generate_text
from .metrics import Metrics
from .strategy_generator import StrategySpec

# (weight, label) per scored component. Weights sum to 1.0.
_WEIGHTS = {
    "sharpe": 0.28,
    "cagr": 0.15,
    "drawdown": 0.17,
    "calmar": 0.12,
    "profit_factor": 0.10,
    "excess_return": 0.13,
    "win_rate": 0.05,
}


def _lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _score_sharpe(s: float) -> float:
    if s <= 0:
        return 0.0
    if s <= 1:
        return _lerp(s, 0, 1, 0, 5)
    if s <= 2:
        return _lerp(s, 1, 2, 5, 8)
    return _lerp(s, 2, 3, 8, 10)


def _score_cagr(c: float) -> float:
    return _lerp(c, 0.0, 0.30, 0.0, 10.0)


def _score_drawdown(dd: float) -> float:
    # dd is negative; shallower is better.
    return _lerp(abs(dd), 0.0, 0.50, 10.0, 0.0)


def _score_calmar(c: float) -> float:
    return _lerp(c, 0.0, 3.0, 0.0, 10.0)


def _score_profit_factor(pf: float) -> float:
    if pf == float("inf"):
        return 10.0
    if pf <= 1.0:
        return _lerp(pf, 0.0, 1.0, 0.0, 3.0)
    if pf <= 2.0:
        return _lerp(pf, 1.0, 2.0, 3.0, 7.0)
    return _lerp(pf, 2.0, 3.0, 7.0, 10.0)


def _score_excess(x: float) -> float:
    return _lerp(x, -0.10, 0.20, 0.0, 10.0)


def _score_win_rate(w: float) -> float:
    return _lerp(w, 0.30, 0.65, 2.0, 10.0)


def _confidence(metrics: Metrics) -> float:
    """0-1 multiplier: too few trades => the stats aren't trustworthy."""
    n = metrics.num_trades
    if n <= 2:
        return 0.35
    if n >= 30:
        return 1.0
    return _lerp(n, 2, 30, 0.35, 1.0)


@dataclass
class Verdict:
    score: float                       # 0-10, confidence-adjusted
    raw_score: float                   # 0-10, before confidence
    confidence: float                  # 0-1
    recommendation: str
    components: dict[str, dict] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)
    llm_assessment: str | None = None


def _recommendation(score: float) -> str:
    if score >= 7.5:
        return "Implement"
    if score >= 6.0:
        return "Implement with caution"
    if score >= 4.0:
        return "Needs work"
    return "Do not implement"


def _risk_flags(metrics: Metrics) -> list[str]:
    flags = []
    if metrics.num_trades < 10:
        flags.append(
            f"Only {metrics.num_trades} trades — results may be statistical noise."
        )
    if metrics.excess_cagr <= 0:
        flags.append("Does not beat buy & hold on a CAGR basis.")
    if metrics.max_drawdown < -0.40:
        flags.append(
            f"Severe max drawdown ({metrics.max_drawdown:.0%}) — hard to hold live."
        )
    if metrics.exposure < 0.05:
        flags.append("Barely in the market (<5% exposure) — fragile sample.")
    if metrics.sharpe > 4 and metrics.num_trades < 20:
        flags.append("Suspiciously high Sharpe on few trades — likely overfit.")
    if metrics.profit_factor == float("inf"):
        flags.append("No losing trades — almost certainly overfit or lookahead.")
    return flags


def evaluate(metrics: Metrics) -> Verdict:
    raw_scores = {
        "sharpe": _score_sharpe(metrics.sharpe),
        "cagr": _score_cagr(metrics.cagr),
        "drawdown": _score_drawdown(metrics.max_drawdown),
        "calmar": _score_calmar(metrics.calmar),
        "profit_factor": _score_profit_factor(metrics.profit_factor),
        "excess_return": _score_excess(metrics.excess_cagr),
        "win_rate": _score_win_rate(metrics.win_rate),
    }
    raw = sum(raw_scores[k] * w for k, w in _WEIGHTS.items())
    conf = _confidence(metrics)
    score = raw * conf

    components = {
        k: {"score": round(raw_scores[k], 2), "weight": _WEIGHTS[k]}
        for k in _WEIGHTS
    }

    return Verdict(
        score=round(score, 2),
        raw_score=round(raw, 2),
        confidence=round(conf, 2),
        recommendation=_recommendation(score),
        components=components,
        risk_flags=_risk_flags(metrics),
    )


# --- Optional qualitative LLM read ------------------------------------------
_ASSESS_SYSTEM = """\
You are a skeptical buy-side risk reviewer. You are given a trading strategy's
description and its backtest metrics. In 4-7 sentences, give a candid assessment:
what works, what is concerning, and whether the edge is likely real or an
artifact of overfitting / regime luck. Be specific about the numbers. Do not
restate every metric; interpret them. End with one short sentence starting with
"Bottom line:".
"""


def attach_llm_assessment(
    verdict: Verdict,
    spec: StrategySpec,
    metrics: Metrics,
    provider: str = ANTHROPIC,
    model: str = "claude-opus-4-7",
    api_key: str | None = None,
) -> Verdict:
    """Best-effort: add a narrative read. Never raises — UI works without it."""
    try:
        payload = {
            "strategy": {
                "name": spec.name,
                "description": spec.description,
                "direction": spec.direction,
            },
            "metrics": metrics.as_dict(),
            "deterministic_score_out_of_10": verdict.score,
            "recommendation": verdict.recommendation,
            "risk_flags": verdict.risk_flags,
        }
        text = generate_text(
            provider=provider,
            model=model,
            system=_ASSESS_SYSTEM,
            user=json.dumps(payload, default=str),
            api_key=api_key,
            max_tokens=1024,
        )
        verdict.llm_assessment = text or None
    except Exception:
        verdict.llm_assessment = None
    return verdict

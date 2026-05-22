"""The improvement loop: feed one backtest's verdict back to an LLM and get a
refined strategy. Two paths share the same short, sufficient context:

  * improve_strategy(...)    -> calls the configured provider, returns a new
    StrategySpec (then runs through the same pipeline + verification).
  * improve_prompt_text(...) -> the same context as a copy-paste prompt for
    the user's own LLM (returns code to paste into the "Paste my own code" box).

The prompt is kept deliberately tight: current code + the decisive numbers +
the top weaknesses. Nothing the model doesn't need to iterate well.
"""
from __future__ import annotations

from .indicators import INDICATOR_NAMESPACE
from .llm import LLMError, generate_json
from .strategy_generator import (
    STRATEGY_SCHEMA,
    StrategyError,
    StrategySpec,
    _validate_ast,
)

_INDICATORS = ", ".join(sorted(INDICATOR_NAMESPACE))

IMPROVE_SYSTEM = f"""\
You improve an existing vectorized trading strategy. Keep the exact contract:

    def generate_signals(df):
        # df columns: open, high, low, close, volume (lowercase floats)
        return positions  # pandas Series in [-1,1] aligned to df.index

1.0=long, 0=flat, -1.0=short; fractions allowed. Must be causal (no lookahead),
pure, deterministic, NO imports. Globals available: pd, np, and these indicator
helpers: {_INDICATORS}.

Propose ONE improved version that fixes the weaknesses listed. Favour
out-of-sample robustness over in-sample fit — do not add complexity or
parameters unless they clearly earn their keep (overfitting is the enemy).
"""


def _context(spec: StrategySpec, rr) -> str:
    m, comp, split, psr = rr.metrics, rr.comp, rr.split, rr.sharpe_conf
    t = rr.tstats
    weaknesses = [msg for status, msg in rr.quality if status in ("warn", "bad")]
    lines = [
        f"Original idea: {spec.description}",
        "",
        "Current code:",
        "```python",
        spec.code.strip(),
        "```",
        "",
        "Backtest results (after costs):",
        f"- CAGR {m.cagr:+.1%} | Sharpe {m.sharpe:.2f} | Sortino {m.sortino:.2f}",
        f"- Out-of-sample Sharpe {split.oos_metrics.sharpe:.2f} "
        f"(in-sample {split.is_metrics.sharpe:.2f}; "
        f"{'holds up' if split.holds_up else 'DEGRADES out-of-sample'})",
        f"- Max drawdown {m.max_drawdown:.0%} | Calmar {m.calmar:.2f}",
        f"- Alpha vs {comp.benchmark_name} {comp.alpha_annual:+.1%}/yr | "
        f"info ratio {comp.information_ratio:.2f}",
        f"- Trades {m.num_trades} | win rate {m.win_rate:.0%} | "
        f"profit factor {('inf' if m.profit_factor == float('inf') else f'{m.profit_factor:.2f}')}",
        f"- Sharpe confidence (PSR) {psr.psr:.0%}"
        + (f" | per-trade t-stat {t.t_stat:.2f}" if t else ""),
    ]
    if weaknesses:
        lines += ["", "Weaknesses to address:"]
        lines += [f"- {w}" for w in weaknesses[:5]]
    return "\n".join(lines)


def improve_strategy(
    spec: StrategySpec, rr, *, provider: str, model: str, api_key: str | None
) -> StrategySpec:
    """Ask the configured provider for an improved spec. Raises StrategyError
    on failure (caller handles)."""
    try:
        data = generate_json(
            provider=provider,
            model=model,
            system=IMPROVE_SYSTEM,
            user=_context(spec, rr),
            schema=STRATEGY_SCHEMA,
            api_key=api_key,
        )
    except LLMError as exc:
        raise StrategyError(str(exc)) from exc

    try:
        improved = StrategySpec(
            name=data["name"],
            description=data["description"],
            rationale=data["rationale"],
            market_regime=data["market_regime"],
            direction=data["direction"],
            indicators_used=data.get("indicators_used", []),
            code=data["code"].strip(),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise StrategyError(f"Improve response missing fields: {exc}") from exc

    _validate_ast(improved.code)
    return improved


def improve_prompt_text(spec: StrategySpec, rr) -> str:
    """Copy-paste prompt for the user's own LLM. Asks for code only, so the
    reply drops straight into the 'Paste my own code' box."""
    return (
        IMPROVE_SYSTEM
        + "\n"
        + _context(spec, rr)
        + "\n\nReturn ONLY the improved `generate_signals` Python code — no "
        "markdown fences, no commentary."
    )

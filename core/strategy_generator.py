"""Turn a natural-language prompt into an executable trading strategy via the Claude API.

Design notes:
  * The large instruction block is sent as a cache_control'd system block so repeated
    runs in a session only pay full input price once (prefix caching).
  * The strategy is requested as a structured JSON object (output_config.format) so we
    get a typed spec back rather than having to scrape prose.
  * Generated code is AST-validated and executed in a restricted namespace before it is
    ever called. This is defence-in-depth, not a true sandbox — see SECURITY note below.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import INDICATOR_DOCS, INDICATOR_NAMESPACE
from .llm import ANTHROPIC, LLMError, generate_json

DEFAULT_PROVIDER = ANTHROPIC
DEFAULT_MODEL = "claude-opus-4-7"

# --- The strategy contract, shown to the model. Stable => cached. -------------
SYSTEM_PROMPT = f"""\
You are a quantitative trading engineer. You convert a plain-English strategy idea
into a single, deterministic Python function that produces target positions for a
vectorized backtest.

You MUST return a function with exactly this contract:

    def generate_signals(df):
        # df is a pandas DataFrame indexed by timestamp, columns:
        #   open, high, low, close, volume   (all lowercase, float)
        ...
        return positions   # pandas Series aligned to df.index

Rules for `generate_signals`:
  * Return a pandas Series of TARGET positions aligned to df.index, values in the
    closed range [-1.0, 1.0]:
        1.0  = fully long, 0.0 = flat, -1.0 = fully short.
    Fractional values (e.g. 0.5) express partial sizing and are allowed.
  * The position for bar t may only use information available up to and including
    bar t (the backtest engine applies an execution lag, so do NOT shift to "fix"
    lookahead — just compute the desired position from current/past data).
  * It must be pure and deterministic: no randomness, no I/O, no network, no
    imports, no file access, no global mutable state.
  * Handle warmup NaNs: fill non-finite positions with 0.0 before returning.
  * If the idea is long-only, never emit negative positions.

You have these pre-imported names available as globals (do NOT import anything):
  * pd  -> pandas
  * np  -> numpy
  * the following vectorized indicator helpers (all take/return pandas Series):

{INDICATOR_DOCS}

Prefer the provided indicator helpers over hand-rolling them. Keep the function
focused and readable. Do not define classes or extra top-level code beyond helper
functions if strictly needed; the entry point must be `generate_signals`.
"""

# Copy-paste this to your own LLM (ChatGPT, local model, etc.) so its output
# drops straight into the "Paste my own code" box and passes the sandbox.
USER_LLM_PROMPT = (
    SYSTEM_PROMPT
    + """
------------------------------------------------------------
TASK: replace the line below with your trading idea, then send
this whole message to your LLM.

My strategy idea: <describe your idea, e.g. "go long when RSI(2)
< 10 and price is above its 200-day SMA; exit when RSI(2) > 60">

Output rules:
- Return ONLY Python code: the `generate_signals(df)` function
  (plus small helper functions only if truly needed).
- No markdown fences, no comments-as-explanation, no prose before
  or after — just the code, ready to paste.
"""
)

STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "rationale": {"type": "string"},
        "market_regime": {"type": "string"},
        "direction": {"type": "string", "enum": ["long_only", "long_short", "short_only"]},
        "indicators_used": {"type": "array", "items": {"type": "string"}},
        "code": {"type": "string"},
    },
    "required": [
        "name",
        "description",
        "rationale",
        "market_regime",
        "direction",
        "indicators_used",
        "code",
    ],
    "additionalProperties": False,
}


@dataclass
class StrategySpec:
    name: str
    description: str
    rationale: str
    market_regime: str
    direction: str
    indicators_used: list[str]
    code: str


class StrategyError(Exception):
    """Raised when generation, validation, or execution of the strategy fails."""


# --- Sandbox -----------------------------------------------------------------
# SECURITY: generated code runs in this process. The checks below block the
# obvious escape hatches (imports, dunder access, eval/exec/open, ...) but are
# best-effort, not a hardened sandbox. Only run prompts/code you would be
# comfortable executing locally.
_FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "__import__", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "exit", "quit", "help", "breakpoint", "memoryview",
}

_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
    "len": len, "range": range, "enumerate": enumerate, "zip": zip,
    "float": float, "int": int, "bool": bool, "str": str, "list": list,
    "dict": dict, "tuple": tuple, "set": set, "sorted": sorted,
    "map": map, "filter": filter, "any": any, "all": all,
    "isinstance": isinstance, "print": lambda *a, **k: None,
}


def _validate_ast(code: str) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:  # pragma: no cover - defensive
        raise StrategyError(f"Generated code has a syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise StrategyError("Generated code attempts an import (not allowed).")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise StrategyError(f"Dunder attribute access blocked: {node.attr}")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise StrategyError(f"Forbidden name used: {node.id}")

    has_entrypoint = any(
        isinstance(n, ast.FunctionDef) and n.name == "generate_signals"
        for n in tree.body
    )
    if not has_entrypoint:
        raise StrategyError("Generated code does not define generate_signals(df).")


def compile_strategy(code: str):
    """Validate then compile generated code; return the generate_signals callable."""
    _validate_ast(code)
    namespace: dict = {
        "__builtins__": _SAFE_BUILTINS,
        "pd": pd,
        "np": np,
        **INDICATOR_NAMESPACE,
    }
    try:
        exec(compile(code, "<strategy>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # pragma: no cover - defensive
        raise StrategyError(f"Generated code failed to load: {exc}") from exc

    fn = namespace.get("generate_signals")
    if not callable(fn):
        raise StrategyError("generate_signals is not callable.")
    return fn


def normalize_positions(raw, index: pd.Index) -> pd.Series:
    """Coerce a strategy's raw output into a clean position Series:
    aligned to `index`, finite, and clipped to [-1, 1]."""
    pos = pd.Series(raw, index=index, dtype="float64")
    pos = pos.reindex(index)
    pos = pos.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return pos.clip(-1.0, 1.0)


def run_strategy(fn, df: pd.DataFrame) -> pd.Series:
    """Execute a compiled strategy and normalise its output to a clean position Series."""
    try:
        raw = fn(df)
    except Exception as exc:
        raise StrategyError(f"Strategy raised at runtime: {exc}") from exc
    return normalize_positions(raw, df.index)


# --- Generation --------------------------------------------------------------
def _data_context(df: pd.DataFrame) -> str:
    head = df.head(3).round(4).to_string()
    desc = df[["open", "high", "low", "close", "volume"]].describe().round(4).to_string()
    return (
        f"Rows: {len(df)} | Period: {df.index[0]} -> {df.index[-1]}\n"
        f"First rows:\n{head}\n\nSummary statistics:\n{desc}"
    )


def generate_strategy(
    prompt: str,
    df: pd.DataFrame,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
) -> StrategySpec:
    """Ask the selected LLM provider for a strategy spec given the user's idea
    and a data preview."""
    user_content = (
        f"Strategy idea:\n{prompt}\n\n"
        f"Dataset context (for shaping the logic; the full series is backtested):\n"
        f"{_data_context(df)}"
    )

    try:
        data = generate_json(
            provider=provider,
            model=model,
            system=SYSTEM_PROMPT,
            user=user_content,
            schema=STRATEGY_SCHEMA,
            api_key=api_key,
        )
    except LLMError as exc:
        raise StrategyError(str(exc)) from exc

    try:
        spec = StrategySpec(
            name=data["name"],
            description=data["description"],
            rationale=data["rationale"],
            market_regime=data["market_regime"],
            direction=data["direction"],
            indicators_used=data.get("indicators_used", []),
            code=data["code"].strip(),
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise StrategyError(
            f"Model response missing expected fields: {exc}"
        ) from exc

    # Fail fast if the code is structurally invalid before we backtest.
    _validate_ast(spec.code)
    return spec

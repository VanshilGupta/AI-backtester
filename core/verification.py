"""Transparency layer: explicit pass/warn/fail checks the user can see.

Two gates:
  * `verify_data`     - is the loaded/fetched OHLCV sane and usable?
  * `verify_strategy` - did the AI's code compile, run, and behave sensibly
    (right shape, in-range positions, actually trades, no obvious lookahead,
    consistent with its own stated direction)?

Nothing here raises — it returns structured `Check`s so the UI can render a
green/amber/red checklist. Hard structural failures (imports, syntax, runtime
errors) are still caught earlier by the sandbox and surface as StrategyError.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .strategy_generator import StrategySpec, normalize_positions

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str          # pass | warn | fail
    detail: str


def _years(idx: pd.DatetimeIndex) -> float:
    return max((idx[-1] - idx[0]).days / 365.25, 1e-9)


def verify_data(df: pd.DataFrame, source: str) -> list[Check]:
    """Sanity checks on cleaned OHLCV. The loader already rejected the truly
    broken cases; this explains *why* the surviving data is trustworthy."""
    checks: list[Check] = []
    idx = df.index

    checks.append(Check("Source", PASS, source))

    yrs = _years(idx)
    checks.append(
        Check(
            "Coverage", PASS,
            f"{len(df):,} bars from {idx[0].date()} to {idx[-1].date()} "
            f"(~{yrs:.1f} years)",
        )
    )

    mono = idx.is_monotonic_increasing and idx.is_unique
    checks.append(
        Check(
            "Date index",
            PASS if mono else FAIL,
            "Sorted, no duplicate timestamps"
            if mono else "Index is not strictly increasing / has duplicates",
        )
    )

    ohlc = df[["open", "high", "low", "close"]]
    n_nan = int(ohlc.isna().to_numpy().sum())
    checks.append(
        Check(
            "No missing OHLC",
            PASS if n_nan == 0 else FAIL,
            "All open/high/low/close present"
            if n_nan == 0 else f"{n_nan} missing OHLC values remain",
        )
    )

    n_nonpos = int((ohlc <= 0).to_numpy().sum())
    checks.append(
        Check(
            "Positive prices",
            PASS if n_nonpos == 0 else FAIL,
            "All prices > 0"
            if n_nonpos == 0 else f"{n_nonpos} non-positive price values",
        )
    )

    hi_ok = (df["high"] >= df[["open", "close", "low"]].max(axis=1))
    lo_ok = (df["low"] <= df[["open", "close", "high"]].min(axis=1))
    bad = int((~(hi_ok & lo_ok)).sum())
    frac = bad / len(df)
    checks.append(
        Check(
            "OHLC consistency",
            PASS if bad == 0 else (WARN if frac < 0.01 else FAIL),
            "high >= open/close/low and low <= open/close/high on every bar"
            if bad == 0
            else f"{bad} bars ({frac:.2%}) violate high/low bounds",
        )
    )

    has_vol = float(df["volume"].abs().sum()) > 0
    checks.append(
        Check(
            "Volume",
            PASS if has_vol else WARN,
            "Present"
            if has_vol
            else "Absent (synthesised as 0 — volume-based rules will be inert)",
        )
    )

    ret = df["close"].pct_change().abs()
    jumps = int((ret > 0.5).sum())
    checks.append(
        Check(
            "Outlier scan",
            PASS if jumps == 0 else WARN,
            "No single-bar move > 50%"
            if jumps == 0
            else f"{jumps} bar(s) move > 50% — possible split/bad tick "
            f"(max {ret.max():.0%})",
        )
    )
    return checks


def verify_strategy(
    spec: StrategySpec, fn, df: pd.DataFrame
) -> tuple[pd.Series, list[Check]]:
    """Run the compiled strategy once and report behavioural checks.

    Returns (clean_positions, checks). Raising is left to the caller's
    run path; here a runtime error becomes a FAIL check + flat positions.
    """
    checks: list[Check] = []

    checks.append(
        Check(
            "Sandbox validation", PASS,
            "No imports / eval / file or attribute escapes; "
            "`generate_signals(df)` entry point present",
        )
    )

    try:
        raw = fn(df)
        ran = True
    except Exception as exc:
        checks.append(
            Check("Executes on data", FAIL, f"Raised at runtime: {exc}")
        )
        return normalize_positions(0.0, df.index), checks

    checks.append(
        Check("Executes on data", PASS, f"Ran on all {len(df):,} bars without error")
    )

    raw_s = pd.Series(raw, index=df.index, dtype="float64").reindex(df.index)
    n_bad = int(raw_s.isna().sum() + np.isinf(raw_s.to_numpy()).sum())
    n_oor = int(((raw_s.abs() > 1.0) & np.isfinite(raw_s)).sum())
    clean = normalize_positions(raw, df.index)

    if not ran:  # pragma: no cover - defensive
        return clean, checks

    if n_bad == 0 and n_oor == 0:
        checks.append(
            Check("Output values", PASS, "All positions finite and within [-1, 1]")
        )
    else:
        checks.append(
            Check(
                "Output values", WARN,
                f"Sanitised {n_bad} NaN/inf and clipped {n_oor} out-of-range "
                f"value(s) into [-1, 1]",
            )
        )

    changes = int((clean.diff().fillna(clean.iloc[0]) != 0).sum())
    exposure = float((clean != 0).mean())
    if changes == 0:
        checks.append(
            Check(
                "Trading activity", WARN,
                f"Position never changes (constant {clean.iloc[0]:.2f}) — "
                f"no trades will be generated",
            )
        )
    else:
        checks.append(
            Check(
                "Trading activity", PASS,
                f"{changes} position change(s); in market {exposure:.0%} of bars",
            )
        )

    has_long = bool((clean > 0).any())
    has_short = bool((clean < 0).any())
    d = spec.direction
    if d == "long_only" and has_short:
        ds = (WARN, "Spec says long_only but code emits short positions")
    elif d == "short_only" and has_long:
        ds = (WARN, "Spec says short_only but code emits long positions")
    else:
        ds = (PASS, f"Consistent with declared direction ({d})")
    checks.append(Check("Direction vs spec", ds[0], ds[1]))

    code = spec.code.replace(" ", "")
    if ".shift(-" in code:
        look = (WARN, "Code uses a negative .shift() — possible lookahead")
    else:
        look = (
            PASS,
            "No negative shifts; engine also applies execution lag so "
            "current-bar signals can't peek ahead",
        )
    checks.append(Check("Lookahead guard", look[0], look[1]))

    return clean, checks


def summarize(checks: list[Check]) -> str:
    p = sum(c.status == PASS for c in checks)
    w = sum(c.status == WARN for c in checks)
    f = sum(c.status == FAIL for c in checks)
    return f"{p} passed | {w} warning(s) | {f} failed"

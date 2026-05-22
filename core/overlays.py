"""Risk / position-sizing overlays applied to a strategy's target positions.

These sit *between* the signal and the backtest, so they work with any
strategy (AI-generated or pasted) without changing its code:

  * Volatility targeting  -> scale exposure so realised volatility tracks a
    target; trims size in turbulent regimes, adds it in calm ones.
  * Max-drawdown breaker  -> force flat once an intra-run drawdown limit is
    breached, re-enter after partial recovery.

Both are causal (only use information up to the prior bar) and **off by
default**, so the default backtest is byte-for-byte unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OverlayConfig:
    vol_target_enabled: bool = False
    target_ann_vol: float = 0.15      # e.g. 15% annualised
    vol_lookback: int = 20            # bars for realised-vol estimate
    max_leverage: float = 1.0         # cap on |position| after scaling

    dd_breaker_enabled: bool = False
    dd_limit: float = 0.20            # flatten if drawdown exceeds this
    dd_resume_frac: float = 0.5       # re-enter after recovering to half the limit

    @property
    def active(self) -> bool:
        return self.vol_target_enabled or self.dd_breaker_enabled


def _vol_target(
    pos: pd.Series, df: pd.DataFrame, cfg: OverlayConfig, ppy: float
) -> pd.Series:
    asset_ret = df["close"].pct_change()
    realised = asset_ret.rolling(cfg.vol_lookback).std().shift(1) * np.sqrt(ppy)
    scale = (cfg.target_ann_vol / realised).clip(lower=0.0, upper=10.0)
    scale = scale.fillna(1.0)  # warmup: leave size unscaled
    return (pos * scale).clip(-cfg.max_leverage, cfg.max_leverage)


def _dd_breaker(pos: pd.Series, df: pd.DataFrame, cfg: OverlayConfig) -> pd.Series:
    """Sequential, causal: track a lag-1 gross equity curve of the (already
    sized) target and gate exposure to 0 once drawdown breaches the limit."""
    ret = df["close"].pct_change().fillna(0.0).to_numpy()
    tgt = pos.to_numpy(dtype="float64")
    out = np.zeros_like(tgt)
    eq = peak = 1.0
    held = 0.0
    halted = False
    resume_at = -cfg.dd_limit * cfg.dd_resume_frac
    for t in range(tgt.size):
        eq *= 1.0 + held * ret[t]
        peak = max(peak, eq)
        dd = eq / peak - 1.0
        if not halted and dd <= -cfg.dd_limit:
            halted = True
        elif halted and dd >= resume_at:
            halted = False
        desired = 0.0 if halted else tgt[t]
        out[t] = desired
        held = desired
    return pd.Series(out, index=pos.index)


def apply_overlays(
    pos: pd.Series, df: pd.DataFrame, cfg: OverlayConfig, ppy: float
) -> tuple[pd.Series, list[str]]:
    """Return (adjusted_positions, human-readable notes). No-op when inactive."""
    if not cfg.active:
        return pos, []

    notes: list[str] = []
    raw = pos.copy()
    out = pos

    if cfg.vol_target_enabled:
        out = _vol_target(out, df, cfg, ppy)
        active = raw != 0
        if active.any():
            avg_lev = float(out[active].abs().mean())
            notes.append(
                f"Vol targeting @ {cfg.target_ann_vol:.0%}/yr "
                f"({cfg.vol_lookback}-bar): avg exposure {avg_lev:.2f}x."
            )

    if cfg.dd_breaker_enabled:
        before = out.copy()
        out = _dd_breaker(out, df, cfg)
        halted_bars = int(((before != 0) & (out == 0)).sum())
        notes.append(
            f"Max-drawdown breaker @ {cfg.dd_limit:.0%}: flat for "
            f"{halted_bars} bar(s) after breaches."
        )

    return out, notes

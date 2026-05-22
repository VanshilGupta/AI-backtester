"""Strategy run history — a compact, portable research journal.

Each backtest produces one record (key numbers + the code). The list is
JSON-serialisable so the UI can offer a download, and re-uploaded later to
restore or compare past experiments. Deliberately small: no equity series,
just what you need to scan and re-run.
"""
from __future__ import annotations

import datetime as dt
import json

SCHEMA_VERSION = 1


def _f(x) -> float | None:
    try:
        v = float(x)
        return None if v != v else round(v, 6)  # drop NaN
    except (TypeError, ValueError):
        return None


def make_record(rr, *, source_desc: str, settings: dict) -> dict:
    """Build a compact record from a pipeline.RunResult (`rr` must be ok)."""
    m, bm, v = rr.metrics, rr.bench_metrics, rr.verdict
    split, psr, mc = rr.split, rr.sharpe_conf, rr.mc
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "name": rr.spec.name,
        "direction": rr.spec.direction,
        "source": source_desc,
        "benchmark": rr.bench_name,
        "score": _f(v.score),
        "recommendation": v.recommendation,
        "cagr": _f(m.cagr),
        "sharpe": _f(m.sharpe),
        "oos_sharpe": _f(split.oos_metrics.sharpe) if split else None,
        "sharpe_confidence": _f(psr.psr) if psr else None,
        "max_drawdown": _f(m.max_drawdown),
        "alpha_annual": _f(rr.comp.alpha_annual) if rr.comp else None,
        "num_trades": int(m.num_trades),
        "bench_score": _f(rr.bench_verdict.score),
        "bench_cagr": _f(bm.cagr),
        "mc_prob_profit": _f(mc.prob_profit) if mc else None,
        "settings": settings,
        "code": rr.spec.code,
    }


def to_json(records: list[dict]) -> str:
    return json.dumps(
        {"schema": SCHEMA_VERSION, "runs": records}, indent=2, default=str
    )


def from_json(text: str | bytes) -> list[dict]:
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    data = json.loads(text)
    runs = data.get("runs", data) if isinstance(data, dict) else data
    if not isinstance(runs, list):
        raise ValueError("Not a valid history file.")
    return runs


# Columns surfaced in the UI table, in order.
TABLE_COLUMNS = [
    "timestamp", "name", "score", "recommendation", "cagr", "sharpe",
    "oos_sharpe", "sharpe_confidence", "max_drawdown", "alpha_annual",
    "num_trades", "source",
]

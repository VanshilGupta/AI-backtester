"""Streamlit UI: prompt + data (upload or ticker) -> AI strategy -> backtest
-> verdict + post-backtest analysis."""
from __future__ import annotations

import datetime as dt
import os

import plotly.graph_objects as go
import streamlit as st

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
from core.constants import DEFAULT_BENCHMARK, DEFAULTS, EXAMPLE_STRATEGIES, CostModel
from core.data import fetch_ohlcv, load_ohlcv
from core.engine import BacktestConfig, run_backtest
from core.evaluator import attach_llm_assessment, evaluate
from core.llm import DEFAULT_PROVIDER, PROVIDER_ENV_KEYS, PROVIDER_MODELS
from core.metrics import compute_metrics
from core.strategy_generator import (
    USER_LLM_PROMPT,
    StrategyError,
    StrategySpec,
    compile_strategy,
    generate_strategy,
)
from core.verification import summarize, verify_data, verify_strategy

st.set_page_config(
    page_title="AI Strategy Backtester", page_icon="📈", layout="wide"
)

st.markdown(
    """
    <style>
      /* Dark theme aligned with .streamlit/config.toml
         (bg #1d293d, panel #0f172b, border #314158, primary #615fff). */
      .block-container {padding-top: 2.2rem; max-width: 1320px;}
      h1, h2, h3 {letter-spacing: -0.02em;}
      [data-testid="stMetric"] {
        background: #0f172b; border: 1px solid #314158;
        padding: 14px 16px; border-radius: 14px;
      }
      [data-testid="stMetricLabel"] p {color:#94a3b8; font-size:.78rem;}
      .verdict-badge {
        display:inline-block; padding:6px 16px; border-radius:999px;
        font-weight:600; font-size:.95rem;
      }
      .pill {background:#0f172b; color:#cbd5e1; border:1px solid #314158;
             border-radius:999px; padding:3px 10px; font-size:.78rem;
             margin-right:6px;}
      .qa {padding:8px 12px; border-radius:10px; margin-bottom:6px;
           font-size:.9rem; border:1px solid transparent;}
      .qa-good {background:rgba(34,197,94,.14); color:#4ade80;
                border-color:rgba(34,197,94,.35);}
      .qa-warn {background:rgba(234,179,8,.14); color:#fbbf24;
                border-color:rgba(234,179,8,.35);}
      .qa-bad  {background:rgba(239,68,68,.14); color:#f87171;
                border-color:rgba(239,68,68,.35);}
      footer {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# Plain-English, semi-technical explanations shown as ⓘ tooltips next to terms.
GLOSSARY: dict[str, str] = {
    "CAGR": "Compound Annual Growth Rate — the steady yearly % the strategy "
    "would have grown at. 15% means it effectively grew 15%/yr.",
    "Total return": "Total % gain/loss over the whole period (not annualised).",
    "Sharpe": "Return earned per unit of total risk, above the risk-free rate. "
    ">1 is good, >2 strong, <0.5 weak. Higher = smoother gains.",
    "Sortino": "Like Sharpe but only penalises *downside* volatility — upside "
    "swings don't count as 'risk'. >2 is strong.",
    "Max drawdown": "Worst peak-to-trough drop in equity. -30% means at some "
    "point you'd have been down 30% from your high — could you hold through it?",
    "Calmar": "CAGR divided by max drawdown. Return per unit of worst-case "
    "pain. >1 decent, >3 excellent.",
    "Volatility": "Annualised standard deviation of returns — how bumpy the "
    "ride is. Lower is calmer.",
    "Profit factor": "Gross profit ÷ gross loss across trades. >1 makes money; "
    "1.5+ is healthy; ∞ means no losing trades (suspicious — likely overfit).",
    "Win rate": "% of trades that were profitable. Note: a low win rate can "
    "still be very profitable if winners are much bigger than losers.",
    "Trades": "Number of completed round-trip trades. <30 means the stats are "
    "a small, noisy sample — treat with caution.",
    "Exposure": "% of bars the strategy was actually in the market (vs flat in "
    "cash). Very low exposure = a fragile, rarely-triggered edge.",
    "Total costs paid": "Cumulative commissions, taxes and slippage paid, in "
    "currency. High turnover strategies bleed here.",
    "vs benchmark": "Strategy CAGR minus the benchmark's CAGR. Positive = it "
    "added value over simply holding the benchmark.",
    "Alpha": "Annualised return the strategy adds *beyond* what its market "
    "exposure (beta) explains. Positive alpha = genuine skill, not just risk.",
    "Beta": "Sensitivity to the benchmark. 1.0 moves with it, 0.5 half as "
    "much, 0 uncorrelated, <0 moves opposite.",
    "Information ratio": "Consistency of out-performance: excess return ÷ how "
    "much it deviates from the benchmark. >0.5 good, >1 excellent.",
    "Correlation": "How tightly strategy returns track the benchmark "
    "(-1 to +1). Lower = better diversification.",
    "Tracking error": "Annualised volatility of the strategy's return "
    "*difference* vs the benchmark.",
    "Up capture": "Of the benchmark's gains in up periods, what % the strategy "
    "captured. >100% = beats it when markets rise.",
    "Down capture": "Of the benchmark's losses in down periods, what % the "
    "strategy took. <100% = loses less when markets fall (good).",
    "Periods outperformed": "Share of bars where the strategy beat the "
    "benchmark. ~50% is coin-flip; higher is persistent edge.",
    "Expectancy / trade": "Average % outcome per trade (the long-run edge per "
    "bet). Must be positive after costs to be worth trading.",
    "Payoff ratio": "Average winning trade ÷ average losing trade (absolute). "
    ">1 means winners are bigger than losers.",
    "t-stat (edge)": "Statistical significance of the average trade being "
    "non-zero. |t| ≥ 2 ≈ 95% confidence it's real, not luck.",
    "Kelly fraction": "Theoretically optimal capital to risk per trade given "
    "the edge. A rough sizing guide — most use a fraction of it.",
    "Max win streak": "Longest run of consecutive winning trades.",
    "Max loss streak": "Longest run of consecutive losing trades — the "
    "psychological worst case you'd have had to sit through.",
    "Avg bars held": "Average holding period of a trade, in bars.",
    "Score": "Deterministic 0–10 quality score = a weighted blend of Sharpe, "
    "CAGR, drawdown, Calmar, profit factor, excess return and win rate, then "
    "multiplied by a confidence factor for low trade counts.",
    "Confidence": "0–1 multiplier that shrinks the score when there are too "
    "few trades for the statistics to be trustworthy.",
}


def _help(term: str) -> str | None:
    return GLOSSARY.get(term)


def _render_checks(checks) -> None:
    icon = {"pass": "✓", "warn": "!", "bad": "✕", "fail": "✕"}
    cls = {"pass": "good", "warn": "warn", "fail": "bad"}
    for c in checks:
        st.markdown(
            f'<div class="qa qa-{cls[c.status]}">'
            f'<b>{icon[c.status]} {c.name}</b> — {c.detail}</div>',
            unsafe_allow_html=True,
        )


def _badge(rec: str) -> str:
    palette = {
        "Implement": ("#067647", "#ecfdf3"),
        "Implement with caution": ("#b54708", "#fffaeb"),
        "Needs work": ("#b42318", "#fef3f2"),
        "Do not implement": ("#912018", "#fee4e2"),
    }
    fg, bg = palette.get(rec, ("#344054", "#f2f4f7"))
    return (
        f'<span class="verdict-badge" style="color:{fg};background:{bg};">'
        f"{rec}</span>"
    )


_GRID = "#314158"
_AXISLINE = "#475569"


def _dark(fig: go.Figure, **extra) -> go.Figure:
    """Apply the dark theme (transparent bg so the panel shows through,
    light font, subtle slate gridlines) to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Space Grotesk, sans-serif"),
        **extra,
    )
    fig.update_xaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_AXISLINE)
    fig.update_yaxes(gridcolor=_GRID, zerolinecolor=_GRID, linecolor=_AXISLINE)
    return fig


def _gauge(score: float) -> go.Figure:
    color = (
        "#067647" if score >= 7.5
        else "#dc6803" if score >= 6.0
        else "#b42318" if score >= 4.0
        else "#912018"
    )
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": " / 10", "font": {"size": 34,
                    "color": "#e2e8f0"}},
            gauge={
                "axis": {"range": [0, 10], "tickwidth": 1,
                         "tickcolor": "#94a3b8"},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": _GRID,
                "steps": [
                    {"range": [0, 4], "color": "rgba(239,68,68,.20)"},
                    {"range": [4, 6], "color": "rgba(234,179,8,.20)"},
                    {"range": [6, 7.5], "color": "rgba(234,179,8,.20)"},
                    {"range": [7.5, 10], "color": "rgba(34,197,94,.22)"},
                ],
            },
        )
    )
    return _dark(fig, height=240, margin=dict(l=20, r=20, t=10, b=10))


def _equity_chart(result, bench_result, bench_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.equity.index, y=result.equity.values,
            name="Strategy", line=dict(color="#615fff", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench_result.equity.index, y=bench_result.equity.values,
            name=bench_name, line=dict(color="#98a2b3", width=1.5, dash="dot"),
        )
    )
    return _dark(
        fig,
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.12, x=0),
        yaxis_title="Equity", hovermode="x unified",
    )


def _drawdown_chart(result) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=result.drawdown.index, y=result.drawdown.values * 100,
            fill="tozeroy", line=dict(color="#f04438", width=1),
            name="Drawdown",
        )
    )
    return _dark(
        fig,
        height=240, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Drawdown %", hovermode="x unified",
    )


def _rolling_sharpe_chart(series) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=series.index, y=series.values,
            line=dict(color="#7a5af8", width=1.5), name="Rolling Sharpe",
        )
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#64748b")
    fig.add_hline(y=0.0, line_color="#475569")
    return _dark(
        fig,
        height=240, margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title="Rolling Sharpe", hovermode="x unified",
    )


def _returns_hist(net_returns) -> go.Figure:
    nz = net_returns[net_returns != 0.0] * 100.0
    fig = go.Figure(go.Histogram(x=nz.values, nbinsx=60, marker_color="#615fff"))
    return _dark(
        fig,
        height=240, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Per-bar return %", yaxis_title="Count",
    )


def _monthly_heatmap(table) -> go.Figure:
    months = [c for c in table.columns if c != "Year"]
    z = table[months].values
    fig = go.Figure(
        go.Heatmap(
            z=z, x=months, y=[str(i) for i in table.index],
            colorscale="RdYlGn", zmid=0,
            text=z, texttemplate="%{text:.1f}", textfont={"size": 9},
            colorbar=dict(title="%"),
        )
    )
    return _dark(
        fig,
        height=max(220, 26 * len(table) + 90),
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_autorange="reversed",
    )


st.title("📈 AI Strategy Backtester")
st.caption(
    "Describe a trading idea, give it data (upload a CSV or fetch by ticker). "
    "Claude or Gemini writes the strategy, it's backtested with realistic "
    "costs, scored out of 10, benchmarked, and analysed."
)

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "Provider",
        list(PROVIDER_MODELS.keys()),
        index=list(PROVIDER_MODELS.keys()).index(DEFAULT_PROVIDER),
        help="Which LLM writes and reviews the strategy.",
    )
    model = st.selectbox(
        "Model",
        PROVIDER_MODELS[provider],
        index=0,
        help="First option is the strongest; the second is faster/cheaper.",
    )
    _env_default = next(
        (os.environ.get(k, "") for k in PROVIDER_ENV_KEYS[provider]
         if os.environ.get(k)),
        "",
    )
    api_key = st.text_input(
        f"{provider} API key",
        value=_env_default,
        type="password",
        help="Used only for this session. Defaults to the provider's env var "
        "(ANTHROPIC_API_KEY, or GEMINI_API_KEY / GOOGLE_API_KEY).",
    )
    st.divider()
    capital = st.number_input(
        "Initial capital", min_value=1000.0,
        value=float(DEFAULTS.initial_capital), step=1000.0,
    )
    lag = st.number_input(
        "Execution lag (bars)", min_value=0, max_value=5,
        value=int(DEFAULTS.execution_lag),
        help="Bars between signal and fill. 1 = no lookahead.",
    )
    risk_free = st.number_input(
        "Risk-free rate (annual %)", min_value=0.0, max_value=20.0,
        value=float(DEFAULTS.risk_free_rate * 100), step=0.25,
        help="Used for excess-return Sharpe/Sortino and CAPM alpha.",
    ) / 100.0
    benchmark_name = st.selectbox(
        "Benchmark", list(BENCHMARKS.keys()),
        index=list(BENCHMARKS.keys()).index(DEFAULT_BENCHMARK),
        help="Your strategy is compared against this, run through the same "
        "engine and costs for a fair comparison.",
    )

    st.divider()
    st.subheader("Charges (per side, bps)")
    st.caption("Defaults = realistic NSE delivery. Edit for your market/segment.")
    d = DEFAULTS.cost_model
    cost_model = CostModel(
        brokerage_bps=st.number_input("Brokerage", 0.0, value=d.brokerage_bps, step=0.5),
        stt_bps=st.number_input("STT / CTT", 0.0, value=d.stt_bps, step=0.5),
        exchange_txn_bps=st.number_input(
            "Exchange txn", 0.0, value=d.exchange_txn_bps, step=0.01, format="%.4f"
        ),
        sebi_bps=st.number_input(
            "SEBI", 0.0, value=d.sebi_bps, step=0.01, format="%.4f"
        ),
        stamp_duty_bps=st.number_input("Stamp duty", 0.0, value=d.stamp_duty_bps, step=0.1),
        gst_pct=st.number_input("GST %", 0.0, value=d.gst_pct, step=1.0),
        slippage_bps=st.number_input("Slippage", 0.0, value=d.slippage_bps, step=0.5),
    )
    st.info(
        f"Effective **{cost_model.effective_bps_per_turnover():.2f} bps/side** "
        f"→ round trip ≈ **{cost_model.round_trip_pct():.3f}%** of notional."
    )

# --- Data source ------------------------------------------------------------
st.markdown("#### 1 · Data")
src = st.radio(
    "Data source", ["Upload CSV", "Fetch by ticker"],
    horizontal=True, label_visibility="collapsed",
)
data_args: dict = {}
if src == "Upload CSV":
    uploaded = st.file_uploader("OHLCV dataset (CSV)", type=["csv"])
    data_args = {"mode": "csv", "uploaded": uploaded}
else:
    c = st.columns([1.2, 1, 1, 1])
    ticker = c[0].text_input("Ticker", placeholder="RELIANCE.NS / AAPL / BTC-USD")
    start = c[1].date_input("Start", value=dt.date.today() - dt.timedelta(days=365 * 8))
    end = c[2].date_input("End", value=dt.date.today())
    interval = c[3].selectbox("Interval", ["1d", "1wk", "1h", "1mo"], index=0)
    st.caption(
        "NSE needs a `.NS` suffix (RELIANCE.NS), indices use `^` (^NSEI, ^GSPC), "
        "crypto is `BTC-USD`. Data via Yahoo Finance."
    )
    data_args = {
        "mode": "ticker", "ticker": ticker,
        "start": start, "end": end, "interval": interval,
    }

# --- Strategy idea ----------------------------------------------------------
st.markdown("#### 2 · Strategy")
strat_mode = st.radio(
    "Strategy source",
    ["Describe it (AI writes the code)", "Paste my own code"],
    horizontal=True, label_visibility="collapsed",
)

prompt = ""
pasted_code = ""
custom_name = "Custom strategy"
custom_direction = "long_only"

if strat_mode == "Describe it (AI writes the code)":
    labels = ["— write my own —"] + [e[0] for e in EXAMPLE_STRATEGIES]
    choice = st.selectbox("Start from an example", labels)
    preset = ""
    if choice != labels[0]:
        ex = next(e for e in EXAMPLE_STRATEGIES if e[0] == choice)
        preset = ex[1]
        st.caption(f"📂 Suggested data: {ex[2]}")
    prompt = st.text_area(
        "Describe the strategy",
        value=preset,
        height=120,
        placeholder="e.g. Go long when the 50-day SMA crosses above the "
        "200-day SMA and RSI(14) is below 70; exit on the reverse cross.",
    )
else:
    ex_labels = ["— blank / my own —"] + [e[0] for e in EXAMPLE_STRATEGIES]
    ex_choice = st.selectbox(
        "Prefill with an example strategy's code",
        ex_labels,
        help="Pick one to auto-fill its working code below — edit it freely, "
        "then backtest. Choosing an example also sets the name & direction.",
    )
    seed_code, seed_name, dir_idx = "", "Custom strategy", 0
    if ex_choice != ex_labels[0]:
        ex = next(e for e in EXAMPLE_STRATEGIES if e[0] == ex_choice)
        seed_code, seed_name = ex[3], ex[0]
        st.caption(f"📂 Suggested data: {ex[2]}")

    with st.expander(
        "📋 Prompt for your own LLM — copy this, fill in your idea, paste the "
        "code back below"
    ):
        st.caption(
            "Send this to ChatGPT / your local model. It returns code in the "
            "exact format the sandbox expects."
        )
        st.code(USER_LLM_PROMPT, language="text")
    cc = st.columns([2, 1])
    custom_name = cc[0].text_input("Strategy name", value=seed_name)
    custom_direction = cc[1].selectbox(
        "Direction", ["long_only", "long_short", "short_only"],
        index=dir_idx,
        help="Used by the direction-vs-code verification check.",
    )
    pasted_code = st.text_area(
        "Paste your generate_signals(df) code",
        value=seed_code,
        height=300,
        key=f"code_{ex_choice}",
        placeholder=(
            "def generate_signals(df):\n"
            "    fast = sma(df['close'], 50)\n"
            "    slow = sma(df['close'], 200)\n"
            "    pos = pd.Series(0.0, index=df.index)\n"
            "    pos[fast > slow] = 1.0\n"
            "    return pos.fillna(0.0)"
        ),
    )

_btn = (
    "Generate & Backtest"
    if strat_mode == "Describe it (AI writes the code)"
    else "Backtest my code"
)
run = st.button(_btn, type="primary", use_container_width=True)

if run:
    use_ai = strat_mode == "Describe it (AI writes the code)"
    if use_ai:
        if not prompt.strip():
            st.error("Describe the strategy idea.")
            st.stop()
        if not api_key.strip():
            st.error(f"Provide a {provider} API key in the sidebar.")
            st.stop()
    else:
        if not pasted_code.strip():
            st.error("Paste your generate_signals(df) code.")
            st.stop()

    try:
        if data_args["mode"] == "csv":
            if data_args["uploaded"] is None:
                st.error("Upload a CSV dataset first.")
                st.stop()
            df = load_ohlcv(data_args["uploaded"])
            source_desc = f"Uploaded CSV · {data_args['uploaded'].name}"
        else:
            if not data_args["ticker"].strip():
                st.error("Enter a ticker symbol.")
                st.stop()
            with st.spinner(f"Fetching {data_args['ticker']}…"):
                df = fetch_ohlcv(
                    data_args["ticker"], data_args["start"],
                    data_args["end"], data_args["interval"],
                )
            source_desc = (
                f"Yahoo Finance · {data_args['ticker'].strip().upper()} · "
                f"{data_args['interval']} · requested "
                f"{data_args['start']} → {data_args['end']}"
            )
        fetched_at = dt.datetime.now()
    except ValueError as exc:
        st.error(f"Dataset problem: {exc}")
        st.stop()

    data_checks = verify_data(df, source_desc)
    if any(c.status == "fail" for c in data_checks):
        st.markdown("#### Data verification")
        _render_checks(data_checks)
        st.error("Data failed verification — fix the dataset and retry.")
        st.stop()

    cfg = BacktestConfig(
        initial_capital=capital,
        execution_lag=int(lag),
        cost_model=cost_model,
    )
    try:
        if use_ai:
            with st.spinner(f"{provider} is designing the strategy…"):
                spec = generate_strategy(
                    prompt, df, provider=provider, model=model, api_key=api_key
                )
        else:
            spec = StrategySpec(
                name=custom_name.strip() or "Custom strategy",
                description="User-supplied strategy code (own LLM).",
                rationale="Code pasted by the user; not generated here.",
                market_regime="unspecified",
                direction=custom_direction,
                indicators_used=[],
                code=pasted_code.strip(),
            )
        with st.spinner("Verifying & backtesting…"):
            fn = compile_strategy(spec.code)
            positions, strat_checks = verify_strategy(spec, fn, df)
            if any(c.status == "fail" for c in strat_checks):
                st.markdown("#### Strategy code verification")
                _render_checks(strat_checks)
                st.error(
                    "The code failed verification — "
                    + ("rephrase the idea or regenerate."
                       if use_ai else "fix your pasted code and retry.")
                )
                st.stop()
            result = run_backtest(df, positions, cfg)
            bench_positions = BENCHMARKS[benchmark_name](df)
            bench_result = run_backtest(df, bench_positions, cfg)
            metrics = compute_metrics(result, risk_free_rate=risk_free)
            bench_metrics = compute_metrics(bench_result, risk_free_rate=risk_free)
            verdict = evaluate(metrics)
            bench_verdict = evaluate(bench_metrics)
        if api_key.strip():
            with st.spinner("Writing the verdict…"):
                verdict = attach_llm_assessment(
                    verdict, spec, metrics,
                    provider=provider, model=model, api_key=api_key,
                )
    except StrategyError as exc:
        st.error(f"Strategy generation/execution failed: {exc}")
        st.stop()

    comp = benchmark_comparison(result, bench_result, benchmark_name, risk_free)
    tstats = trade_stats(result)
    m_table = monthly_returns_table(result.net_returns)
    dd_table = drawdown_table(result.equity)
    quality = quality_report(metrics, tstats, comp)
    llm_report = format_llm_report(
        spec=spec, config=cfg, source_desc=source_desc, df=df,
        metrics=metrics, bench_metrics=bench_metrics,
        benchmark_name=benchmark_name, verdict=verdict, comp=comp,
        tstats=tstats, dd_table=dd_table, monthly_table=m_table,
        quality=quality,
    )
    analyzed_at = dt.datetime.now()

    st.session_state["result"] = {
        "spec": spec, "result": result, "bench_result": bench_result,
        "bench_name": benchmark_name, "metrics": metrics,
        "bench_metrics": bench_metrics, "verdict": verdict,
        "bench_verdict": bench_verdict, "rf_used": risk_free,
        "data_checks": data_checks, "strat_checks": strat_checks,
        "source_desc": source_desc, "fetched_at": fetched_at,
        "analyzed_at": analyzed_at, "n_bars": len(df),
        "date_span": (df.index[0].date(), df.index[-1].date()),
        "comp": comp, "tstats": tstats, "m_table": m_table,
        "dd_table": dd_table, "quality": quality, "llm_report": llm_report,
    }

if "result" in st.session_state:
    R = st.session_state["result"]
    spec, result, bench_result = R["spec"], R["result"], R["bench_result"]
    bench_name, metrics, verdict = R["bench_name"], R["metrics"], R["verdict"]
    bench_metrics, bench_verdict = R["bench_metrics"], R["bench_verdict"]
    rf_used, data_checks, strat_checks = (
        R["rf_used"], R["data_checks"], R["strat_checks"]
    )
    comp, tstats = R["comp"], R["tstats"]
    m_table, dd_table, quality = R["m_table"], R["dd_table"], R["quality"]

    st.divider()
    _f = R["fetched_at"].strftime("%Y-%m-%d %H:%M:%S")
    _a = R["analyzed_at"].strftime("%Y-%m-%d %H:%M:%S")
    _d0, _d1 = R["date_span"]

    # --- Always-visible snapshot: decide at a glance -----------------------
    st.subheader(spec.name)
    st.write(spec.description)
    st.markdown(
        f'<span class="pill">{spec.direction}</span>'
        f'<span class="pill">regime: {spec.market_regime}</span>'
        + "".join(f'<span class="pill">{i}</span>' for i in spec.indicators_used),
        unsafe_allow_html=True,
    )

    st.markdown("##### 🎯 Quick read — the deciding numbers")
    st.markdown(_badge(verdict.recommendation), unsafe_allow_html=True)
    q = st.columns(5)
    q[0].metric(
        "Score", f"{verdict.score}/10",
        delta=f"{verdict.score - bench_verdict.score:+.1f} vs bench",
        help=_help("Score"),
    )
    q[1].metric(
        "CAGR", f"{metrics.cagr:.1%}",
        delta=f"{metrics.excess_cagr:+.1%} vs bench", help=_help("CAGR"),
    )
    q[2].metric(
        "Sharpe", f"{metrics.sharpe:.2f}",
        delta=f"{metrics.sharpe - bench_metrics.sharpe:+.2f} vs bench",
        help=_help("Sharpe"),
    )
    q[3].metric(
        "Max drawdown", f"{metrics.max_drawdown:.1%}",
        delta=f"{metrics.max_drawdown - bench_metrics.max_drawdown:+.1%} vs bench",
        help=_help("Max drawdown"),
    )
    q[4].metric(
        "Alpha (annual)", f"{comp.alpha_annual:+.1%}",
        delta=f"t-stat {tstats.t_stat:.1f}" if tstats else None,
        delta_color="off", help=_help("Alpha"),
    )
    st.caption(
        f"**Benchmark — {bench_name}** (same costs & period): "
        f"Score {bench_verdict.score}/10 · CAGR {bench_metrics.cagr:.1%} · "
        f"Sharpe {bench_metrics.sharpe:.2f} · "
        f"Max DD {bench_metrics.max_drawdown:.1%}. &nbsp; "
        f"→ *Recommendation: {verdict.recommendation}.* If it's not decent, "
        f"you can stop here; otherwise open the sections below for the full "
        f"analysis."
    )

    st.divider()
    st.caption(
        "All details are collapsed — expand only what you need. Verification "
        "and analysis are optional reading."
    )

    # --- Everything else: sibling, collapsible accordions ------------------
    with st.expander("🟢 Run status — data fetched & analysis timestamps"):
        st.success(
            f"✅ **Data fetched successfully** — {R['source_desc']} · "
            f"{R['n_bars']:,} bars ({_d0} → {_d1}) · at **{_f}**\n\n"
            f"✅ **Analysis completed successfully** — at **{_a}**"
        )
        st.caption(
            "Every click of the button re-fetches data with the current "
            "sidebar/ticker settings, so these timestamps always reflect "
            "this run."
        )

    with st.expander("🔍 Verification — data & code checks (optional)"):
        vc1, vc2 = st.columns(2, gap="large")
        with vc1:
            st.markdown(f"**Data** · {summarize(data_checks)}")
            _render_checks(data_checks)
        with vc2:
            st.markdown(f"**Strategy code** · {summarize(strat_checks)}")
            _render_checks(strat_checks)

    with st.expander("📊 Performance — full metrics & charts"):
        st.caption("Hover the ⓘ on any metric for a plain-English explanation.")
        c = st.columns(4)
        c[0].metric("CAGR", f"{metrics.cagr:.1%}", help=_help("CAGR"))
        c[1].metric("Total return", f"{metrics.total_return:.1%}",
                    help=_help("Total return"))
        c[2].metric("Sharpe", f"{metrics.sharpe:.2f}", help=_help("Sharpe"))
        c[3].metric("Sortino", f"{metrics.sortino:.2f}", help=_help("Sortino"))
        c = st.columns(4)
        c[0].metric("Max drawdown", f"{metrics.max_drawdown:.1%}",
                    help=_help("Max drawdown"))
        c[1].metric("Calmar", f"{metrics.calmar:.2f}", help=_help("Calmar"))
        c[2].metric("Volatility", f"{metrics.volatility:.1%}",
                    help=_help("Volatility"))
        c[3].metric(f"vs {bench_name}", f"{metrics.excess_cagr:+.1%}",
                    help=_help("vs benchmark"))
        c = st.columns(4)
        pf = metrics.profit_factor
        c[0].metric("Win rate", f"{metrics.win_rate:.0%}",
                    help=_help("Win rate"))
        c[1].metric("Profit factor",
                    "∞" if pf == float("inf") else f"{pf:.2f}",
                    help=_help("Profit factor"))
        c[2].metric("Trades", f"{metrics.num_trades}", help=_help("Trades"))
        c[3].metric("Total costs paid", f"{result.total_costs:,.0f}",
                    help=_help("Total costs paid"))

        bpf = bench_metrics.profit_factor
        st.markdown(
            f'<div style="color:#98a2b3;font-size:.82rem;margin-top:6px;">'
            f'<b>Benchmark — {bench_name}</b> (same costs &amp; period): '
            f'CAGR {bench_metrics.cagr:.1%} · '
            f'Sharpe {bench_metrics.sharpe:.2f} · '
            f'Sortino {bench_metrics.sortino:.2f} · '
            f'Max DD {bench_metrics.max_drawdown:.1%} · '
            f'Calmar {bench_metrics.calmar:.2f} · '
            f'Vol {bench_metrics.volatility:.1%} · '
            f'Win {bench_metrics.win_rate:.0%} · '
            f'PF {"∞" if bpf == float("inf") else f"{bpf:.2f}"} · '
            f'Trades {bench_metrics.num_trades} · '
            f'Score <b>{bench_verdict.score}/10</b> '
            f'({bench_verdict.recommendation})</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            _equity_chart(result, bench_result, bench_name),
            use_container_width=True,
        )
        st.plotly_chart(_drawdown_chart(result), use_container_width=True)

    with st.expander("⚖️ Verdict — score gauge, narrative & risk flags"):
        v1, v2 = st.columns([1, 1.6], gap="large")
        with v1:
            st.plotly_chart(_gauge(verdict.score), use_container_width=True)
            st.markdown(_badge(verdict.recommendation),
                        unsafe_allow_html=True)
            st.caption(
                f"Raw {verdict.raw_score}/10 × confidence "
                f"{verdict.confidence:.0%} (trade-count adjusted) = "
                f"{verdict.score}/10"
            )
        with v2:
            if verdict.llm_assessment:
                st.write(verdict.llm_assessment)
            if verdict.risk_flags:
                st.markdown("**Risk flags**")
                for flag in verdict.risk_flags:
                    st.markdown(f"- {flag}")

    with st.expander("🧪 Post-backtest analysis — is the edge real?"):
        st.markdown("**Is this strategy actually good?**")
        for status, msg in quality:
            st.markdown(
                f'<div class="qa qa-{status}">{msg}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"**Versus {bench_name} (fair, same costs)**")
        bc = st.columns(4)
        bc[0].metric("Alpha (annual)", f"{comp.alpha_annual:+.1%}",
                     help=_help("Alpha"))
        bc[1].metric("Beta", f"{comp.beta:.2f}", help=_help("Beta"))
        bc[2].metric("Information ratio", f"{comp.information_ratio:.2f}",
                     help=_help("Information ratio"))
        bc[3].metric("Correlation", f"{comp.correlation:.2f}",
                     help=_help("Correlation"))
        bc = st.columns(4)
        bc[0].metric("Tracking error", f"{comp.tracking_error:.1%}",
                     help=_help("Tracking error"))
        bc[1].metric("Up capture", f"{comp.up_capture:.0%}",
                     help=_help("Up capture"))
        bc[2].metric("Down capture", f"{comp.down_capture:.0%}",
                     help=_help("Down capture"))
        bc[3].metric("Periods outperformed",
                     f"{comp.pct_periods_outperformed:.0%}",
                     help=_help("Periods outperformed"))

        if tstats is not None:
            st.markdown("**Trade statistics**")
            tc = st.columns(4)
            tc[0].metric("Expectancy / trade", f"{tstats.expectancy:.2%}",
                         help=_help("Expectancy / trade"))
            tc[1].metric(
                "Payoff ratio",
                "∞" if tstats.payoff_ratio == float("inf")
                else f"{tstats.payoff_ratio:.2f}",
                help=_help("Payoff ratio"),
            )
            tc[2].metric("t-stat (edge)", f"{tstats.t_stat:.2f}",
                         help=_help("t-stat (edge)"))
            tc[3].metric("Kelly fraction", f"{tstats.kelly_fraction:.0%}",
                         help=_help("Kelly fraction"))
            tc = st.columns(4)
            tc[0].metric("Max win streak", f"{tstats.max_win_streak}",
                         help=_help("Max win streak"))
            tc[1].metric("Max loss streak", f"{tstats.max_loss_streak}",
                         help=_help("Max loss streak"))
            tc[2].metric("Avg bars held", f"{tstats.avg_bars_held:.1f}",
                         help=_help("Avg bars held"))
            tc[3].metric(
                "Profit factor",
                "∞" if tstats.profit_factor == float("inf")
                else f"{tstats.profit_factor:.2f}",
                help=_help("Profit factor"),
            )

        g1, g2 = st.columns([1.4, 1], gap="large")
        with g1:
            st.markdown("**Monthly returns (%)**")
            st.plotly_chart(_monthly_heatmap(m_table),
                            use_container_width=True)
        with g2:
            st.markdown("**Rolling Sharpe**")
            st.plotly_chart(
                _rolling_sharpe_chart(
                    rolling_sharpe(result.net_returns,
                                   result.periods_per_year)
                ),
                use_container_width=True,
            )

        g1, g2 = st.columns([1, 1], gap="large")
        with g1:
            st.markdown("**Return distribution**")
            st.plotly_chart(_returns_hist(result.net_returns),
                            use_container_width=True)
            dist = return_distribution(result.net_returns)
            if dist:
                st.caption(
                    f"skew {dist['skew']} · excess kurtosis "
                    f"{dist['excess_kurtosis']} · 95% VaR {dist['var_95_%']}% "
                    f"· 95% CVaR {dist['cvar_95_%']}% per bar"
                )
        with g2:
            st.markdown("**Worst drawdowns**")
            st.dataframe(dd_table, use_container_width=True, hide_index=True)

    with st.expander("📖 Glossary — what every term means (plain English)"):
        for term, desc in GLOSSARY.items():
            st.markdown(f"- **{term}** — {desc}")

    with st.expander("🧮 Score breakdown — how the 0–10 was built"):
        st.dataframe(
            {
                "Component": list(verdict.components.keys()),
                "Score /10": [v["score"] for v in verdict.components.values()],
                "Weight": [v["weight"] for v in verdict.components.values()],
            },
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("🧾 Generated strategy code"):
        st.code(spec.code, language="python")
        st.caption(f"Rationale: {spec.rationale}")

    if len(result.trades):
        with st.expander(f"📒 Trade log ({len(result.trades)} trades)"):
            st.dataframe(result.trades, use_container_width=True,
                         hide_index=True)

    with st.expander("📤 Hand this run to your LLM (copy-paste)"):
        st.caption(
            "A compact summary of this run (settings, strategy code, metrics, "
            "benchmark, verdict, diagnostics — no raw price series). Paste it "
            "into any LLM and ask it to critique and improve the strategy."
        )
        st.code(R["llm_report"], language="text")

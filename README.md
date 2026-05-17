# AI Strategy Backtester

Describe a trading idea in plain English, give it data (upload a CSV **or**
fetch by ticker), and your chosen LLM (Claude or Gemini) writes the strategy.
It's backtested with realistic costs, scored out of 10, compared against a
benchmark, and put through a full post-backtest analysis so you can judge
whether the edge is real.

---

## 1. Running the app

From a terminal in the project folder:

```powershell
cd C:\Personal\OtherProjects\backtester
python -m streamlit run app.py
```

Streamlit prints a **Local URL** (default `http://localhost:8501`). Open that in
your browser. The app stays running in the terminal — press `Ctrl+C` to stop it.

> **Do NOT run `python app.py`.** That runs the script outside Streamlit and you
> get `missing ScriptRunContext! ... bare mode.` with no UI. Always launch with
> `python -m streamlit run app.py`.

First time only — install dependencies:

```powershell
python -m pip install -r requirements.txt
```

> **Note:** `pip` and `streamlit` are not on this machine's PATH, so always run
> them through Python: `python -m pip ...` and `python -m streamlit run app.py`.

Offline sanity check (no API key, no network):

```powershell
python smoke_test.py
```

It should end with `ALL CHECKS PASSED`.

---

## 2. Provider & API key

The sidebar **"Provider"** dropdown lets you pick who writes (and reviews) the
strategy:

- **Anthropic (Claude)** — key starts with `sk-ant-...`, from
  https://console.anthropic.com
- **Google (Gemini)** — key from https://aistudio.google.com/apikey

The API-key box relabels itself to the chosen provider. Two ways to provide
the key:

- **In the app (easiest):** paste it into the API-key box. Used only for that
  session, never saved.
- **As an environment variable** (the box then auto-fills for that provider):

  ```powershell
  # Anthropic
  $env:ANTHROPIC_API_KEY = "sk-ant-..."   # current terminal only
  setx ANTHROPIC_API_KEY "sk-ant-..."     # permanent (reopen terminal after)

  # Google Gemini (either name works)
  $env:GEMINI_API_KEY = "..."
  setx GEMINI_API_KEY "..."
  ```

You only need a key for the provider you actually select.

---

## 3. Giving it data — two ways

In the app, section **"1 · Data"** has a toggle:

- **Upload CSV** — any OHLCV file. Columns are auto-detected
  (Date/Open/High/Low/Close/Volume and common aliases). Volume optional.
- **Fetch by ticker** — just type a symbol and pick a date range + interval;
  data is pulled from Yahoo Finance. No file needed.

Ticker conventions:

| Market | Example |
|---|---|
| US stocks / ETFs | `AAPL`, `SPY`, `QQQ`, `TSLA` |
| NSE (India) | `RELIANCE.NS`, `INFY.NS`, `NIFTYBEES.NS` |
| BSE (India) | `500325.BO` |
| Indices | `^NSEI` (NIFTY 50), `^GSPC` (S&P 500) |
| Crypto | `BTC-USD`, `ETH-USD` |

---

## 4. Two ways to get the strategy

Section **"2 · Strategy"** has a toggle:

- **Describe it (AI writes the code)** — type an idea (or pick an example);
  the selected provider writes the `generate_signals` function. Needs an API
  key.
- **Paste my own code** — bring code from your *own* LLM (ChatGPT, a local
  model, etc.) or hand-written. No API key required to backtest it.

When you choose **Paste my own code**, expand **"📋 Prompt for your own LLM"**
and copy the ready-made instruction block. It already contains the exact
function contract and the list of available indicators, so whatever your LLM
returns drops straight into the paste box and passes the sandbox. Just replace
the `My strategy idea: <...>` line with your idea before sending it.

Then set a name + direction (the direction is used by the verification check),
paste the code, and hit **Backtest my code**. It goes through the *same*
verification, costs, benchmark, scoring, and analysis as AI-generated code —
so this doubles as a way to validate your own model's output.

In **Paste my own code** there's also a **"Prefill with an example
strategy's code"** dropdown — pick any of the 4 bundled strategies (SMA
golden cross, RSI(2) mean reversion, Bollinger breakout, dual-momentum 12-1)
and its working code auto-fills the box (name & direction set too). Edit it
freely, then backtest.

> The code must define `def generate_signals(df):` returning a position Series
> in `[-1, 1]`, may use the built-in indicators (`sma`, `rsi`, …), and must not
> import anything or do I/O — the sandbox enforces this and the Verification
> panel will tell you exactly what's wrong if it doesn't.

---

## 5. Choosing the model

The **"Model"** dropdown depends on the selected provider:

| Provider | Models (strongest → faster/cheaper) |
|---|---|
| Anthropic (Claude) | `claude-opus-4-7`, `claude-sonnet-4-6` |
| Google (Gemini) | `gemini-2.5-pro`, `gemini-2.5-flash` |

To change defaults or add/remove models, edit `PROVIDER_MODELS` in
[`core/llm.py`](core/llm.py) — the sidebar reads its options from there.

---

## 6. Costs & charges — the constants file

All trading frictions live in **[`core/constants.py`](core/constants.py)** in
one `CostModel` (brokerage, STT/CTT, exchange txn, SEBI, stamp duty, GST,
slippage — all in basis points per side). Defaults are a realistic, slightly
conservative **NSE delivery** profile.

- **They're shown and editable by default** in the sidebar under
  **"Charges (per side, bps)"**. The app shows the effective cost per side and
  the round-trip cost as a % of notional, recomputed live.
- To change the **defaults** (what the sidebar starts with), edit the
  `CostModel` field values in `core/constants.py`. Adjust for your segment —
  intraday, F&O, US equities (much lower), crypto, etc.

Also in `constants.py`: initial capital, execution lag, risk-free rate (used
for excess-return Sharpe/Sortino and CAPM alpha), the default benchmark, and
the starter example strategies — all editable in one place.

---

## 7. Benchmark & post-backtest analysis

Every run also backtests a **benchmark** through the *same engine and costs*
(fair comparison, not a costless ideal). Pick it in the sidebar:
**Buy & Hold** (default), **SMA 200 Trend**, or **50% Invested**. The
benchmark's **own standalone results and 0–10 score** are shown in a light
line right under the strategy's metrics — not just the comparison.

Each result also opens with a green banner: **"Data fetched successfully"**
and **"Analysis completed successfully"**, each with a timestamp and the exact
source/date-range used. Every click of the button re-fetches with the current
settings, so the timestamps always prove the data is fresh for that run.

Every metric has an **ⓘ tooltip** with a plain-English, semi-technical
explanation (hover it), plus a full **"What do all these terms mean?"**
glossary expander.

At the very bottom, **"📤 Hand this run to your LLM"** gives a compact,
copy-pasteable text dump (settings, strategy code, strategy + benchmark
metrics, verdict, diagnostics — no raw price series) you can paste into any
LLM and ask it to critique and improve the strategy.

The **Post-backtest analysis** section answers "is this actually good?":

- **Plain-English quality report** — colour-coded good/warn/bad calls on
  Sharpe, alpha vs benchmark, statistical significance (t-stat), sample size,
  drawdown, and overfitting tells.
- **Benchmark comparison** — alpha (annualised), beta, information ratio,
  correlation, tracking error, up/down capture, % of periods outperformed.
- **Trade statistics** — expectancy, payoff ratio, t-stat, Kelly fraction,
  win/loss streaks, average holding period.
- **Monthly returns heatmap**, **rolling Sharpe** (is the edge persistent or
  one lucky stretch?), **return distribution** (skew/kurtosis/VaR/CVaR), and a
  **worst-drawdowns** table.

Rules of thumb when reading it: want **Sharpe ≥ 1**, **positive alpha with
information ratio ≥ 0.5**, **|t-stat| ≥ 2** on per-trade returns, **≥ 30
trades**, and a **max drawdown you could actually sit through**. A profit
factor of ∞ / no losing trades almost always means lookahead bias.

**Finance methodology (audited):** Sharpe & Sortino use returns *in excess of
the risk-free rate*; Sortino's downside deviation is the textbook RMS of
shortfall over **all** periods (not the sample stdev of only-negative
returns). CAPM **beta** = cov(strategy, benchmark)/var(benchmark) and
**Jensen's alpha** is annualised arithmetically. Backtest applies a 1-bar
execution lag (no lookahead) and charges the full cost model on turnover;
idle cash earns 0 (a deliberately conservative assumption). The 0–10 score is
a fixed, transparent weighted blend (weights sum to 1) shrunk by a confidence
factor when the trade count is low.

---

## 8. Verification — how you know it's correct

Before any metric is computed, the app runs two independent checklists and
shows them in a **Verification** panel (green = pass, amber = warning, red =
fail):

**Data** — source, coverage (bars + date span), sorted/unique index, no
missing OHLC, positive prices, OHLC consistency (high/low actually bound
open/close every bar), volume present, and an outlier scan for split/bad-tick
jumps. A red check stops the run — the numbers would be meaningless.

**AI strategy code** — sandbox validation (no imports/eval/file access),
executes on the data without error, output is finite and within [-1, 1],
actually trades (position changes, not a constant), matches its own declared
direction (no shorts in a "long only" strategy), and a lookahead guard
(flags negative `.shift()`; the engine's execution lag also blocks
current-bar peeking). A red check stops the run.

Only if both checklists are clean do you get performance, verdict, and
analysis — so you can trust them.

---

## 9. Starter ideas — what to try first

Pick one from the **"Start from an example"** dropdown in the app (it fills the
prompt and suggests what data to load), or use these:

| Strategy prompt | Data to upload / fetch |
|---|---|
| *"Go long when the 50-day SMA crosses above the 200-day SMA; exit on the reverse cross. Long only."* | Daily, 8+ yrs of a trending large-cap/index: `RELIANCE.NS`, `SPY`, `^NSEI` |
| *"Long when RSI(2) < 10 while price is above its 200-day SMA; exit when RSI(2) > 60 or price drops below the 200-day SMA."* | Daily, 10+ yrs of a liquid ETF: `SPY`, `QQQ`, `NIFTYBEES.NS` |
| *"Go long on a close above the upper Bollinger Band (20, 2.0); exit on a close back below the 20-period middle band."* | Daily/hourly, 5+ yrs of something volatile: `BTC-USD`, `TSLA` |
| *"12-month momentum skipping the last month: long when positive, else flat."* | Daily, 15+ yrs of a broad index: `^GSPC`, `^NSEI` |

Good first run: **Fetch by ticker → `^NSEI` (or `SPY`) → last 10–15 years →
pick the SMA golden-cross example → Generate & Backtest.** Then read the
quality report and the alpha vs Buy & Hold.

Tips for good results:
- More history = more trades = more trustworthy stats (aim for 30+ trades).
- Be specific about entry, exit, direction (long-only vs long/short), and sizing.
- If the verdict flags too few trades or insignificant t-stat, the "edge" is
  probably noise — loosen the rules or test on more data.

---

## 10. What each file does

| File | Purpose |
|---|---|
| [`app.py`](app.py) | Streamlit UI — data input, settings, charts, verdict, analysis. Launch this. |
| [`requirements.txt`](requirements.txt) | Python package list. |
| [`smoke_test.py`](smoke_test.py) | Offline end-to-end test (no API key) — pipeline + sandbox + analytics. |
| [`setup_wsl.sh`](setup_wsl.sh) | Optional WSL setup script (not needed on Windows). |

Inside **`core/`**:

| File | Purpose |
|---|---|
| [`core/constants.py`](core/constants.py) | **Single place for costs/charges, capital, risk-free rate, benchmark, examples.** |
| [`core/data.py`](core/data.py) | CSV loading **and** ticker fetch (yfinance) → clean OHLCV. |
| [`core/llm.py`](core/llm.py) | Provider abstraction — Anthropic (Claude) & Google (Gemini), unified JSON/text calls. |
| [`core/strategy_generator.py`](core/strategy_generator.py) | Turns the prompt into strategy code via `core/llm.py`; AST-sandboxes & runs it. |
| [`core/indicators.py`](core/indicators.py) | Indicator library (SMA, EMA, RSI, MACD, Bollinger, ATR, …) available to strategies. |
| [`core/engine.py`](core/engine.py) | Vectorized backtest engine — execution lag, cost model, equity curve, trades. |
| [`core/metrics.py`](core/metrics.py) | CAGR, Sharpe/Sortino (excess-return), drawdown, Calmar, profit factor, … |
| [`core/verification.py`](core/verification.py) | Pass/warn/fail checks on the data and the AI's code, shown before any metric. |
| [`core/benchmark.py`](core/benchmark.py) | Default benchmark strategies, run through the same engine for fair comparison. |
| [`core/analysis.py`](core/analysis.py) | Post-backtest analytics: monthly table, rolling Sharpe, drawdowns, trade stats, alpha/beta, plain-English quality report. |
| [`core/evaluator.py`](core/evaluator.py) | Turns metrics into the deterministic 0–10 score, recommendation, risk flags, LLM verdict. |

---

## 11. Typical workflow

1. `python -m streamlit run app.py`
2. Paste your API key (skip if you'll paste your own code) or set the env var.
3. Pick a model and a benchmark; tweak charges if your market differs.
4. Choose data: upload a CSV **or** fetch by ticker + date range.
5. Either describe an idea, or switch to **Paste my own code** (copy the
   provided prompt for your own LLM, paste the code back).
6. Click **Generate & Backtest** / **Backtest my code**.
7. Check the **Verification** panel first (data + code). Then read
   performance → verdict → **post-backtest analysis**. Expand the bottom
   sections for the score breakdown, generated code, and full trade log.

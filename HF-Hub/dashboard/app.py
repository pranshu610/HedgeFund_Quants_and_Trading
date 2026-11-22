"""
dashboard/app.py — HF-Hub interactive dashboard
Run:  streamlit run dashboard/app.py     (from the repo root)

Amber-on-dark visual hub over the whole platform: strategy universe, an
interactive backtest lab, walk-forward optimiser, regime detection, capital
allocation, Monte Carlo robustness, per-trade forecasts, and an execution panel.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

import dashboard.engine as E
from config import INITIAL_CAPITAL, START, END, FEED

st.set_page_config(page_title="HF-Hub", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family:'IBM Plex Sans',sans-serif; background-color:#0B0E14; color:#DDE4EE; }
h1 { font-family:'IBM Plex Mono',monospace !important; color:#F5A623 !important; letter-spacing:-0.02em; }
h2,h3 { font-family:'IBM Plex Sans',sans-serif !important; color:#8A9BB0 !important; font-weight:400 !important; }
[data-testid="metric-container"] { background:#111520; border:1px solid #1A2030; border-radius:6px; padding:12px 16px; }
[data-testid="metric-container"] label { font-family:'IBM Plex Mono',monospace !important; font-size:0.6rem !important; letter-spacing:0.1em; color:#4A5A70 !important; text-transform:uppercase; }
[data-testid="metric-container"] [data-testid="metric-value"] { font-family:'IBM Plex Mono',monospace !important; font-size:1.15rem !important; font-weight:600; color:#F5A623 !important; }
[data-testid="stSidebar"] { background:#080B10; border-right:1px solid #1A2030; }
.stTabs [data-baseweb="tab-list"] { gap:6px; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace; font-size:0.72rem; background:#111520; border-radius:4px; border:1px solid #1A2030; color:#4A5A70; }
.stTabs [aria-selected="true"] { background:#1A2030 !important; border-color:#F5A623 !important; color:#F5A623 !important; }
hr { border-color:#1A2030 !important; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 📈 HF-Hub")
    from config import feed_status, default_symbol
    _fs = feed_status()
    st.caption(f"data feed: `{FEED}` " + ("🟢" if _fs["ok"] else "🔴"))
    if not _fs["ok"]:
        st.error(f"Feed not ready → {_fs['hint']}", icon="⚙️")

    # --- search ANY ticker (free text) + optional quick-picks ---
    symbol = st.text_input("Ticker (search any asset)", value=default_symbol(),
                           help="Any ticker your feed supports: AAPL, BTC-USD, EURUSD=X, GC=F, "
                                "^FTSE, CL=F, TSLA, NVDA, SPY, TLT ... ('DEMO' is synthetic-only)")
    with st.expander("ℹ️ How to type tickers (US, India, global…)"):
        st.markdown(
            "**Feed coverage:**\n"
            "- `alpaca` → **US stocks + crypto only** (and it's the only feed that can *paper-trade*).\n"
            "- `yfinance` → **near-global data** (India, London, indices, FX, commodities, small-caps) "
            "but data-only, no trading.\n\n"
            "**Ticker formats (yfinance):**\n"
            "- US stock: `AAPL`, `NVDA`, `TSLA`\n"
            "- Crypto: `BTC-USD`, `XRP-USD`, `SOL-USD`\n"
            "- 🇮🇳 India NSE: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`  ·  BSE: `500325.BO`\n"
            "- 🇮🇳 Indices: SENSEX `^BSESN` · Nifty 50 `^NSEI`\n"
            "- 🇬🇧 London: `VOD.L`, `HSBA.L`  ·  🇩🇪 `SAP.DE`  ·  🇯🇵 `7203.T`\n"
            "- FX: `EURUSD=X`, `GBPINR=X`  ·  Commodity futures: `GC=F` (gold), `CL=F` (oil)\n"
            "- Indices: S&P `^GSPC`, FTSE `^FTSE`, VIX `^VIX`\n\n"
            "*Small/'underground' names: yfinance has many micro-caps and OTC tickers; very obscure "
            "or newly-listed ones may have little/no history.*")

    _QUICK = {
        "US Equity": ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"],
        "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD"],
        "🇮🇳 India": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "^BSESN", "^NSEI"],
        "🌍 Global": ["VOD.L", "SAP.DE", "7203.T", "^FTSE", "^N225"],
        "FX": ["EURUSD=X", "GBPUSD=X", "USDINR=X", "USDJPY=X"],
        "Commodity": ["GC=F", "CL=F", "SI=F", "NG=F"],
        "Index/Rates": ["^GSPC", "^VIX", "TLT"],
    }
    qc = st.selectbox("…or quick-pick class", ["—"] + list(_QUICK))
    if qc != "—":
        symbol = st.radio("Quick ticker", _QUICK[qc], horizontal=True)
    if FEED == "alpaca" and (symbol.endswith((".NS", ".BO", ".L", ".DE", ".T")) or symbol.startswith("^")
                             or "=X" in symbol or "=F" in symbol):
        st.warning("Alpaca can't load this symbol type (US stocks + crypto only). "
                   "Relaunch with `HF_FEED=yfinance` for India/global/FX/commodity data.", icon="🌍")

    # --- timeframe: bar interval + lookback window ---
    interval = st.selectbox("Bar interval", ["1d", "1h", "30m", "15m", "5m", "1m"], index=0,
                            help="Intraday intervals need HF_FEED=yfinance and have limited history "
                                 "(1m ≈ 7 days, ≤1h ≈ 60–730 days).")
    lookback = st.selectbox("Lookback", ["Custom dates", "5 days", "1 month", "3 months",
                                         "6 months", "YTD", "1 year", "5 years", "Max"], index=6)
    import datetime as _dt
    _today = _dt.date.today()
    _map = {"5 days": 5, "1 month": 30, "3 months": 91, "6 months": 182,
            "1 year": 365, "5 years": 1825, "Max": 3650}
    if lookback == "Custom dates":
        start = st.text_input("Start", value=START)
        end = st.text_input("End", value=END)
    elif lookback == "YTD":
        start, end = f"{_today.year}-01-01", str(_today)
    else:
        start = str(_today - _dt.timedelta(days=_map[lookback])); end = str(_today)
    capital = st.number_input("Capital ($)", value=int(INITIAL_CAPITAL), step=10_000)
    st.markdown("---")
    cost_bps = st.slider("Commission (bps/side)", 0.0, 10.0, 1.0, 0.5)
    slip_bps = st.slider("Slippage (bps/side)", 0.0, 10.0, 2.0, 0.5)
    spread_bps = st.slider("Bid/ask spread (bps/side)", 0.0, 20.0, 1.0, 0.5,
                           help="Half-spread paid crossing the book each side. Wider for illiquid "
                                "names, crypto, small-caps.")
    fin_bps = st.slider("Overnight financing (bps/yr)", 0.0, 1500.0, 0.0, 50.0,
                        help="Annual borrow/margin cost charged on held exposure every bar — the "
                             "cost of carrying positions overnight (shorts & leverage especially).")
    st.caption("Switch to real data by setting HF_FEED=yfinance or alpaca in your env.")

@st.cache_data(show_spinner=False)
def _bars(sym, s, e, iv):
    return E.load_bars(sym, s, e, iv)

try:
    df = _bars(symbol, start, end, interval)
except Exception as ex:
    st.error(f"Could not load `{symbol}` ({interval}) from the **{FEED}** feed.")
    st.markdown(f"**Reason:** {ex}")
    fs = feed_status()
    st.markdown("### How to fix")
    if FEED == "alpaca":
        st.markdown(
            "- **Install the package** (same env as Streamlit): `pip install alpaca-py`\n"
            "- **Set your paper keys** as env vars before launching:\n"
            "  ```\n  export ALPACA_KEY=your_key\n  export ALPACA_SECRET=your_secret\n  ```\n"
            "- **Use a real ticker** — `DEMO` only exists on the synthetic feed. Try `AAPL`, "
            "`SPY`, `BTC-USD`, `EURUSD=X`.")
    elif FEED == "yfinance":
        st.markdown("- Install it: `pip install yfinance`\n- Use a real ticker (`AAPL`, `SPY`, "
                    "`BTC-USD`). `DEMO` is synthetic-only.")
    st.info("Or switch back to the offline demo: relaunch with `HF_FEED=synthetic` "
            "(or unset HF_FEED) and use ticker `DEMO`.", icon="🎲")
    st.stop()

st.markdown("# HF-Hub — quant research & risk hub")
st.markdown(f"`{len(df):,} bars` · `{df.index[0].date()} → {df.index[-1].date()}` · "
            f"`{symbol}` · `costs {cost_bps+slip_bps+spread_bps:.1f}bps/side + {fin_bps:.0f}bps/yr`")

if FEED == "synthetic":
    st.warning(
        "**You're on SYNTHETIC (random) data.** Every number below is illustrative only. "
        "Random data has no real edge, so strategies *should* look unprofitable here — that's the "
        "integrity check working, not you losing money. To see real results, set `HF_FEED=yfinance` "
        "in your terminal before launching (see the Overview tab).", icon="🎲")
if len(df) < 120:
    st.warning(f"Only {len(df)} bars loaded — some strategies need 100+ bars. Widen the lookback "
               f"or use a coarser interval if a strategy errors.", icon="⏳")
st.markdown("---")

tabs = st.tabs(["🏠 Cockpit", "🔭 Scanner", "🩺 Post-mortem", "💼 Fund", "🧪 Strategies",
                "📖 Explained", "🎛️ Optimizer", "🌡️ Regime", "⚖️ Allocation", "🎲 Monte Carlo",
                "🔮 Forecast", "🚀 Execution", "🎲 Sizing / GTO", "⚡ Crypto Arb",
                "📊 Report", "🎓 Tutorial"])
(TC, TSCAN, TPM, TF, TS, TEX, TO, TR, TA, TM, TFC, TE, TK, TARB, TRP, TUT) = tabs

# cached heavy computations — keyed on primitives so tab-switching is instant
@st.cache_data(show_spinner=False)
def cached_book(sym, s, e, iv, cb, sb):
    return E.build_book(E.load_bars(sym, s, e, iv), cb, sb, symbol=sym)

@st.cache_data(show_spinner=False)
def cached_rec(sym, s, e, iv):
    return E.recommended_strategy(E.load_bars(sym, s, e, iv))

@st.cache_data(show_spinner=False)
def cached_rank(sym, s, e, iv, m, cb, sb):
    return E.rank_strategies_on_df(E.load_bars(sym, s, e, iv), m, cb, sb, symbol=sym)

@st.cache_data(show_spinner=False)
def cached_report(sym, s, e, iv, cap):
    return E.fund_report(sym, E.load_bars(sym, s, e, iv), cap)

@st.cache_data(show_spinner=False)
def cached_signals(sym, s, e, iv, cap, cb, sb, sp, fb):
    return E.actionable_signals(sym, E.load_bars(sym, s, e, iv), cap, cb, sb, sp, fb)

# ------------------------------------------------------------------ Cockpit
with TC:
    st.subheader(f"Cockpit — live view for {symbol}")

    rec = cached_rec(symbol, start, end, interval)
    book = cached_book(symbol, start, end, interval, cost_bps, slip_bps)
    beq = (1 + book["book_returns"]).cumprod() * capital
    from backtest.metrics import sharpe as _sh, max_drawdown as _mdd

    c = st.columns(4)
    c[0].metric("Regime now", rec["regime"].split(" / ")[0])
    c[1].metric("Recommended strategy", rec["pick"])
    c[2].metric("Book Sharpe", f"{_sh(book['book_returns']):.2f}")
    c[3].metric("Book Max DD", f"{_mdd(beq):.1%}")
    st.caption(f"Regime: **{rec['regime']}** · favour {', '.join(rec['favour'])} · "
               f"avoid {', '.join(rec['avoid']) or '—'} · gross scaled to {rec['gross']:.0%}. "
               f"Top regime-fit strategy: **{rec['pick']}** ({rec['pick_category']}, "
               f"backtest Sharpe {rec['pick_sharpe']}).")

    st.markdown(f"### 🔴 Live signals — actionable trade tickets for {symbol}")
    st.info(E.relevance_note(symbol), icon="🎯")
    st.caption("Each row is a real ticket: what to do NOW, at what price, with stop, target, R:R and "
               "size. `status`: 🔔 NEW ENTRY (act now) · 🔄 FLIP · 🚪 EXIT · ✅ already in · ⚪ no setup "
               "(waiting for price to confirm). `quality` = how this strategy has performed on THIS "
               "asset historically — ❌ means don't take the signal.")
    st.caption("**Stop/target/R:R are DERIVED per strategy** from its own historical trade "
               "distribution on this asset (`levels_from`), not a fixed ATR multiple — a breakout "
               "and a mean-reversion strategy have opposite payoff shapes. Sorted by "
               "**expect_%/trade** = win% × avgWin − loss% × avgLoss. That, not R:R, is the number "
               "that decides: a 2.1 R:R at a 25% win rate LOSES money; a 0.54 R:R at 72% MAKES it.")
    ls = cached_signals(symbol, start, end, interval, float(capital), cost_bps, slip_bps, spread_bps, fin_bps)
    def _hl(v):
        return ("color:#4CAF7D" if v == "LONG" else "color:#E8472A" if v == "SHORT"
                else "color:#8A9BB0")
    st.dataframe(ls.style.map(_hl, subset=["side"]), use_container_width=True, hide_index=True)
    with st.expander("Why each signal fired (reasoning per strategy)"):
        for _, r in ls.iterrows():
            st.markdown(f"**{r['strategy']}** — {r['status']} · {r['side']}  \n{r['why']}")

    # candlestick of the reference asset
    st.plotly_chart(E.candlestick_fig(df, None, f"{symbol} — candlesticks"), use_container_width=True)

    # signal scanner: one strategy across a whole basket (the screener)
    st.markdown("### 🔭 Signal scanner — potential trades across a basket")
    sc1, sc2 = st.columns([1, 2])
    scan_strat = sc1.selectbox("Strategy", list(E.single_strategies()), key="cockpit_scan_strat")
    default_b = "SPY,QQQ,AAPL,NVDA,GLD,BTC-USD,EURUSD=X" if FEED != "synthetic" else "AST1,AST2,AST3,AST4,AST5"
    scan_syms = sc2.text_input("Basket", value=default_b, key="cockpit_scan_basket")
    if st.button("🔭 Scan for signals"):
        with st.spinner("scanning the field..."):
            scan = E.signal_scanner(scan_strat, [s.strip() for s in scan_syms.split(",") if s.strip()],
                                    start, end, interval, cost_bps, slip_bps)
        def _hl2(v): return ("color:#4CAF7D" if v == "LONG" else "color:#E8472A" if v == "SHORT"
                             else "color:#8A9BB0")
        st.dataframe(scan.style.map(_hl2, subset=["signal"]), use_container_width=True, hide_index=True)
        st.caption("Confidence blends historical Sharpe, profit factor and exposure. "
                   "Actionable signals (LONG/SHORT) ranked first — surveillance before the strike.")

    st.plotly_chart(E.equity_curve_fig(beq, "Blended book equity (vol-targeted)"),
                    use_container_width=True)
    st.markdown("---")

# ------------------------------------------------------------------ Overview
with TC:
    st.subheader("What this is")
    st.markdown(
        "A multi-strategy quant **research + risk + paper-trading** hub. It generates signals, "
        "backtests them honestly (with costs, slippage, no lookahead), sizes trades by risk, "
        "allocates capital across strategies, stress-tests with Monte Carlo, and paper-trades — "
        "all the machinery of a small systematic book.")

    ut = E.universe_table()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategies", f"{len(E.ALL_STRATEGIES)} (all live)")
    c2.metric("Categories", f"{ut.category.nunique()}")
    c3.metric("Data feed", FEED)
    c4.metric("Broker", "paper only")

    st.markdown(f"### The {len(E.ALL_STRATEGIES)} strategies")
    groups = {}
    for key, group, desc in E.ALL_STRATEGIES:
        groups.setdefault(group, []).append((key, desc))
    for g, items in groups.items():
        st.markdown(f"**{g}**")
        st.dataframe(pd.DataFrame([{"strategy": k, "what it does": d} for k, d in items]),
                     use_container_width=True, hide_index=True)

    st.markdown("### How to use it")
    st.markdown(
        "1. **Strategies** — pick any of the 16, see its equity curve, drawdown, metrics, and a "
        "plain-English explanation. Single-asset ones let you tune parameters live.\n"
        "2. **Optimizer** — walk-forward search for the best settings, scored out-of-sample.\n"
        "3. **Regime** — what market regime we're in and which strategies to favour/avoid now.\n"
        "4. **Allocation** — how to split capital across strategies (HRP is the recommended method).\n"
        "5. **Monte Carlo** — is a strategy/book robust or fragile under resampling?\n"
        "6. **Forecast** — the per-trade report: direction, size, stop, target, why.\n"
        "7. **Execution** — run the paper trader: it places risk-sized orders in a simulated account.")

    st.info("**To see REAL numbers instead of synthetic:** close the app, then in your terminal run "
            "`HF_FEED=yfinance streamlit run dashboard/app.py` (free daily data, no keys). "
            "Type a real symbol like `AAPL` or `SPY` in the sidebar.", icon="💡")

# ------------------------------------------------------------------ Strategies
with TS:
    st.subheader(f"Strategies — all {len(E.ALL_STRATEGIES)}")
    keys = [k for k, _, _ in E.ALL_STRATEGIES]
    labels = {k: f"{k}  ·  {g}" for k, g, _ in E.ALL_STRATEGIES}
    key = st.selectbox("Strategy", keys, format_func=lambda k: labels[k], index=0)
    meta_desc = dict((k, d) for k, _, d in E.ALL_STRATEGIES)[key]
    st.caption(meta_desc)

    single = key in E.single_strategies()
    is_pair = key in ("pairs_statarb", "kalman_pairs")
    params = {}

    if is_pair:
        st.markdown("**This is a relative-value strategy — it LONGS one asset and SHORTS another.**")
        pc = st.columns(2)
        sym_a = pc[0].text_input("Long-leg ticker (A)", value="AMD" if FEED != "synthetic" else "PAIR_A")
        sym_b = pc[1].text_input("Hedge ticker (B)", value="NVDA" if FEED != "synthetic" else "PAIR_B")
        try:
            res = E.run_real_pair(key, sym_a, sym_b, start, end, capital)
            st.success(f"**Current trade: {res['legs']}**")
            st.caption(res["reason"])
        except Exception as ex:
            st.error(f"Could not run pair ({sym_a}/{sym_b}): {ex}")
            res = None
    elif single:
        st.markdown("**Parameters** (defaults = industry standard; tweak and watch it update)")
        st.caption("Tip: string options like CCI's **mode** (reversion vs momentum) are pickable here. "
                   "The Optimizer tab can also choose the best mode+params for you automatically.")
        scls = E.single_strategies()[key]
        defaults = scls().params
        grid = getattr(scls, "param_grid", {}) or {}
        cols = st.columns(min(4, max(1, len(defaults))))
        for i, (pn, pv) in enumerate(defaults.items()):
            col = cols[i % len(cols)]
            if isinstance(pv, bool):
                params[pn] = col.checkbox(pn, value=pv)
            elif isinstance(pv, str):
                # categorical: use grid options if available, else the current value
                opts = [o for o in grid.get(pn, [pv]) if isinstance(o, str)] or [pv]
                if pv not in opts:
                    opts = [pv] + opts
                params[pn] = col.selectbox(pn, opts, index=opts.index(pv))
            elif isinstance(pv, int):
                params[pn] = col.number_input(pn, value=int(pv), step=1)
            elif isinstance(pv, float):
                params[pn] = col.number_input(pn, value=float(pv))
            else:
                params[pn] = pv
        strat, bt = E.run_strategy_backtest(key, df, params, cost_bps, slip_bps, capital,
                                            spread_bps, fin_bps)
        res = {"equity": bt.equity, "returns": bt.returns, "stats": bt.stats, "position": bt.position,
               "note": strat._explain(df) if hasattr(strat, "_explain") else meta_desc}
    else:
        st.caption("This strategy uses a basket / macro / event dataset, so it runs on its own "
                   "synthetic inputs (real feeds plug in the same way).")
        res = E.run_any_strategy(key, df, capital, cost_bps, slip_bps)

    if res is None:
        st.stop()

    s = res["stats"]
    m = st.columns(6)
    m[0].metric("Sharpe", f"{s['Sharpe']:.2f}")
    m[1].metric("Sortino", f"{s.get('Sortino', float('nan')):.2f}")
    m[2].metric("Calmar", f"{s.get('Calmar', float('nan')):.2f}")
    m[3].metric("CAGR", f"{s['CAGR']:.1%}")
    m[4].metric("Max DD", f"{s['MaxDrawdown']:.1%}")
    m[5].metric("Vol (ann)", f"{s.get('Vol(ann)', float('nan')):.1%}")
    m2 = st.columns(6)
    m2[0].metric("Win rate", f"{s.get('WinRate', float('nan')):.0%}" if 'WinRate' in s else "—")
    m2[1].metric("Profit factor", f"{s.get('ProfitFactor', float('nan')):.2f}" if 'ProfitFactor' in s else "—")
    m2[2].metric("Trades", f"{s.get('NumTrades', 0)}")
    m2[3].metric("Exposure", f"{s.get('Exposure', float('nan')):.0%}" if 'Exposure' in s else "—")
    m2[4].metric("Avg win", f"{s.get('AvgWin', float('nan')):+.2%}" if 'AvgWin' in s else "—")
    m2[5].metric("Final", f"${s['FinalEquity']:,.0f}")

    ctx = E.winrate_context(s)
    if ctx:
        st.info(ctx, icon="ℹ️")

    # signal-marked price chart where we have a position + the traded price
    if is_pair and res.get("a") is not None:
        pxdf = pd.DataFrame({"open": res["a"], "high": res["a"], "low": res["a"],
                             "close": res["a"], "volume": 1.0})
        st.plotly_chart(E.signal_price_fig(pxdf, res["position"], f"{sym_a} price with pair signals"),
                        use_container_width=True)
    elif single:
        chart_type = st.radio("Chart", ["🕯️ Candlestick", "📈 Line"], horizontal=True, key="strat_chart_type")
        if chart_type.startswith("🕯️"):
            st.plotly_chart(E.candlestick_fig(df, res["position"], f"{symbol} — candles + signals"),
                            use_container_width=True)
        else:
            st.plotly_chart(E.signal_price_fig(df, res["position"], f"{symbol} — price + signals"),
                            use_container_width=True)

    st.plotly_chart(E.equity_curve_fig(res["equity"], f"{key} — strategy equity"), use_container_width=True)
    st.plotly_chart(E.dd_fig(res["equity"]), use_container_width=True)
    st.markdown(f"**Why / how this reads now:** {res['note']}")
    with st.expander("Full metrics"):
        st.dataframe(pd.DataFrame({"metric": list(s),
                                   "value": [round(v, 4) if isinstance(v, float) else v for v in s.values()]}),
                     use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ Optimizer
with TO:
    st.subheader("Optimizer — match the strategy to the asset (and why)")
    st.caption("Everything is walk-forward: params tuned in-sample, scored OUT-of-sample. "
               "Ranked on a robust metric (not win rate — low win rate is normal for trend strategies).")
    metric = st.selectbox("Objective", ["Sharpe", "CAGR", "Sortino"], index=0, key="opt_metric")
    omode = st.radio("Mode", ["🎯 Best strategy for THIS stock",
                              "🌍 Best stock for a strategy",
                              "🔧 Tune one strategy"], key="opt_mode")

    if omode.startswith("🎯"):
        st.markdown(f"Optimising every strategy on **{symbol}**…")
        st.caption("Ranked by a ROBUST score: full-sample Sharpe + profit factor − drawdown penalty, "
                   "with a sanity gate (must be profitable, ≥8 trades, no blow-up). A money-loser is "
                   "never called optimal — even if a noisy out-of-sample fold flatters it.")
        if st.button("Find the best strategy for this stock"):
            with st.spinner("Walk-forward optimising every strategy valid for this asset — "
                            "this runs hundreds of real backtests, ~5-10s. Cached after the first run."):
                rep = cached_rank(symbol, start, end, interval, metric, cost_bps, slip_bps)
                rep["symbol"] = symbol
            if rep["any_edge"]:
                st.success(f"**Best for {rep['symbol']}: `{rep['best']}`**  ·  {rep['best_params']}")
            else:
                st.error(f"**No robust edge on {rep['symbol']} in this window.**")
            st.markdown(f"**Why:** {rep['reason']}")
            st.caption(f"Regime: {rep['regime']}")
            st.dataframe(rep["table"], use_container_width=True, hide_index=True)
            st.caption("`verdict`: ✅ solid · ⚠️ ok/thin/fragile · ❌ loses money.  "
                       "`score` is the robust composite (higher = better).  "
                       "`oos_*` = out-of-sample consistency check (noisy on short windows — that's why "
                       "it isn't the ranking).")

    elif omode.startswith("🌍"):
        skey = st.selectbox("Strategy", list(E.single_strategies()), key="opt_strat_for_universe")
        default_b = "SPY,QQQ,AAPL,NVDA,GLD,TSLA" if FEED != "synthetic" else "AST1,AST2,AST3,AST4,AST5"
        basket = st.text_input("Universe (comma-separated)", value=default_b, key="opt_basket")
        if st.button("Find the best stock for this strategy"):
            with st.spinner("walk-forward optimising across the universe…"):
                rep = E.best_assets_for_strategy(skey, [s.strip() for s in basket.split(",") if s.strip()],
                                                 start, end, interval, metric, cost_bps, slip_bps)
            if rep["any_edge"]:
                st.success(f"**`{rep['strategy']}` fits `{rep['best_symbol']}` best**  ·  {rep['best_params']}")
            else:
                st.error(f"**`{rep['strategy']}` has no robust edge on this universe.**")
            st.markdown(f"**Why:** {rep['reason']}")
            st.dataframe(rep["table"], use_container_width=True, hide_index=True)
            st.caption("Same strategy, every asset optimised independently — where the edge actually lives.")

    else:
        okey = st.selectbox("Strategy to tune", list(E.relevant_strategies(symbol)), key="opt_tune_strat")
        if st.button("Run walk-forward optimisation"):
            with st.spinner("searching parameter grid across folds..."):
                rep = E.optimizer_report(okey, df, metric, cost_bps, slip_bps)
            wf = rep["walk_forward"]
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Default (industry standard)**"); st.json(rep["default_params"])
                st.metric(f"Default {metric}", f"{rep['default_stats'].get(metric, 0):.3f}")
            with c2:
                st.markdown("**▶ Recommended (walk-forward)**"); st.json(wf["recommended_params"])
                st.metric(f"OOS {metric}", f"{wf['oos_score']:.3f}")
            fs = wf["full_sample_stats"]
            st.markdown(f"Full-sample @ recommended → Sharpe **{fs['Sharpe']:.2f}** · "
                        f"CAGR **{fs['CAGR']:.1%}** · MaxDD **{fs['MaxDrawdown']:.1%}** "
                        f"({wf['n_configs_tested']} configs tested)")

# ------------------------------------------------------------------ Regime
with TR:
    st.subheader("Market regime detection")
    rr = E.regime_report(df)
    reg, r = rr["regime"], rr["route"]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(E.regime_gauge(reg), use_container_width=True)
    with c2:
        st.metric("Regime", reg.label())
        st.metric("Gross exposure", f"{r['gross_exposure']:.0%}")
        st.markdown(f"**Favour:** {', '.join(r['favour'])}")
        st.markdown(f"**Avoid:** {', '.join(r['avoid']) or '—'}")
        st.caption(r["note"])
    st.plotly_chart(E.regime_trend_fig(rr["history"]), use_container_width=True)

# ------------------------------------------------------------------ Allocation
with TA:
    st.subheader("Optimal capital allocation")
    st.caption("HRP is the recommended default — robust to estimation error. max_sharpe is a "
               "reference that over-concentrates; the comparison shows why.")
    book = cached_book(symbol, start, end, interval, cost_bps, slip_bps)
    alloc = E.allocation_report(book["returns"])
    st.dataframe(alloc["table"], use_container_width=True)
    method = st.selectbox("Show weights for", list(alloc["weights"]), index=0)
    st.plotly_chart(E.weights_fig(alloc["weights"][method], f"{method} weights"),
                    use_container_width=True)
    st.markdown("**Regime-adjusted book weights** (allocator × regime overlay)")
    st.plotly_chart(E.weights_fig(book["regime_weights"], "regime-adjusted weights"),
                    use_container_width=True)

# ------------------------------------------------------------------ Monte Carlo
with TM:
    st.subheader("Monte Carlo robustness")
    book = cached_book(symbol, start, end, interval, cost_bps, slip_bps)
    src = st.radio("Return stream", ["Blended book", "Single strategy"], horizontal=True)
    if src == "Single strategy":
        k = st.selectbox("Strategy", list(E.relevant_strategies(symbol)), key="mc_strat")
        _, bt = E.run_strategy_backtest(k, df, {}, cost_bps, slip_bps, capital)
        rets = bt.returns
    else:
        rets = book["book_returns"]
    mc = E.montecarlo_report(rets, capital)["robustness"]
    c = st.columns(4)
    c[0].metric("P(loss)", f"{mc.prob_loss:.1%}")
    c[1].metric("Median terminal", f"${mc.p50_terminal:,.0f}")
    c[2].metric("p05 terminal", f"${mc.p05_terminal:,.0f}")
    c[3].metric("Median maxDD", f"{mc.median_maxdd:.1%}")
    st.info(mc.verdict)
    cc1, cc2 = st.columns(2)
    cc1.plotly_chart(E.mc_terminal_fig(mc), use_container_width=True)
    cc2.plotly_chart(E.mc_drawdown_fig(mc), use_container_width=True)

# ------------------------------------------------------------------ Forecast
with TFC:
    st.subheader("Per-trade forecast")
    fkey = st.selectbox("Strategy", list(E.relevant_strategies(symbol)), key="forecast_strat")
    sizing = st.selectbox("Sizing method", ["vol_target", "kelly", "fixed_fractional"])
    fc = E.forecast_report(fkey, df, symbol, capital, sizing)["forecast"]
    c = st.columns(5)
    c[0].metric("Direction", fc.direction)
    c[1].metric("Entry", f"{fc.entry}")
    c[2].metric("Stop / Target", f"{fc.stop} / {fc.target}")
    c[3].metric("R:R", f"{fc.rr}")
    c[4].metric("Size", f"{fc.size_frac:.1%}")
    st.markdown(f"**Why:** {fc.reasoning}")
    st.markdown(f"**Risk:** {fc.risk_notes}")
    st.markdown(f"**Robustness:** {fc.robustness}  ·  P(loss) MC {fc.prob_loss}")

# ------------------------------------------------------------------ Execution
with TE:
    st.subheader("Execution — you're the PM, this is your analyst")
    st.caption("Two modes: **Heads-up** — the analyst proposes trades (with reasoning + risk check), "
               "you confirm each. **Auto** — it executes the risk-sized book itself. Paper only.")

    from live.trader import Trader, SimBroker, default_mandates
    from config import make_feed, alpaca_ready, make_broker

    connected = alpaca_ready()
    use_alpaca = False
    if connected:
        use_alpaca = st.toggle("Trade the real Alpaca paper account (instead of offline SimBroker)",
                               value=False,
                               help="Uses your ALPACA_KEY/ALPACA_SECRET env vars. Paper endpoint only.")
    if "trader" not in st.session_state or st.session_state.get("_use_alpaca") != use_alpaca:
        st.session_state.trader = Trader(make_broker(float(capital), use_alpaca), make_feed(),
                                         default_mandates(symbol), capital=float(capital))
        st.session_state._use_alpaca = use_alpaca
    trader = st.session_state.trader

    cc = st.columns(4)
    cc[0].metric("Alpaca paper keys", "🟢 detected" if connected else "⚪ not set")
    cc[1].metric("Broker", "Alpaca paper" if use_alpaca else "SimBroker")
    rstate = trader.risk.update_equity(trader.broker.equity())
    cc[2].metric("Drawdown", f"{rstate['drawdown']:.1%}")
    cc[3].metric("Risk state", "🟢 OK" if rstate["state"] == "OK" else "🔴 HALT")
    if rstate["state"] != "OK":
        st.error(f"Risk manager: {rstate['state']}")

    mode = st.radio("Execution mode", ["🫱 Heads-up (I confirm)", "🤖 Auto-execute"], horizontal=True)

    st.markdown("### 📋 Mandates — your universe, auto-routed (nothing hardcoded)")
    uni = st.text_input("Universe (comma-separated tickers you want the book to consider)",
                        value=symbol if FEED != "synthetic" else "AST1,AST2,AST3,AST4",
                        key="exec_universe")
    ac1, ac2 = st.columns([1, 2])
    if ac1.button("🧭 Auto-route universe"):
        with st.spinner("evaluating each ticker's history, regime and valid strategies…"):
            st.session_state.routed = E.auto_route_mandates(
                [x.strip() for x in uni.split(",") if x.strip()], start, end, interval,
                cost_bps, slip_bps)
    routed = st.session_state.get("routed")
    if routed is not None and len(routed):
        st.dataframe(routed[["symbol", "strategy", "verdict", "score", "Sharpe", "regime",
                             "trade", "params"]], use_container_width=True, hide_index=True)
        with st.expander("Why each ticker got that strategy"):
            for _, r in routed.iterrows():
                st.markdown(f"**{r['symbol']} → {r['strategy']}** ({r['verdict']})  \n{r['why']}")
        tradeable = routed[routed["trade"]]
        st.caption(f"{len(tradeable)}/{len(routed)} tickers have a strategy that passes the sanity "
                   f"gate. Only those get mandates — no edge, no trade.")
        if st.button("✅ Load these as the book's mandates"):
            from live.trader import Mandate
            mds = [Mandate(r["strategy"], r["symbol"]) for _, r in tradeable.iterrows()]
            if mds:
                trader.mandates = mds
                st.success(f"Loaded {len(mds)} auto-routed mandates.")
            else:
                st.error("No ticker passed the gate — nothing to trade. That's the honest answer.")
    st.markdown("**Active mandates**")
    st.dataframe(pd.DataFrame([{"strategy": m.strategy_key, "symbol": m.symbol}
                               for m in trader.mandates]), use_container_width=True, hide_index=True)

    if mode.startswith("🫱"):
        if st.button("🔎 Generate proposals (analyst heads-up)"):
            st.session_state.proposals = trader.propose()
        props = st.session_state.get("proposals", [])
        if props:
            st.markdown("### 📋 Proposed trades — review & confirm")
            chosen = []
            for i, p in enumerate(props):
                cols = st.columns([0.5, 3, 1])
                ok = cols[0].checkbox(" ", value=p.approved, key=f"prop{i}", disabled=not p.approved)
                side_c = "#4CAF7D" if p.side == "buy" else "#E8472A"
                cols[1].markdown(
                    f"<span style='color:{side_c};font-weight:600'>{p.side.upper()} {p.qty} {p.symbol}</span> "
                    f"(${p.notional:,.0f}) · <span style='color:#8A9BB0'>{p.strategy}</span><br>"
                    f"<span style='color:#8A9BB0;font-size:0.85em'>{p.reason}</span>", unsafe_allow_html=True)
                cols[2].markdown(f"<span style='color:#F5A623;font-size:0.8em'>{p.risk_note}</span>",
                                 unsafe_allow_html=True)
                if ok and p.approved:
                    chosen.append(p)
            if st.button(f"✅ Confirm & execute {len(chosen)} trade(s)"):
                orders = trader.execute(chosen)
                st.success(f"Executed {len(orders)} paper order(s).")
                st.session_state.proposals = []
        else:
            st.info("Click **Generate proposals** — the analyst will flag what to trade and why.")
    else:
        if st.button("🤖 Run auto cycle"):
            rep = trader.run_once()
            if rep.get("errors"):
                st.warning("Skipped: " + "; ".join(rep["errors"][:4]))
            st.success(f"Auto cycle · regime {rep['regime']} · gross {rep['gross_exposure']:.0%} "
                       f"· {len(rep['orders'])} orders")

    if st.button("↺ Reset paper account"):
        st.session_state.trader = Trader(make_broker(float(capital), use_alpaca), make_feed(),
                                         default_mandates(symbol), capital=float(capital))
        st.session_state.proposals = []
        st.info("Paper account reset.")

    snap = trader.snapshot()
    m = st.columns(3)
    m[0].metric("Equity", f"${snap['equity']:,.0f}" if snap['equity'] else "—")
    m[1].metric("Cash", f"${snap['cash']:,.0f}" if snap['cash'] else "—")
    m[2].metric("Open positions", f"{len(snap['positions'])}")
    if snap["positions"]:
        st.markdown("**Positions**"); st.dataframe(pd.DataFrame(snap["positions"]),
                                                   use_container_width=True, hide_index=True)
    if snap["orders"]:
        st.markdown("**Recent orders**")
        od = pd.DataFrame(snap["orders"])
        if "ts" in od: od["ts"] = od["ts"].astype(str)
        st.dataframe(od, use_container_width=True, hide_index=True)

    st.caption("For live paper on a real API: set ALPACA_KEY/ALPACA_SECRET, swap SimBroker→AlpacaBroker "
               "in live/trader.py, and schedule run_once()/propose() per timeframe. config.ALPACA_PAPER stays True.")

# ------------------------------------------------------------------ Fund
with TF:
    st.subheader("Fund desk — the book as a fund")
    book = cached_book(symbol, start, end, interval, cost_bps, slip_bps)
    beq = (1 + book["book_returns"]).cumprod() * capital
    from backtest.metrics import sharpe as _fsh, max_drawdown as _fmdd, cagr as _fcagr
    rp = E.risk_panel(book, capital)

    f = st.columns(6)
    f[0].metric("NAV", f"${beq.iloc[-1]:,.0f}")
    f[1].metric("CAGR", f"{_fcagr(beq):.1%}")
    f[2].metric("Sharpe", f"{_fsh(book['book_returns']):.2f}")
    f[3].metric("Max DD", f"{_fmdd(beq):.1%}")
    f[4].metric("VaR95 (1d)", f"${rp['VaR95_1d']:,.0f}")
    f[5].metric("CVaR95 (1d)", f"${rp['CVaR95_1d']:,.0f}")

    st.plotly_chart(E.equity_curve_fig(beq, "Fund NAV (vol-targeted, regime-adjusted)"),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    attr = E.pnl_attribution(book, capital)
    c1.plotly_chart(E.attribution_fig(attr), use_container_width=True)
    c2.plotly_chart(E.corr_heatmap(rp["corr"]), use_container_width=True)

    st.markdown("### 📒 Trade blotter — historical signals & per-trade P&L")
    bkey = st.selectbox("Strategy blotter", list(E.relevant_strategies(symbol)), key="fund_blotter_strat")
    bres = E.run_any_strategy(bkey, df, capital, cost_bps, slip_bps)
    blot = E.trade_blotter(df, bres["position"])
    if len(blot):
        bs = E.blotter_summary(blot)
        q = st.columns(6)
        q[0].metric("Net P&L", f"{bs['net_pnl_pct']:+.1f}%")
        q[1].metric("Avg / trade", f"{bs['avg_pnl_pct']:+.2f}%")
        q[2].metric("Win rate", f"{bs['win_rate']:.0%}")
        q[3].metric("Profit factor", f"{bs['profit_factor']}" if bs['profit_factor'] else "—")
        q[4].metric("Avg win / loss", f"{bs['avg_win_pct']:+.1f}% / {bs['avg_loss_pct']:+.1f}%")
        q[5].metric("Avg hold", f"{bs['avg_bars_held']} bars")
        st.caption(f"{bs['trades']} closed · {bs['longs']} long / {bs['shorts']} short · "
                   f"best {bs['best_pct']:+.1f}% · worst {bs['worst_pct']:+.1f}%. "
                   f"(If you see 0 shorts, that strategy is long-only or `allow_short` is off — "
                   f"vix_rsi and overnight_drift are long-only by design.)")
        def _pl(v): return "color:#4CAF7D" if v > 0 else "color:#E8472A"
        st.dataframe(blot.style.map(_pl, subset=["pnl_pct"]), use_container_width=True,
                     hide_index=True, height=330)
    else:
        st.info("No trades for this strategy on this data.")

    st.markdown("### 🛡️ Hedge calculator — 'how much do I short to be neutral?'")
    h1, h2 = st.columns(2)
    hedge_sym = h1.text_input("Hedge instrument (e.g. SPY)", value="SPY" if FEED != "synthetic" else "MKT")
    notional = h2.number_input("Position notional ($)", value=50_000, step=5_000)
    try:
        hres = E.hedge_calculator(df, E.load_bars(hedge_sym, start, end), float(notional))
        st.metric(f"β of {symbol} vs {hedge_sym}", f"{hres['beta']:.2f}",
                  delta=f"hedge: short ${abs(hres['hedge_notional']):,.0f}")
        st.caption(hres["explain"])
    except Exception as ex:
        st.warning(f"Hedge calc failed: {ex}")

    st.markdown("### 🧭 Trade management plan — stops, scale-outs, trailing")
    from core.indicators import atr as _atr
    from risk.sizing import trade_management_plan
    t1, t2 = st.columns(2)
    tdir = t1.radio("Direction", ["LONG", "SHORT"], horizontal=True, key="trade_mgmt_dir")
    _a = float(_atr(df, 14).iloc[-1]); _e = float(df["close"].iloc[-1])
    plan = trade_management_plan(_e, _a, +1 if tdir == "LONG" else -1)
    p1, p2, p3 = st.columns(3)
    p1.metric("Entry ~", f"{plan['entry']}")
    p2.metric("Initial stop", f"{plan['initial_stop']}")
    p3.metric("Risk / unit", f"{plan['risk_per_unit']}")
    st.caption("**" + plan["summary"] + "** — scale out bit by bit, trail the runner.")

    st.markdown("### 🌍 Multi-asset scan — this strategy across a basket")
    scan_key = st.selectbox("Strategy to scan", list(E.relevant_strategies(symbol)), key="fund_scan_strat")
    default_basket = "SPY,QQQ,GLD,BTC-USD,EURUSD=X" if FEED != "synthetic" else "AST1,AST2,AST3,AST4"
    basket = st.text_input("Tickers (comma-separated)", value=default_basket)
    if st.button("Run scan"):
        with st.spinner("backtesting across the basket..."):
            scan = E.multi_asset_scan(scan_key, [s.strip() for s in basket.split(",") if s.strip()],
                                      start, end, cost_bps, slip_bps)
        st.dataframe(scan, use_container_width=True, hide_index=True)
        st.caption("Sorted by Sharpe — where does this edge actually live?")

    st.markdown("### 📄 Daily fund report")
    if st.button("Generate report"):
        rep_md = E.fund_report(symbol, df, capital)
        st.download_button("⬇ Download report (.md)", rep_md,
                           file_name=f"fund_report_{df.index[-1].date()}.md")
        with st.expander("Preview"):
            st.markdown(rep_md)

# ------------------------------------------------------------------ Explained
with TEX:
    st.subheader("Strategies explained — the analyst's playbook")
    st.caption("Every strategy: what it is, WHERE the edge comes from, when it works, when it "
               "breaks, its natural timeframe/horizon, and whether it's realistically retail-runnable.")
    cards = E.strategy_explainer()
    groups = {}
    for c in cards:
        groups.setdefault(c["category"], []).append(c)
    for cat, items in groups.items():
        st.markdown(f"### {cat}")
        for c in items:
            with st.expander(f"**{c['name']}**  ·  {c['horizon']} · {c['timeframe']} · {c['assets']}"):
                st.markdown(f"**What it is:** {c['what']}")
                if c.get("edge"):  st.markdown(f"**Where the edge comes from:** {c['edge']}")
                cc = st.columns(2)
                if c.get("works"): cc[0].markdown(f"✅ **Works when:** {c['works']}")
                if c.get("fails"): cc[1].markdown(f"⚠️ **Breaks when:** {c['fails']}")
                if c.get("retail"): st.markdown(f"**Retail-runnable?** {c['retail']}")
                st.caption(f"Shorts used: {'yes' if c['shorts'] else 'no'} · "
                           f"Run it live in the Strategies tab.")


# ------------------------------------------------------------------ Report
with TRP:
    st.subheader("📊 Daily fund report")
    st.caption("Full desk report: regime, optimal strategy + why, book NAV, risk, attribution, signals.")
    if st.button("Generate full report", key="report_generate"):
        with st.spinner("Building report — includes a full walk-forward optimisation "
                        "(~10s first time, cached after)."):
            rep_md = cached_report(symbol, start, end, interval, float(capital))
        st.download_button("⬇ Download (.md)", rep_md,
                           file_name=f"fund_report_{symbol}_{df.index[-1].date()}.md")
        st.markdown(rep_md)

# ------------------------------------------------------------------ Tutorial
with TUT:
    st.subheader("🎓 Full tutorial — how to operate this thing")

    st.markdown("""
### 0. The one rule
This is a **paper research platform**. It measures edge honestly; it does not create edge.
Every number is net of the costs you set. Nothing here trades real money.

### 1. Sidebar — your inputs
- **Ticker** — type *anything* (`AAPL`, `BTC-USD`, `RELIANCE.NS`, `^BSESN`, `GC=F`, `EURUSD=X`).
  On `HF_FEED=alpaca` it tries Alpaca first, then falls back to yfinance automatically.
- **Bar interval** — `1d` for swing/position work; `1h`/`15m`/`5m`/`1m` for intraday.
  *Intraday needs yfinance and has short history (1m ≈ 7 days, ≤1h ≈ 60–730 days).*
- **Lookback** — how much history to load. **Use 3–5 years for trustworthy stats.**
  1 year (~250 bars) gives noisy, low-confidence results — the app warns you when this happens.
- **Capital** — the notional the sizing math uses.

### 2. The cost sliders — how to actually set them
These are what separate a fantasy backtest from an honest one. Charged per side, on turnover:

| Slider | What it is | Sensible values |
|---|---|---|
| **Commission** | Broker fee | 0–1 bps (US equities, zero-commission) · 5–10 bps (small brokers) |
| **Slippage** | Price moves against you between decision and fill | 1–2 bps (SPY/AAPL) · 5–20 bps (small caps, crypto) |
| **Bid/ask spread** | Half-spread you cross entering AND exiting | 0.5–1 bps (SPY) · 2–5 bps (mid-cap) · 10–30 bps (micro-cap, alts) |
| **Overnight financing** | Annual borrow/margin on held exposure | 0 (cash long) · 300–800 bps (margin/shorts) · 800–1500 bps (hard-to-borrow) |

**How to use them:** set them to *your* broker's reality, then see if the edge survives.
A strategy that only works at 0 bps is not a strategy. Turn financing on for any short or
leveraged sleeve — it's the cost most retail backtests hide.

### 3. Tab-by-tab

**🔭 Scanner** — sweeps your whole universe: every asset × every strategy valid for its
asset class. States: **JUST FIRED** (triggered in the last few bars — act now) · **LIVE**
(already in) · **FORMING** (flat, but close to triggering) · **DORMANT**. Sorted by
expectancy, never by signal count. Click any row to get the candlestick with entry/exit
markers. Below it, the **opportunity heatmap** (asset × strategy, green pays / red bleeds)
and the asset ranking bar (most-profitable → most-lossable, graded 🟢/🟡/⚠️/❌).

**🩺 Post-mortem** — "why did we lose, and what do we change?". It dissects every trade's
MFE/MAE (how far it ran your way vs against you), names the failure mode with evidence
(gave the profit back · stopped inside the noise floor · whipsaw · costs ate it · one lucky
tail · entries are simply wrong · one side loses · regime mismatch), then proposes concrete
fixes and **tests each on held-out data**. It only recommends a change that genuinely
improves out-of-sample; otherwise it says so and tells you to trade something else. Use
this whenever a strategy is losing and you want a reason rather than a guess.

**⚡ Crypto Arb** — prices arbitrage net of depth, fees, transfer and latency. The waterfall
chart shows gross → net so you can see exactly which cost killed an "obvious" gap. Key
insight it will show you: **wider gaps die faster** (everyone can see them), so a 50bps
spread at 250ms latency has near-zero fill confidence while a boring 3bps one survives.

**🏠 Cockpit** — your live view for the selected ticker. Shows the regime, the recommended
strategy, and the **actionable signal table**: per strategy, what to do now, entry, stop,
target, R:R, size, and a `quality` verdict from that strategy's history *on this asset*.
Only strategies valid for the asset class are shown (no commodity seasonality on a stock).
`status` tells you: 🔔 NEW ENTRY = act now · ⚪ no setup = waiting for price to confirm.

**💼 Fund** — the book as a fund. NAV, P&L attribution (which strategy made/lost the money),
VaR/CVaR, **strategy correlation matrix**, the **trade blotter** (every historical trade with
entry/stop/target/exit/P&L/duration + net & average metrics), a beta-hedge calculator, a
trailing-stop/scale-out planner, and a multi-asset scan.

**🧪 Strategies** — run any of the 23 individually. Tune parameters live (including
categorical ones like CCI's reversion/momentum mode). Candlestick chart with entry/exit markers.

**📖 Explained** — the playbook: each strategy's edge, when it works, when it breaks.

**🎛️ Optimizer** — walk-forward. Three modes: best strategy for THIS stock · best stock for a
strategy · tune one. Ranked by a robust composite with a sanity gate, so a money-loser is
never recommended.

**🌡️ Regime** — trend/vol/volume classification of the current market, which strategy
categories to favour/avoid, and the gross-exposure scaling. The **big grey block** is just a
histogram of the trend regime across history — one bar because every bar classified the same
(usually a sign your window is too short or the trend threshold is strict).

**⚖️ Allocation** — how to split capital across strategies. HRP is the default (robust).

**🎲 Monte Carlo** — block-bootstraps your return stream thousands of times.
**p(loss)** = % of resampled paths ending below where you started.
**Horizon = the length of your loaded window** (1y lookback → 1y projection).
It runs the blended book (only asset-appropriate strategies) or one strategy.

**🔮 Forecast** — per-trade ticket: direction, size, stop, target, R:R, reasoning.

**🚀 Execution** — you're the PM. Paste a universe → **Auto-route** → it evaluates each
ticker's history/regime, tests only valid strategies, and assigns the best one *if* it passes
the gate. Then **Heads-up** (it proposes, you confirm each) or **Auto**. In-house SimBroker
by default; toggle Alpaca paper if keys are set.

**📊 Report** — the full downloadable desk report.

### 4. Reading the metrics
- **Sharpe** — return per unit of risk. >1 good, >2 rare/suspicious on daily bars.
- **Win rate** — *ignore it in isolation.* Trend strategies win ~35–40% and still make money
  because winners are 3× losers. Check **profit factor** (>1.3 is real) and Sharpe instead.
- **Max drawdown** — worst peak-to-trough. >50% means it effectively blew up.
- **Profit factor** — gross wins ÷ gross losses. <1 = losing.
- **Exposure** — % of time in the market.

### 5. Sizing: why $5,000 a trade?
That's the risk manager: **5% of capital per position**. It's a *survival* control, not a
profit-max setting. True profit-maximisation is full Kelly — mathematically optimal for
growth and the standard way accounts get destroyed, because it assumes your edge estimate is
exact (it never is). The platform uses vol-targeting + fractional Kelly + the 5% cap. If you
want more aggression, raise the cap in `risk/manager.py` — but understand you're trading
survival probability for expected return, and drawdowns scale faster than gains.
""")


# ------------------------------------------------------------------ Sizing / GTO
with TK:
    st.subheader("🎲 Sizing / GTO — the growth-optimal answer to 'maximise profit'")

    st.markdown(r"""
**The poker frame (this is exactly your GTO/exploitative split):**

In poker, *maximally exploitative* play has higher EV than GTO — **if your read is right**.
GTO gives up EV in exchange for being unexploitable when your read is wrong. You don't play
GTO because it prints the most money; you play it because it can't be destroyed.

Sizing is the identical trade-off. **Full Kelly is maximally exploitative**: it's the
mathematically maximal growth rate *if* your edge estimate (μ, σ) is exact. Your edge
estimate comes from a finite, noisy backtest — it is never exact. **Fractional Kelly is the
GTO play**: give up a little growth to be robust to your own estimation error.

The asymmetry is what makes it non-negotiable — overbetting is punished far harder than
underbetting: g(c*f*)/g(f*) = **2c − c²**
""")
    c = st.columns(4)
    c[0].metric("½ Kelly", "75% of max growth", delta="half the vol", delta_color="normal")
    c[1].metric("1× Kelly", "100% growth", delta="if μ,σ exact", delta_color="off")
    c[2].metric("2× Kelly", "0% growth", delta="zero, forever", delta_color="inverse")
    c[3].metric(">2× Kelly", "negative", delta="ruin w/ +edge", delta_color="inverse")

    src = st.radio("Analyse which return stream?", ["Blended book", "Single strategy"],
                   horizontal=True, key="kelly_source")
    if src == "Blended book":
        _bk = cached_book(symbol, start, end, interval, cost_bps, slip_bps)
        rets = _bk["book_returns"]
        label = f"blended book on {symbol}"
    else:
        kk = st.selectbox("Strategy", list(E.relevant_strategies(symbol)), key="kelly_strat")
        rets = E.run_any_strategy(kk, df, capital, cost_bps, slip_bps)["returns"]
        label = f"{kk} on {symbol}"

    lab = E.kelly_lab(rets)
    if not lab["ok"]:
        st.error(f"**{lab['msg']}**")
        st.caption("This is the correct answer, not a failure. Kelly on a negative edge is zero — "
                   "no sizing scheme, leverage, or risk overlay converts a losing strategy into a "
                   "winner. Find an edge first; size it second.")
    else:
        m = st.columns(5)
        m[0].metric("Edge (μ, ann)", f"{lab['mu']:.1%}")
        m[1].metric("Vol (σ, ann)", f"{lab['sigma']:.1%}")
        m[2].metric("Full Kelly f*", f"{lab['f_star']:.2f}×")
        m[3].metric("Peak growth", f"{lab['g_max']:.1%}/yr")
        m[4].metric("Edge t-stat", f"{lab['t_stat']:.1f}",
                    delta="significant" if lab["t_stat"] > 2 else "weak — size down")
        st.plotly_chart(E.kelly_curve_fig(lab), use_container_width=True)
        st.dataframe(lab["table"], use_container_width=True, hide_index=True)
        st.success(f"**Recommended: {lab['recommended_fraction']}× Kelly** — {lab['note']}")
        st.caption(f"Estimated from {lab['n_obs']} bars. `P(halving before doubling)` is a ruin proxy: "
                   f"at full Kelly it's ~37% — a one-in-three chance of halving your capital *while "
                   f"holding a genuine positive edge*. That's what 'maximum profit' actually costs.")

        st.markdown("### Why this platform caps at 5% per position")
        st.markdown(
            "f* above is the *unconstrained* growth-optimal leverage on a single return stream. "
            "It ignores: (1) that μ and σ drift, (2) that your strategies correlate in a crisis, "
            "(3) gap risk that skips your stop. The 5% position cap, the 1.5× gross cap and the "
            "12%/15% drawdown kill-switches exist because f* is computed from a world that doesn't "
            "have those. **Profit-max and risk-max aren't two dials you set to 11 — they're one "
            "dial, and ½ Kelly is where it should sit.** To change it, edit `risk/manager.py` "
            "(`position_cap`, `max_gross`) — but you're trading survival probability for expected "
            "return, and drawdown scales faster than growth.")

# ------------------------------------------------------------------ Scanner
with TSCAN:
    import dashboard.terminal_engine as T
    st.subheader("🔭 Universe scanner & opportunity heatmap")
    st.caption("Sweeps every asset × every strategy valid for its class. "
               "**JUST FIRED** = triggered in the last few bars (act now) · **LIVE** = already in · "
               "**FORMING** = flat but close to triggering · **DORMANT** = nothing. "
               "Sorted by expectancy, never by signal count.")
    default_uni = ("AAPL,NVDA,TSLA,MSFT,AMD,SPY,QQQ,GLD,BTC-USD,ETH-USD"
                   if FEED != "synthetic" else "AST1,AST2,AST3,AST4,AST5,AST6")
    uni = st.text_area("Universe (comma-separated)", value=default_uni, height=70, key="scanner_universe")
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    run_scan = sc1.button("🔭 Run scan", type="primary")
    only_act = sc2.checkbox("Actionable only", value=True)
    min_exp = sc3.slider("Min expectancy (%/trade)", -1.0, 2.0, -1.0, 0.05, key="scanner_min_expectancy")
    if run_scan:
        syms = [x.strip() for x in uni.replace("\n", ",").split(",") if x.strip()]
        with st.spinner(f"scanning {len(syms)} assets…"):
            st.session_state.scan_results = T.scan_universe(
                syms, start, end, interval, float(capital), cost_bps, slip_bps,
                spread_bps, fin_bps, min_expectancy=None if min_exp <= -1.0 else min_exp)
    scan = st.session_state.get("scan_results")
    if scan is None or scan.empty:
        st.info("Set a universe and hit **Run scan**.")
    else:
        act = T.actionable_only(scan)
        k = st.columns(5)
        k[0].metric("Combos", f"{len(scan)}")
        k[1].metric("Actionable", f"{len(act)}")
        k[2].metric("Just fired", f"{int((scan['state']=='JUST FIRED').sum())}")
        k[3].metric("Live", f"{int((scan['state']=='LIVE').sum())}")
        _b = act["expectancy"].max() if len(act) else float("nan")
        k[4].metric("Best expectancy", f"{_b:+.2f}%" if np.isfinite(_b) else "—")

        view = act if only_act else scan
        cols = [c for c in ["symbol", "asset_class", "strategy", "state", "side", "entry", "stop",
                            "target", "rr", "expectancy", "win_rate", "sharpe", "trades",
                            "verdict"] if c in view.columns]
        if view.empty:
            st.warning("Nothing actionable. That's a real answer — no edge, no trade.")
        else:
            def _cs(v):
                return {"JUST FIRED": "color:#F5A623;font-weight:600", "LIVE": "color:#4CAF7D",
                        "FORMING": "color:#3FB6C4"}.get(v, "color:#8A9BB0")
            def _ce(v):
                try:
                    return "color:#4CAF7D" if float(v) > 0 else "color:#E8472A"
                except Exception:
                    return ""
            st.dataframe(view[cols].style.map(_cs, subset=["state"]).map(_ce, subset=["expectancy"]),
                         use_container_width=True, hide_index=True, height=380)

            pick = st.selectbox("🔬 Inspect a signal", view.index,
                                format_func=lambda i: (f"{view.loc[i,'symbol']} · "
                                                       f"{view.loc[i,'strategy']} · "
                                                       f"{view.loc[i,'state']}"))
            if pick is not None:
                r = view.loc[pick]
                m = st.columns(6)
                m[0].metric("side", str(r.get("side", "—")))
                m[1].metric("entry", f"{r.get('entry','—')}")
                m[2].metric("stop", f"{r.get('stop','—')}")
                m[3].metric("target", f"{r.get('target','—')}")
                m[4].metric("R:R", f"{r.get('rr','—')}")
                m[5].metric("expectancy", f"{(r.get('expectancy') or 0):+.3f}%")
                st.markdown(f"**Why:** {r.get('why','')}")
                try:
                    d2 = E.load_bars(r["symbol"], start, end, interval)
                    p2 = E.REGISTRY[r["strategy"]]().generate(d2)
                    st.plotly_chart(E.candlestick_fig(d2, p2, f"{r['symbol']} — {r['strategy']}"),
                                    use_container_width=True)
                except Exception as ex:
                    st.caption(f"(chart unavailable: {ex})")

        st.markdown("### 🔥 Opportunity heatmap")
        metric_c = st.selectbox("Colour by", ["expectancy", "sharpe", "score", "win_rate"], index=0)
        st.plotly_chart(T.opportunity_heatmap(scan, metric_c), use_container_width=True)
        rank = T.asset_ranking(scan)
        st.plotly_chart(T.ranking_bar(rank), use_container_width=True)
        st.dataframe(rank, use_container_width=True, hide_index=True)
        st.caption("`grade` from the best available edge per asset: 🟢 strong · 🟡 tradeable · "
                   "⚠️ marginal · ❌ avoid. `losers` = strategies that lose money on that name.")

# ------------------------------------------------------------------ Post-mortem
with TPM:
    import dashboard.postmortem as PM
    st.subheader("🩺 Post-mortem — why did we lose, and what do we change?")
    st.caption("Runs the strategy, dissects every trade's **MFE/MAE** (how far it ran our way vs "
               "against us), classifies the failure mode, proposes concrete fixes, then **validates "
               "each fix on held-out data**. A fix is only recommended if it actually improves "
               "out-of-sample — otherwise it's reported as rejected.")

    pmk = st.selectbox("Strategy to review", list(E.relevant_strategies(symbol)), key="pm_strat")
    if st.button("🩺 Run post-mortem", type="primary"):
        with st.spinner("dissecting trades and validating fixes out-of-sample…"):
            st.session_state.pm = PM.prescribe(df, pmk, None, cost_bps, slip_bps,
                                               spread_bps, fin_bps)
    pm = st.session_state.get("pm")
    if pm is None:
        st.info("Pick a strategy and run the post-mortem.")
    else:
        d = pm["diagnosis"]
        s = d["stats"]
        m = st.columns(5)
        m[0].metric("Return", f"{s['TotalReturn']*100:+.1f}%")
        m[1].metric("Trades", f"{d['n_trades']}")
        m[2].metric("Win rate", f"{s.get('WinRate', float('nan')):.0%}" if 'WinRate' in s else "—")
        m[3].metric("Cost drag", f"{d.get('cost_drag_pct', 0):.1f}%")
        m[4].metric("Regime", d.get("regime", "?").split(" / ")[0])

        st.markdown("### 1️⃣ Why did we lose?")
        sev_icon = {"high": "🔴", "medium": "🟠", "low": "🟢"}
        for f in d["findings"]:
            with st.expander(f"{sev_icon.get(f['severity'],'•')} **{f['title']}**", expanded=True):
                st.markdown(f["detail"])
                st.markdown(f"**→ Fix to test:** {f['fix']}")

        ex = d["excursions"]
        if len(ex):
            st.markdown("### 2️⃣ The evidence — excursion analysis")
            e1, e2 = st.columns(2)
            wins, losses = ex[ex.win], ex[~ex.win]
            with e1:
                st.markdown("**Losing trades**")
                if len(losses):
                    st.metric("Avg run-up before losing", f"{losses['mfe_pct'].mean():+.2f}%")
                    st.metric("Avg give-back", f"{losses['give_back'].mean():.2f}%")
                    st.metric("Avg adverse excursion", f"{losses['mae_pct'].mean():.2f}%")
            with e2:
                st.markdown("**Winning trades**")
                if len(wins):
                    st.metric("Avg peak gain", f"{wins['mfe_pct'].mean():+.2f}%")
                    st.metric("Avg captured", f"{wins['pnl_pct'].mean():+.2f}%")
                    st.metric("Avg pain endured", f"{wins['mae_pct'].mean():.2f}%")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ex["mae_pct"], y=ex["mfe_pct"], mode="markers",
                                     marker=dict(size=9,
                                                 color=[E.GREEN if w else E.RUST for w in ex["win"]],
                                                 line=dict(width=0)),
                                     text=[f"{r.side} {r.bars}b · P&L {r.pnl_pct:+.2f}%"
                                           for r in ex.itertuples()],
                                     hovertemplate="%{text}<br>MAE %{x:.2f}% · MFE %{y:.2f}%<extra></extra>"))
            fig.add_hline(y=0, line=dict(color=E.MUTED, width=1))
            fig.add_vline(x=0, line=dict(color=E.MUTED, width=1))
            st.plotly_chart(E._style(fig, title="MFE vs MAE — green = winner, red = loser",
                                     xaxis=dict(title="max adverse excursion %", gridcolor=E.GRID),
                                     yaxis=dict(title="max favourable excursion %", gridcolor=E.GRID)),
                            use_container_width=True)
            st.caption("Points high on the Y-axis but red = trades that were winning and gave it "
                       "back (exit problem). Points hugging the left edge = trades that went "
                       "against us immediately (entry problem).")
            with st.expander("Per-trade excursion table"):
                st.dataframe(ex, use_container_width=True, hide_index=True)

        st.markdown("### 3️⃣ Fixes tested on held-out data")
        cands = pm["candidates"]
        cur_e = pm["current"]["oos_expectancy"]
        st.caption(f"Current configuration scores **{cur_e:+.3f}%** expectancy per trade "
                   f"out-of-sample." if cur_e is not None else "Current config could not be scored.")
        if cands:
            cd = pd.DataFrame(cands)[["label", "oos_expectancy", "delta_expectancy",
                                      "oos_trades", "improves", "rationale"]]
            def _ci(v):
                return "color:#4CAF7D;font-weight:600" if v else "color:#8A9BB0"
            st.dataframe(cd.style.map(_ci, subset=["improves"]),
                         use_container_width=True, hide_index=True)
        else:
            st.info("No testable parameter changes for this strategy/finding combination.")

        st.markdown("### 4️⃣ The updated strategy")
        if pm["best"]:
            st.success(pm["summary"])
            fx = PM.apply_fix(pmk, pm["base_params"], pm["best"])
            cc1, cc2 = st.columns(2)
            cc1.markdown("**Changed parameters**")
            cc1.dataframe(pd.DataFrame([{"param": k, "from": v[0], "to": v[1]}
                                        for k, v in fx["changed"].items()]),
                          use_container_width=True, hide_index=True)
            cc2.markdown("**Updated strategy**")
            cc2.code(fx["code"], language="python")
            if st.button("▶ Run the updated strategy now"):
                _st, _bt = E.run_strategy_backtest(pmk, df, fx["params"], cost_bps, slip_bps,
                                                   capital, spread_bps, fin_bps)
                a, b = st.columns(2)
                _old = E.run_strategy_backtest(pmk, df, pm["base_params"], cost_bps, slip_bps,
                                               capital, spread_bps, fin_bps)[1]
                a.metric("Before — return", f"{_old.stats['TotalReturn']*100:+.1f}%",
                         f"Sharpe {_old.stats['Sharpe']:.2f}")
                b.metric("After — return", f"{_bt.stats['TotalReturn']*100:+.1f}%",
                         f"Sharpe {_bt.stats['Sharpe']:.2f}")
                st.plotly_chart(E.equity_curve_fig(_bt.equity, f"{pmk} — updated"),
                                use_container_width=True)
        else:
            st.error(pm["summary"])
            st.caption("This is the correct answer when it happens. A post-mortem that always "
                       "finds a fix is overfitting with extra steps — the discipline is knowing "
                       "when the honest move is to stop tuning and trade something else.")

# ------------------------------------------------------------------ Crypto Arb
with TARB:
    import dashboard.terminal_engine as T
    from strategies.crypto_arb import CryptoArbEngine, book_from_quotes
    st.subheader("⚡ Crypto arbitrage desk")
    st.caption("Prices arbitrage the way a desk does — **net of depth, fees, transfer and "
               "latency**. A top-of-book gap is not an edge; this shows what survives.")
    ac = st.columns(4)
    latency = ac[0].number_input("Round-trip latency (ms)", 1.0, 5000.0, 250.0, 10.0)
    min_net = ac[1].number_input("Min net edge (bps)", 0.0, 100.0, 1.0, 0.5)
    max_not = ac[2].number_input("Max clip ($)", 1_000.0, 5_000_000.0, 50_000.0, 5_000.0)
    volann = ac[3].number_input("Pair vol (annualised)", 0.05, 3.0, 0.6, 0.05)
    pre_pos = st.checkbox("Inventory pre-positioned on both venues (skip transfer cost)", value=True)
    books_df = st.data_editor(pd.DataFrame([
        {"venue": "binance", "symbol": "BTC/USDT", "mid": 67000.0, "spread_bps": 1.5, "depth_usd": 400_000},
        {"venue": "okx", "symbol": "BTC/USDT", "mid": 67035.0, "spread_bps": 2.0, "depth_usd": 250_000},
        {"venue": "kraken", "symbol": "BTC/USDT", "mid": 67090.0, "spread_bps": 4.0, "depth_usd": 90_000},
        {"venue": "coinbase", "symbol": "BTC/USDT", "mid": 67120.0, "spread_bps": 3.0, "depth_usd": 150_000},
    ]), use_container_width=True, hide_index=True, num_rows="dynamic", key="arb_book_editor")
    if st.button("⚡ Scan for arbitrage", type="primary"):
        eng = CryptoArbEngine(latency_ms=float(latency), min_net_bps=float(min_net),
                              max_notional_usd=float(max_not))
        bks = [book_from_quotes(r["venue"], r["symbol"], float(r["mid"]), float(r["spread_bps"]),
                                float(r["depth_usd"]), seed=i) for i, r in books_df.iterrows()]
        st.session_state.arb = eng.scan_spatial(bks, float(volann), pre_positioned=pre_pos)
    opps = st.session_state.get("arb")
    if opps:
        execu = [o for o in opps if o.net_bps >= min_net and o.size > 0]
        k = st.columns(4)
        k[0].metric("Gaps seen", f"{len(opps)}")
        k[1].metric("Executable", f"{len(execu)}")
        k[2].metric("Best gross", f"{max(o.gross_bps for o in opps):.1f} bps")
        k[3].metric("Best NET", f"{max(o.net_bps for o in opps):+.1f} bps")
        adf = pd.DataFrame([o.to_row() for o in opps])
        show = ["buy_venue", "sell_venue", "gross_bps", "slippage_bps", "fees_bps", "transfer_bps",
                "latency_decay_bps", "net_bps", "notional_usd", "confidence", "half_life_ms",
                "verdict"]
        def _cn(v):
            try:
                return "color:#4CAF7D;font-weight:600" if float(v) > 0 else "color:#E8472A"
            except Exception:
                return ""
        st.dataframe(adf[show].style.map(_cn, subset=["net_bps"]),
                     use_container_width=True, hide_index=True)
        pk = st.selectbox("Break down an opportunity", range(len(opps)),
                          format_func=lambda i: (f"{opps[i].buy_venue} → {opps[i].sell_venue} · "
                                                 f"gross {opps[i].gross_bps:.1f} → net "
                                                 f"{opps[i].net_bps:+.1f}bps"))
        o = opps[pk]
        st.plotly_chart(T.arb_waterfall(o), use_container_width=True)
        st.markdown(f"**Why:** {o.why}")
        if o.legs:
            st.dataframe(pd.DataFrame(o.legs), use_container_width=True, hide_index=True)
        st.caption(f"Edge half-life ≈ **{o.half_life_ms:.0f}ms**; your round-trip is "
                   f"**{latency:.0f}ms** → confidence **{o.confidence:.0%}**. Wider gaps die "
                   f"faster — everyone can see them.")
        if not execu:
            st.error("**Nothing executable.** Every visible gap dies to costs — the normal state "
                     "of crypto arb. Gaps visible from a laptop are the ones colocated desks "
                     "didn't want.")

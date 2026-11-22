"""
dashboard/engine.py
-------------------
Pure compute + chart builders for the dashboard. NO streamlit imports here, so
every function is unit-testable headlessly and app.py stays a thin layout shell.
All figures are plotly go.Figure in the amber-on-dark house style.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import make_feed, COST_BPS, SLIPPAGE_BPS, INITIAL_CAPITAL, START, END
from strategies.registry import REGISTRY, PANEL_REGISTRY, PAIR_REGISTRY, ADVANCED_REGISTRY, UNIVERSE
from backtest.engine import run_backtest
from backtest.optimizer import walk_forward
from backtest.metrics import summary
from risk.allocation import allocate, compare_allocations, ALLOCATORS
from risk.sizing import build_trade_plan
from core.regime import latest_regime, route, classify
from montecarlo.simulator import bootstrap_strategy, run_simulation
from report.forecast import forecast_trade

# ---- house style ----
AMBER, RUST, TEAL = "#F5A623", "#E8472A", "#2ABFB0"
GREEN, MUTED, BG = "#4CAF7D", "#8A9BB0", "#0B0E14"
CARD, GRID, TEXT = "#111520", "#1A2030", "#DDE4EE"

_LAYOUT = dict(
    plot_bgcolor=BG, paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color=TEXT, size=11),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID),
)

def _style(fig, **extra):
    fig.update_layout(**{**_LAYOUT, **extra})
    return fig


# ---------------------------------------------------------------- data helpers
def get_feed():
    return make_feed()

def load_bars(symbol="DEMO", start=START, end=END, interval="1d"):
    return get_feed().history(symbol, start, end, interval=interval)

def universe_table() -> pd.DataFrame:
    df = pd.DataFrame(UNIVERSE, columns=["key", "category", "horizon", "asset_classes", "status"])
    df["asset_classes"] = df["asset_classes"].apply(lambda x: ", ".join(x))
    return df


# ---------------------------------------------------------------- single-series backtest
def single_strategies() -> dict:
    """Strategies the Backtest Lab can run interactively (single-series + vol_arb)."""
    d = dict(REGISTRY)
    d["vol_arb"] = ADVANCED_REGISTRY["vol_arb"]
    return d

def run_strategy_backtest(key: str, df: pd.DataFrame, params: dict,
                          cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS,
                          capital=INITIAL_CAPITAL, spread_bps=0.0, financing_bps_annual=0.0):
    cls = single_strategies()[key]
    strat = cls(**params)
    _cost = dict(cost_bps=cost_bps, slippage_bps=slippage_bps, spread_bps=spread_bps,
                 financing_bps_annual=financing_bps_annual, initial_capital=capital)
    if key == "vol_arb":
        bt = run_backtest(strat.vol_equity_df(df), strat.generate(df), **_cost)
    else:
        bt = run_backtest(df, strat.generate(df), **_cost)
    return strat, bt


def equity_fig(bt, title="Equity curve") -> go.Figure:
    eq = bt.equity
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Equity",
                             line=dict(color=AMBER, width=2), fill="tozeroy",
                             fillcolor="rgba(245,166,35,0.06)"))
    fig.add_hline(y=eq.iloc[0], line=dict(color=MUTED, dash="dash", width=1))
    return _style(fig, title=title, yaxis=dict(gridcolor=GRID, tickprefix="$", tickformat=",.0f"))


def drawdown_fig(bt) -> go.Figure:
    eq = bt.equity
    dd = (eq / eq.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy",
                               line=dict(color=RUST, width=1), name="Drawdown"))
    return _style(fig, title="Underwater (drawdown %)",
                  yaxis=dict(gridcolor=GRID, ticksuffix="%"))


def positions_fig(bt) -> go.Figure:
    fig = go.Figure(go.Scatter(x=bt.position.index, y=bt.position.values,
                               line=dict(color=TEAL, width=1), name="Position"))
    return _style(fig, title="Executed position (post 1-bar lag)")


def optimizer_report(key: str, df: pd.DataFrame, metric="Sharpe",
                     cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> dict:
    cls = single_strategies()[key]
    wf = walk_forward(cls, df, n_splits=4, metric=metric,
                      cost_bps=cost_bps, slippage_bps=slippage_bps)
    default = cls()
    base = run_backtest(df, default.generate(df) if key != "vol_arb" else default.generate(df),
                        cost_bps=cost_bps, slippage_bps=slippage_bps) \
        if key != "vol_arb" else run_backtest(default.vol_equity_df(df), default.generate(df),
                                              cost_bps=cost_bps, slippage_bps=slippage_bps)
    return {"walk_forward": wf, "default_params": default.params, "default_stats": base.stats}


# ---------------------------------------------------------------- regime
def regime_report(df: pd.DataFrame) -> dict:
    reg = latest_regime(df)
    r = route(reg)
    hist = classify(df)
    return {"regime": reg, "route": r, "history": hist}

def regime_gauge(reg) -> go.Figure:
    """Gauge of realised vol with the regime label."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=reg.rv_annual * 100,
        number=dict(suffix="%", font=dict(color=AMBER)),
        title=dict(text=f"Realised Vol · {reg.volatility}", font=dict(color=MUTED, size=13)),
        gauge=dict(axis=dict(range=[0, 60], tickcolor=MUTED),
                   bar=dict(color=AMBER),
                   steps=[dict(range=[0, 15], color="#14311f"),
                          dict(range=[15, 30], color="#3a2f14"),
                          dict(range=[30, 60], color="#3a1714")]),
    ))
    return _style(fig, height=240)

def regime_trend_fig(hist: pd.DataFrame) -> go.Figure:
    counts = hist["trend"].value_counts()
    colors = {"bull": GREEN, "bear": RUST, "sideways": MUTED}
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                           marker=dict(color=[colors.get(i, AMBER) for i in counts.index])))
    return _style(fig, title="Trend regime distribution (history)")


# ---------------------------------------------------------------- allocation
def allocation_report(returns: pd.DataFrame) -> dict:
    table = compare_allocations(returns)
    weights = {m: allocate(returns, method=m) for m in ALLOCATORS}
    return {"table": table, "weights": weights}

def weights_fig(weights: pd.Series, title="Weights") -> go.Figure:
    colors = [AMBER if v >= 0 else RUST for v in weights.values]
    fig = go.Figure(go.Bar(x=weights.index, y=weights.values * 100,
                           marker=dict(color=colors),
                           text=[f"{v:.0%}" for v in weights.values], textposition="outside"))
    return _style(fig, title=title, yaxis=dict(gridcolor=GRID, ticksuffix="%"))


# ---------------------------------------------------------------- monte carlo
def montecarlo_report(returns: pd.Series, capital=INITIAL_CAPITAL) -> dict:
    rob = bootstrap_strategy(returns.replace(0, np.nan).dropna(), initial_value=capital,
                             n_sims=3000, seed=7)
    return {"robustness": rob}

def mc_terminal_fig(rob) -> go.Figure:
    fig = go.Figure(go.Histogram(x=rob.terminal_dist, nbinsx=60,
                                 marker=dict(color=AMBER, opacity=0.7)))
    fig.add_vline(x=rob.p05_terminal, line=dict(color=RUST, dash="dash"),
                  annotation_text="p05")
    fig.add_vline(x=rob.p50_terminal, line=dict(color=TEAL, dash="dash"),
                  annotation_text="median")
    return _style(fig, title="Bootstrapped terminal equity distribution",
                  xaxis=dict(gridcolor=GRID, tickprefix="$", tickformat=",.0f"))

def mc_drawdown_fig(rob) -> go.Figure:
    fig = go.Figure(go.Histogram(x=rob.maxdd_dist * 100, nbinsx=50,
                                 marker=dict(color=RUST, opacity=0.7)))
    fig.add_vline(x=rob.median_maxdd * 100, line=dict(color=AMBER, dash="dash"),
                  annotation_text="median")
    return _style(fig, title="Max-drawdown distribution",
                  xaxis=dict(gridcolor=GRID, ticksuffix="%"))


# ---------------------------------------------------------------- forecast
def forecast_report(key: str, df: pd.DataFrame, symbol: str, capital=INITIAL_CAPITAL,
                    sizing="vol_target") -> dict:
    cls = single_strategies()[key]
    strat = cls()
    fc = forecast_trade(strat, df, symbol, capital=capital, sizing_method=sizing)
    return {"forecast": fc}


# ---------------------------------------------------------------- multi-strategy book
def build_book(df: pd.DataFrame, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS,
               target_vol: float = 0.12, symbol: str | None = None) -> dict:
    """Backtest every single-series strategy, allocate across them, apply regime
    overlay, then VOL-TARGET the whole book to a realistic annual vol so the
    return stream can't compound to absurd numbers on trending real data."""
    rets, cats = {}, {}
    _pool = relevant_strategies(symbol) if symbol else REGISTRY
    for key, cls in _pool.items():
        strat = cls()
        bt = run_backtest(df, strat.generate(df), cost_bps=cost_bps, slippage_bps=slippage_bps)
        rets[cls.meta.name] = bt.returns
        cats[cls.meta.name] = cls.meta.category
    rdf = pd.DataFrame(rets).dropna(how="all").fillna(0.0)
    w = allocate(rdf, method="hrp")
    reg = latest_regime(df)
    from core.regime import strategy_multipliers
    mult = strategy_multipliers(reg, cats)
    adj = (w * pd.Series(mult)).fillna(0.0)
    adj = adj / adj.sum() if adj.sum() > 0 else w
    raw_book = (rdf * adj).sum(axis=1) * route(reg)["gross_exposure"]

    # vol-target: scale the whole book so realised vol ≈ target (leverage capped)
    realized = raw_book.rolling(63, min_periods=20).std() * np.sqrt(252)
    lever = (target_vol / realized).clip(upper=3.0).shift(1).fillna(1.0)
    book = (raw_book * lever).clip(-0.25, 0.25)   # hard daily-return sanity clip
    return {"returns": rdf, "hrp_weights": w, "regime_weights": adj,
            "multipliers": mult, "book_returns": book, "regime": reg}


# ============================================================================
# Universal strategy runner — makes ALL 16 strategies runnable/visible in the UI
# regardless of their data shape (single / panel / pair / advanced). Each returns
# a unified result: equity curve, return stream, stats, and a plain explanation.
# ============================================================================
from types import SimpleNamespace
from core.altdata import (synthetic_macro, synthetic_asset_bars, synthetic_fx_panel,
                          synthetic_earnings, synthetic_ma_deals, synthetic_index_events,
                          synthetic_funding)

# grouped, ordered list for the UI (key, group, one-line what-it-is)
ALL_STRATEGIES = [
    ("vix_rsi",         "Single-asset",  "Mean reversion: dip-buy oversold RSI, VIX-gated"),
    ("breakout",        "Single-asset",  "Trend: Donchian channel breakout with ATR trail"),
    ("trendline",       "Single-asset",  "Trend: trade the sign of a rolling regression slope"),
    ("commodity_trend", "Single-asset",  "Trend + month-of-year seasonality blend"),
    ("ichimoku",        "Single-asset",  "Ichimoku cloud: full trend-confluence system"),
    ("cci",             "Single-asset",  "CCI oscillator: fade extremes or ride breakouts"),
    ("macd",            "Single-asset",  "MACD line vs signal — momentum trend"),
    ("bollinger_rev",   "Single-asset",  "Bollinger band mean-reversion"),
    ("overnight_drift", "Single-asset",  "Hold close→open only (overnight anomaly)"),
    ("vol_arb",         "Single-asset",  "Volatility: harvest the variance risk premium"),
    ("multifactor",     "Equity basket", "Long/short on momentum+value+low-vol+trend blend"),
    ("equity_ls",       "Equity basket", "Long/short on beta-neutral (residual) momentum"),
    ("pairs_statarb",   "Pairs",         "Cointegration pairs, static OLS hedge ratio"),
    ("kalman_pairs",    "Pairs",         "Cointegration pairs, adaptive Kalman hedge ratio"),
    ("fixed_income_arb","Pairs",         "Duration/curve convergence (e.g. IEF/TLT)"),
    ("index_arb",       "Pairs",         "ETF vs future basis convergence (e.g. SPY/ES=F)"),
    ("global_macro",    "Macro / FX",    "Rules on growth/inflation/rates tilt a multi-asset book"),
    ("fx_carry",        "Macro / FX",    "Long high-yield / short low-yield currencies"),
    ("crypto_funding",  "Crypto",        "Delta-neutral perp funding-rate carry"),
    ("event_earnings",  "Event / flow",  "Post-earnings drift (PEAD)"),
    ("merger_arb",      "Event / flow",  "Cash M&A deal-spread capture"),
    ("index_rebal",     "Event / flow",  "Front-run index add/delete forced flows"),
    ("mm_scalper",      "Simulation",    "Market-making inventory sim (infra-bound)"),
]

def _wrap(equity: pd.Series, returns: pd.Series, note: str,
          position: pd.Series | None = None) -> dict:
    stats = summary(equity, returns)
    stats["FinalEquity"] = float(equity.iloc[-1])
    return dict(equity=equity, returns=returns,
                position=position if position is not None else pd.Series(0.0, index=equity.index),
                stats=stats, note=note)

def run_any_strategy(key: str, df: pd.DataFrame, capital: float = INITIAL_CAPITAL,
                     cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> dict:
    """Run any of the 16 strategies, synthesising the extra data each one needs
    (baskets, pairs, macro series, events) so everything is visible in the UI."""
    cls = _get_cls(key)
    strat = cls()
    idx = df.index

    # ---- single-asset ----
    if key in REGISTRY:
        bt = run_backtest(df, strat.generate(df), initial_capital=capital,
                          cost_bps=cost_bps, slippage_bps=slippage_bps)
        note = strat._explain(df) if hasattr(strat, "_explain") else cls.meta.description
        return _wrap(bt.equity, bt.returns, note, bt.position)

    if key == "vol_arb":
        bt = run_backtest(strat.vol_equity_df(df), strat.generate(df), initial_capital=capital,
                          cost_bps=cost_bps, slippage_bps=slippage_bps)
        return _wrap(bt.equity, bt.returns, strat._explain(df), bt.position)

    # ---- equity basket (panel) ----
    if key in PANEL_REGISTRY:
        bars = _synth_basket(idx, n=10)
        w = strat.generate_panel(bars)
        rets = pd.DataFrame({s: b["close"].pct_change() for s, b in bars.items()}).fillna(0)
        port = (w.shift(1) * rets).sum(axis=1)
        eq = capital * (1 + port).cumprod()
        return _wrap(eq, port, f"{cls.meta.description} — dollar-neutral over a {len(bars)}-name basket.")

    # ---- pairs ----
    if key in PAIR_REGISTRY:
        a, b = _synth_pair(df)
        pos = strat.generate_pair(a, b)
        from strategies.pairs_statarb import pairs_equity_df as _pe
        from strategies.kalman_pairs import pairs_equity_df as _ke
        eq_df = _ke(strat, a, b) if key == "kalman_pairs" else _pe(strat, a, b)
        bt = run_backtest(eq_df, pos, initial_capital=capital, cost_bps=cost_bps, slippage_bps=slippage_bps)
        note = f"{cls.meta.description} — cointegration-tested synthetic pair."
        return _wrap(bt.equity, bt.returns, note, bt.position)

    # ---- advanced ----
    if key == "global_macro":
        macro = synthetic_macro(periods=150, seed=0); assets = synthetic_asset_bars(macro, seed=1)
        w = strat.generate_macro(macro, assets)
        rets = pd.DataFrame({k: v["close"].pct_change() for k, v in assets.items()}).fillna(0)
        port = (w.shift(1) * rets).sum(axis=1); eq = capital * (1 + port).cumprod()
        return _wrap(eq, port, strat.explain(macro, assets))
    if key == "fx_carry":
        spots, rates = synthetic_fx_panel(periods=len(idx), seed=2)
        port = strat.carry_returns(spots, rates); eq = capital * (1 + port).cumprod()
        return _wrap(eq, port, cls.meta.description)
    if key == "crypto_funding":
        fund = synthetic_funding(df)
        bt = run_backtest(strat.equity_df(fund), strat.generate(fund), initial_capital=capital,
                          cost_bps=cost_bps, slippage_bps=slippage_bps)
        return _wrap(bt.equity, bt.returns, strat.latest_signal(fund, "PERP").reason, bt.position)
    if key == "event_earnings":
        ev = synthetic_earnings(df); pos = strat.generate(df, ev)
        bt = run_backtest(df, pos, initial_capital=capital, cost_bps=cost_bps, slippage_bps=slippage_bps)
        return _wrap(bt.equity, bt.returns, f"{cls.meta.description} — {len(ev)} synthetic earnings events.", bt.position)
    if key == "merger_arb":
        deals = synthetic_ma_deals(n_deals=40)
        bt = run_backtest(strat.equity_df(deals, idx), pd.Series(1.0, index=idx),
                          initial_capital=capital, cost_bps=cost_bps, slippage_bps=slippage_bps)
        return _wrap(bt.equity, bt.returns, f"{cls.meta.description} — {len(deals)} synthetic deals.")
    if key == "index_rebal":
        ev = synthetic_index_events(df); pos = strat.generate(df, ev)
        bt = run_backtest(df, pos, initial_capital=capital, cost_bps=cost_bps, slippage_bps=slippage_bps)
        return _wrap(bt.equity, bt.returns, f"{cls.meta.description} — {len(ev)} synthetic index events.", bt.position)
    if key == "mm_scalper":
        bt = run_backtest(strat.equity_df(df), pd.Series(1.0, index=idx), initial_capital=capital)
        return _wrap(bt.equity, bt.returns, cls.meta.description + " (simulation, not deployable).")

    raise KeyError(key)

def _get_cls(key):
    from strategies.registry import get
    return get(key)

def _synth_basket(idx, n=10):
    rng = np.random.default_rng(7)
    mkt = np.cumsum(rng.standard_normal(len(idx)) * 0.01)
    bars = {}
    for i in range(n):
        io = np.cumsum(rng.standard_normal(len(idx)) * 0.008 + (0.0003 if i % 3 == 0 else -0.0002))
        px = 100 * np.exp(mkt * (0.8 + 0.4 * (i / n)) + io)
        bars[f"S{i}"] = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)
    return bars

def _synth_pair(df):
    a = df["close"]
    rng = np.random.default_rng(11)
    sp = np.zeros(len(a))
    for i in range(1, len(a)):
        sp[i] = 0.93 * sp[i - 1] + rng.standard_normal() * 0.02
    b = pd.Series(a.values * np.exp(sp + 0.2), index=a.index)
    return a, b

def equity_curve_fig(equity: pd.Series, title="Equity") -> go.Figure:
    fig = go.Figure(go.Scatter(x=equity.index, y=equity.values, line=dict(color=AMBER, width=2),
                               fill="tozeroy", fillcolor="rgba(245,166,35,0.06)"))
    fig.add_hline(y=equity.iloc[0], line=dict(color=MUTED, dash="dash", width=1))
    return _style(fig, title=title, yaxis=dict(gridcolor=GRID, tickprefix="$", tickformat=",.0f"))

def dd_fig(equity: pd.Series) -> go.Figure:
    dd = (equity / equity.cummax() - 1) * 100
    fig = go.Figure(go.Scatter(x=dd.index, y=dd.values, fill="tozeroy", line=dict(color=RUST, width=1)))
    return _style(fig, title="Underwater (drawdown %)", yaxis=dict(gridcolor=GRID, ticksuffix="%"))


# ============================================================================
# Cockpit: live signals across all 16, recommended strategy, signal-marked charts
# ============================================================================
def live_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Current stance + key facts for every strategy — the 'what do I trade now' TLDR."""
    rows = []
    for key, group, desc in ALL_STRATEGIES:
        cls = _get_cls(key)
        meta = cls.meta
        stance, conv = "—", None
        try:
            res = run_any_strategy(key, df)
            pos = res.get("position")
            if pos is not None and (pos != 0).any():
                last = float(pos.iloc[-1])
                stance = "LONG" if last > 0 else "SHORT" if last < 0 else "FLAT"
            sh = res["stats"]["Sharpe"]
        except Exception:
            sh = float("nan")
        rows.append({
            "strategy": key, "category": meta.category, "stance": stance,
            "horizon": meta.horizon, "timeframe": meta.timeframe,
            "assets": ", ".join(meta.asset_classes), "backtest_Sharpe": round(sh, 2),
        })
    return pd.DataFrame(rows)


def recommended_strategy(df: pd.DataFrame) -> dict:
    """Pick the best-fit LIVE strategy for the current regime: a favoured category
    with the strongest recent Sharpe. Honest, regime-conditioned recommendation."""
    reg = latest_regime(df)
    r = route(reg)
    favoured = set(r["favour"])
    sigs = live_signals(df)
    pool = sigs[sigs["category"].isin(favoured)].copy()
    if pool.empty:
        pool = sigs.copy()
    pool = pool.sort_values("backtest_Sharpe", ascending=False)
    best = pool.iloc[0]
    return {"regime": reg.label(), "favour": r["favour"], "avoid": r["avoid"],
            "gross": r["gross_exposure"], "pick": best["strategy"],
            "pick_category": best["category"], "pick_sharpe": best["backtest_Sharpe"]}


def signal_price_fig(df: pd.DataFrame, position: pd.Series, title="Price with signals") -> go.Figure:
    """Price line with entry markers where the executed position flips — the chart
    that shows WHERE the strategy trades, not just the equity curve."""
    px = df["close"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=px.index, y=px.values, line=dict(color=MUTED, width=1.2),
                             name="Price"))
    flips = position.fillna(0).diff().fillna(position.fillna(0))
    longs = position[(flips != 0) & (position > 0)]
    shorts = position[(flips != 0) & (position < 0)]
    exits = position[(flips != 0) & (position == 0)]
    fig.add_trace(go.Scatter(x=longs.index, y=px.reindex(longs.index), mode="markers",
                             marker=dict(color=GREEN, size=9, symbol="triangle-up"), name="Long"))
    fig.add_trace(go.Scatter(x=shorts.index, y=px.reindex(shorts.index), mode="markers",
                             marker=dict(color=RUST, size=9, symbol="triangle-down"), name="Short"))
    fig.add_trace(go.Scatter(x=exits.index, y=px.reindex(exits.index), mode="markers",
                             marker=dict(color=MUTED, size=6, symbol="x"), name="Exit"))
    return _style(fig, title=title, yaxis=dict(gridcolor=GRID, tickprefix="$", tickformat=",.2f"))


def winrate_context(stats: dict) -> str | None:
    """Explain a low win rate when it's by design (trend/breakout: few big wins)."""
    wr = stats.get("WinRate")
    pf = stats.get("ProfitFactor")
    if wr is None:
        return None
    if wr < 0.40 and pf is not None and pf > 1.0:
        return ("Low win rate is EXPECTED here — trend/breakout strategies win a minority of "
                "trades but the winners are much larger than the losers (profit factor "
                f"{pf:.2f}). Win rate alone is the wrong lens; payoff asymmetry is the point.")
    if wr < 0.40:
        return ("Low win rate AND profit factor ≤ 1 — this configuration isn't working on this "
                "data. Try the Optimizer, or a different asset/timeframe.")
    return None


def run_real_pair(key: str, sym_a: str, sym_b: str, start: str, end: str,
                  capital: float = INITIAL_CAPITAL) -> dict:
    """Run a pairs strategy on TWO real tickers so the long/short legs are explicit."""
    feed = get_feed()
    a = feed.history(sym_a, start, end)["close"]
    b = feed.history(sym_b, start, end)["close"]
    a, b = a.align(b, join="inner")
    cls = _get_cls(key)
    strat = cls()
    pos = strat.generate_pair(a, b)
    from strategies.pairs_statarb import pairs_equity_df as _pe
    from strategies.kalman_pairs import pairs_equity_df as _ke
    eq_df = _ke(strat, a, b) if key == "kalman_pairs" else _pe(strat, a, b)
    bt = run_backtest(eq_df, pos, initial_capital=capital, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    sig = strat.latest_signal(a, b, sym_a, sym_b)
    beta = getattr(getattr(strat, "fit", None), "beta", None)
    if beta is None and getattr(strat, "state", None) is not None:
        beta = float(strat.state.beta.iloc[-1])
    d = sig.direction
    legs = (f"LONG {sym_a} / SHORT {beta:.2f}×{sym_b}" if d > 0 else
            f"SHORT {sym_a} / LONG {beta:.2f}×{sym_b}" if d < 0 else "FLAT (no dislocation)") \
        if beta is not None else "n/a"
    return {"equity": bt.equity, "returns": bt.returns, "stats": bt.stats,
            "spread": (a.apply(np.log) - (getattr(getattr(strat,'fit',None),'alpha',0) or 0)
                       - (beta or 1) * b.apply(np.log)),
            "legs": legs, "reason": sig.reason, "position": pos, "a": a, "b": b}


# ============================================================================
# FUND DESK: trade blotter, P&L attribution, risk panel, hedge calc, scan, report
# ============================================================================
def blotter_summary(blot: pd.DataFrame) -> dict:
    """Net + average metrics for a blotter — what the desk actually reports."""
    if blot is None or blot.empty:
        return {}
    closed = blot[blot["status"] == "closed"]
    wins = closed[closed["pnl_pct"] > 0]["pnl_pct"]
    losses = closed[closed["pnl_pct"] < 0]["pnl_pct"]
    longs = closed[closed["side"] == "LONG"]
    shorts = closed[closed["side"] == "SHORT"]
    return {
        "trades": int(len(closed)),
        "net_pnl_pct": round(float(closed["pnl_pct"].sum()), 2),
        "avg_pnl_pct": round(float(closed["pnl_pct"].mean()), 2) if len(closed) else 0.0,
        "win_rate": round(float((closed["pnl_pct"] > 0).mean()), 3) if len(closed) else 0.0,
        "avg_win_pct": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss_pct": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "profit_factor": round(float(wins.sum() / -losses.sum()), 2) if losses.sum() < 0 else None,
        "avg_bars_held": int(closed["bars_held"].mean()) if len(closed) else 0,
        "longs": int(len(longs)), "shorts": int(len(shorts)),
        "best_pct": round(float(closed["pnl_pct"].max()), 2) if len(closed) else 0.0,
        "worst_pct": round(float(closed["pnl_pct"].min()), 2) if len(closed) else 0.0,
    }


def trade_blotter(df: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    """Historical trade log: every entry/exit with side, duration, and P&L.
    This is the 'historical signals' record — the desk blotter."""
    from core.indicators import atr as _atr_f
    pos = position.fillna(0.0)
    px = df["close"]
    _atr_s = _atr_f(df, 14)
    trades, cur, entry_i = [], 0.0, None
    for i in range(len(pos)):
        p = float(pos.iloc[i])
        if p != cur:
            if cur != 0.0 and entry_i is not None:
                _ep = float(px.iloc[entry_i]); _a = float(_atr_s.iloc[entry_i] or 0); _d = np.sign(cur)
                trades.append(dict(
                    entry=str(pos.index[entry_i].date()), exit=str(pos.index[i].date()),
                    side="LONG" if cur > 0 else "SHORT", bars_held=i - entry_i,
                    entry_px=round(_ep, 2),
                    stop=round(_ep - _d * 2 * _a, 2), target=round(_ep + _d * 3 * _a, 2),
                    exit_px=round(float(px.iloc[i]), 2),
                    pnl_pct=round(float((px.iloc[i] / _ep - 1) * _d) * 100, 2),
                    status="closed"))
            entry_i = i if p != 0.0 else None
            cur = p
    if cur != 0.0 and entry_i is not None:      # still-open trade
        _ep = float(px.iloc[entry_i]); _a = float(_atr_s.iloc[entry_i] or 0); _d = np.sign(cur)
        trades.append(dict(
            entry=str(pos.index[entry_i].date()), exit="OPEN",
            side="LONG" if cur > 0 else "SHORT", bars_held=len(pos) - 1 - entry_i,
            entry_px=round(_ep, 2),
            stop=round(_ep - _d * 2 * _a, 2), target=round(_ep + _d * 3 * _a, 2),
            exit_px=round(float(px.iloc[-1]), 2),
            pnl_pct=round(float((px.iloc[-1] / _ep - 1) * _d) * 100, 2),
            status="open"))
    return pd.DataFrame(trades)


def pnl_attribution(book: dict, capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """Cumulative P&L contribution per strategy — who made/lost the money."""
    contrib = (book["returns"] * book["regime_weights"]).cumsum() * capital
    final = contrib.iloc[-1].sort_values(ascending=False)
    return pd.DataFrame({"strategy": final.index,
                         "pnl_$": final.round(0).values,
                         "weight": [round(book["regime_weights"].get(k, 0), 3) for k in final.index]})


def attribution_fig(attr: pd.DataFrame) -> go.Figure:
    colors = [GREEN if v >= 0 else RUST for v in attr["pnl_$"]]
    fig = go.Figure(go.Bar(x=attr["strategy"], y=attr["pnl_$"], marker=dict(color=colors),
                           text=[f"${v:,.0f}" for v in attr["pnl_$"]], textposition="outside"))
    return _style(fig, title="P&L attribution by strategy",
                  yaxis=dict(gridcolor=GRID, tickprefix="$", tickformat=",.0f"))


def risk_panel(book: dict, capital: float = INITIAL_CAPITAL) -> dict:
    """Desk risk snapshot: historical VaR/CVaR of the live book + correlations."""
    r = book["book_returns"].dropna()
    var95 = -np.percentile(r, 5) * capital
    var99 = -np.percentile(r, 1) * capital
    tail = r[r <= np.percentile(r, 5)]
    cvar95 = -tail.mean() * capital if len(tail) else var95
    return {"VaR95_1d": var95, "VaR99_1d": var99, "CVaR95_1d": cvar95,
            "ann_vol": float(r.std() * np.sqrt(252)),
            "corr": book["returns"].corr()}


def corr_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                               zmin=-1, zmax=1, colorscale=[[0, RUST], [0.5, BG], [1, TEAL]],
                               text=corr.round(2).values, texttemplate="%{text}"))
    return _style(fig, title="Strategy correlation matrix", height=380)


def hedge_calculator(asset_df: pd.DataFrame, market_df: pd.DataFrame,
                     notional: float, window: int = 126) -> dict:
    """Beta hedge: how much of the market instrument to SHORT to neutralise a
    long position ('hedging amounts'). Rolling-beta based."""
    ra = asset_df["close"].pct_change()
    rm = market_df["close"].pct_change()
    ra, rm = ra.align(rm, join="inner")
    beta = ra.rolling(window, min_periods=window // 2).cov(rm) / rm.rolling(window, min_periods=window // 2).var()
    b = float(beta.dropna().iloc[-1]) if beta.notna().any() else 1.0
    return {"beta": round(b, 3),
            "hedge_notional": round(-b * notional, 2),
            "explain": (f"Position has β={b:.2f} to the hedge instrument. To be market-neutral, "
                        f"short ${abs(b * notional):,.0f} of it against your ${notional:,.0f} position. "
                        f"Residual P&L is then the idiosyncratic (alpha) component.")}


def multi_asset_scan(key: str, symbols: list, start: str, end: str,
                     cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> pd.DataFrame:
    """Run one strategy across a basket of tickers — the cross-asset backtest."""
    feed = get_feed()
    rows = []
    cls = single_strategies().get(key)
    for sym in symbols:
        try:
            d = feed.history(sym, start, end)
            strat = cls()
            if key == "vol_arb":
                bt = run_backtest(strat.vol_equity_df(d), strat.generate(d),
                                  cost_bps=cost_bps, slippage_bps=slippage_bps)
            else:
                bt = run_backtest(d, strat.generate(d), cost_bps=cost_bps, slippage_bps=slippage_bps)
            s = bt.stats
            rows.append({"symbol": sym, "Sharpe": round(s["Sharpe"], 2),
                         "CAGR": f"{s['CAGR']:.1%}", "MaxDD": f"{s['MaxDrawdown']:.1%}",
                         "WinRate": f"{s.get('WinRate', float('nan')):.0%}" if 'WinRate' in s else "—",
                         "PF": round(s.get("ProfitFactor", float("nan")), 2),
                         "Trades": s.get("NumTrades", 0)})
        except Exception as e:
            rows.append({"symbol": sym, "Sharpe": None, "CAGR": f"error: {str(e)[:30]}",
                         "MaxDD": "", "WinRate": "", "PF": None, "Trades": 0})
    out = pd.DataFrame(rows)
    return out.sort_values("Sharpe", ascending=False, na_position="last")


def _md_table(df: pd.DataFrame) -> str:
    """Markdown table without the tabulate dependency."""
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def fund_report(symbol: str, df: pd.DataFrame, capital: float = INITIAL_CAPITAL) -> str:
    """One-click daily fund report (markdown) — regime, book, signals, risk."""
    rec = recommended_strategy(df)
    book = build_book(df)
    beq = (1 + book["book_returns"]).cumprod() * capital
    from backtest.metrics import sharpe as _sh, max_drawdown as _mdd, cagr as _cagr
    rp = risk_panel(book, capital)
    attr = pnl_attribution(book, capital)
    sig = live_signals(df)
    try:
        rk = rank_strategies_on_df(df, symbol=symbol)
        if rk["any_edge"]:
            opt_block = (f"\n## Optimal strategy for {symbol}\n"
                         f"**{rk['best']}**  ·  params {rk['best_params']}\n\n"
                         f"_Why:_ {rk['reason']}\n\n" + _md_table(rk["table"]))
        else:
            opt_block = (f"\n## Optimal strategy for {symbol}\n"
                         f"**No robust edge found.** {rk['reason']}\n\n" + _md_table(rk["table"]))
    except Exception:
        opt_block = ""
    lines = [
        f"# HF-Hub Daily Fund Report — {df.index[-1].date()}",
        f"\n**Reference asset:** {symbol}  ·  **Capital:** ${capital:,.0f}",
        f"\n## Regime\n{rec['regime']} — favour {', '.join(rec['favour'])}; "
        f"avoid {', '.join(rec['avoid']) or 'none'}; gross scaled to {rec['gross']:.0%}. "
        f"Top regime-fit strategy: **{rec['pick']}**.",
        opt_block,
        f"\n## Book\nNAV ${beq.iloc[-1]:,.0f} · CAGR {_cagr(beq):.1%} · "
        f"Sharpe {_sh(book['book_returns']):.2f} · MaxDD {_mdd(beq):.1%}",
        f"\n## Risk (1-day, historical)\nVaR95 ${rp['VaR95_1d']:,.0f} · "
        f"CVaR95 ${rp['CVaR95_1d']:,.0f} · VaR99 ${rp['VaR99_1d']:,.0f} · "
        f"ann vol {rp['ann_vol']:.1%}",
        "\n## P&L attribution\n" + _md_table(attr),
        "\n## Live signals\n" + _md_table(sig[["strategy", "category", "stance", "horizon",
                                               "backtest_Sharpe"]]),
        "\n---\n*Paper research platform. Backtests measure edge net of costs; they do not create it.*",
    ]
    return "\n".join(lines)


# ============================================================================
# Strategies explained — expert plain-English reference for each strategy
# ============================================================================
STRATEGY_EXPLAINERS = {
 "vix_rsi": dict(edge="Short-term overreaction in index prices, amplified by the volatility risk premium.",
   works="Choppy, fearful markets — sharp oversold dips inside an uptrend.",
   fails="Strong sustained downtrends (falling knives) — the trend filter guards this.",
   retail="Yes. One of the most accessible edges; also the most crowded."),
 "breakout": dict(edge="Momentum / underreaction to new information — trends persist.",
   works="Strong directional trends across futures, FX, crypto, equities.",
   fails="Range-bound chop — repeated false breakouts bleed via whipsaw.",
   retail="Excellent. The classic CTA entry; high capacity."),
 "trendline": dict(edge="Persistence of established trend, measured by regression slope.",
   works="Smooth trending regimes with a clear slope.",
   fails="Sideways markets; the MA filter reduces but doesn't remove whipsaw.",
   retail="Yes."),
 "commodity_trend": dict(edge="Supply/demand momentum plus a genuine calendar seasonal premium.",
   works="Trending commodities with seasonal demand (heating oil winter, grains harvest).",
   fails="Non-seasonal, mean-reverting commodity regimes.",
   retail="Yes, via futures or commodity ETFs."),
 "ichimoku": dict(edge="Trend confluence — only trade when price, cloud and the Tenkan/Kijun cross agree.",
   works="Established trends; the multi-condition filter cuts marginal signals.",
   fails="Choppy markets where price oscillates through the cloud.",
   retail="Yes; popular in FX and crypto day-trading."),
 "cci": dict(edge="Statistical deviation from a moving mean; ±100 marks stretched conditions.",
   works="Reversion mode in ranges; momentum mode in trends (configurable).",
   fails="Using reversion in a trend (or vice-versa) — regime match matters.",
   retail="Yes."),
 "macd": dict(edge="Momentum: the MACD line crossing its signal flags acceleration/deceleration.",
   works="Trending markets; the signal-line filter reduces noise vs raw crossovers.",
   fails="Choppy ranges — crossovers whipsaw. Pair with a trend filter.",
   retail="Yes — one of the most common algo entries."),
 "bollinger_rev": dict(edge="Mean reversion: price stretched 2σ from its mean tends to snap back.",
   works="Range-bound / high-liquidity names with stable volatility.",
   fails="Strong trends — price 'walks the band' and reversion bleeds. Needs a trend filter.",
   retail="Yes."),
 "overnight_drift": dict(edge="The equity risk premium historically accrues close→open, not intraday.",
   works="Broad, liquid equity indices/ETFs; long-only, trend-filtered.",
   fails="Financing/borrow can eat the edge; single-name gap risk is large. Use ETFs, mind costs.",
   retail="Yes, but net-of-cost edge is thin — a good lesson in why overnight cost matters."),
 "multifactor": dict(edge="Style-factor risk premia (momentum, value, low-vol) harvested cross-sectionally.",
   works="Broad equity universes over months; dollar-neutral survives market crashes.",
   fails="Sharp factor reversals (e.g. Nov-2020 momentum crash).",
   retail="Excellent with enough names."),
 "equity_ls": dict(edge="Idiosyncratic (beta-stripped) momentum — underreaction to firm-specific news.",
   works="Diversified equity baskets; smoother than naive momentum.",
   fails="Broad de-risking events where idiosyncratic signal is swamped by beta.",
   retail="Yes with a universe of names."),
 "pairs_statarb": dict(edge="Relative-value mean reversion between cointegrated names.",
   works="Stable structural relationships (same sector, same drivers).",
   fails="When the relationship breaks (M&A, regime shift) — the ADF gate guards entry.",
   retail="Yes; Python is enough. Crowded."),
 "kalman_pairs": dict(edge="Same convergence edge, with a hedge ratio that ADAPTS as the relationship drifts.",
   works="Real pairs whose beta moves over time.",
   fails="Constant-beta pairs, where its flexibility just chases noise (static wins there).",
   retail="Yes."),
 "fixed_income_arb": dict(edge="Duration/curve convergence — rate instruments revert to a stable spread.",
   works="Liquid rate ETFs / curve segments with tight, mean-reverting spreads.",
   fails="Regime breaks in the curve (policy shifts). Institutionally leverage-dependent.",
   retail="Signal yes; the returns need scale/leverage retail rarely has."),
 "index_arb": dict(edge="ETF-vs-future basis dislocations converge to fair value.",
   works="Tight index/future relationships intraday.",
   fails="Real-world execution needs low latency; daily-bar version is research-only.",
   retail="Difficult live (latency); fine as research."),
 "global_macro": dict(edge="Cross-asset risk premia are conditional on the economic cycle.",
   works="Clear macro regimes (growth accelerating, inflation trending).",
   fails="Whipsawing/contradictory data; slow-moving by design.",
   retail="Yes, rule-based on public macro data."),
 "fx_carry": dict(edge="Forward-rate bias / uncovered-interest-parity failure — a crash-risk premium.",
   works="Calm risk-on regimes; you earn the rate differential.",
   fails="Risk-off crashes ('picking up pennies') — momentum overlay + vol-scaling mitigate.",
   retail="Yes."),
 "vol_arb": dict(edge="Variance risk premium — implied vol exceeds realised most of the time.",
   works="Calm markets; you collect the insurance premium.",
   fails="Vol spikes (Volmageddon) — negative skew. YOUR BUILD DOC BANS this as a primary sleeve; hard-capped here.",
   retail="Yes but dangerous; size tiny."),
 "crypto_funding": dict(edge="Structural perp funding/basis premium from leverage demand.",
   works="Bull markets with persistently positive funding; delta-neutral by construction.",
   fails="Funding compresses as capital arrives; capacity-limited.",
   retail="Very accessible via exchange APIs."),
 "event_earnings": dict(edge="Post-earnings-announcement drift — markets underreact to surprises.",
   works="Diversified across many names; each name a few dates/year.",
   fails="Decaying as it gets arbitraged; low per-name capacity.",
   retail="Yes with an earnings calendar + estimates."),
 "merger_arb": dict(edge="A risk premium for underwriting deal-completion risk.",
   works="Diversified book of announced cash deals in normal markets.",
   fails="Risk-off crises where deals break together — negative skew.",
   retail="Possible but capital-intensive."),
 "index_rebal": dict(edge="Predictable forced flows from price-insensitive index funds.",
   works="Around scheduled index add/delete effective dates.",
   fails="Crowded in the big indices; more edge in smaller ones.",
   retail="Yes; some quants specialise only in this."),
 "mm_scalper": dict(edge="Liquidity provision earns the bid/ask spread.",
   works="High-flow, controlled-inventory conditions (SIMULATION here).",
   fails="Adverse selection / inventory blow-outs. Real MM needs colocated infrastructure.",
   retail="No — infrastructure-bound. Included as an educational simulation."),
}

def strategy_explainer() -> list[dict]:
    """Rich reference card per strategy for the 'Explained' tab."""
    out = []
    for key, group, desc in ALL_STRATEGIES:
        cls = _get_cls(key); meta = cls.meta
        ex = STRATEGY_EXPLAINERS.get(key, {})
        out.append(dict(key=key, name=meta.name, category=meta.category, group=group,
                        horizon=meta.horizon, timeframe=meta.timeframe,
                        assets=", ".join(meta.asset_classes), shorts=meta.uses_short,
                        what=desc, **ex))
    return out


# ============================================================================
# Candlesticks + signal scanner (screener)
# ============================================================================
from plotly.subplots import make_subplots

def candlestick_fig(df: pd.DataFrame, position: pd.Series | None = None,
                    title="Price", max_bars: int = 400) -> go.Figure:
    """OHLC candlesticks + volume subplot, with entry/exit markers if a position
    series is supplied. Trimmed to the last `max_bars` for readability."""
    d = df.tail(max_bars)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d["open"], high=d["high"], low=d["low"],
                                 close=d["close"], name="",
                                 increasing_line_color=GREEN, decreasing_line_color=RUST,
                                 increasing_fillcolor=GREEN, decreasing_fillcolor=RUST), row=1, col=1)
    if position is not None:
        p = position.reindex(d.index).fillna(0)
        flips = p.diff().fillna(p)
        longs = p[(flips != 0) & (p > 0)]; shorts = p[(flips != 0) & (p < 0)]
        exits = p[(flips != 0) & (p == 0)]
        fig.add_trace(go.Scatter(x=longs.index, y=d["low"].reindex(longs.index) * 0.995,
                                 mode="markers", marker=dict(color=GREEN, size=11, symbol="triangle-up"),
                                 name="Long"), row=1, col=1)
        fig.add_trace(go.Scatter(x=shorts.index, y=d["high"].reindex(shorts.index) * 1.005,
                                 mode="markers", marker=dict(color=RUST, size=11, symbol="triangle-down"),
                                 name="Short"), row=1, col=1)
        fig.add_trace(go.Scatter(x=exits.index, y=d["close"].reindex(exits.index),
                                 mode="markers", marker=dict(color=MUTED, size=7, symbol="x"),
                                 name="Exit"), row=1, col=1)
    vol_colors = [GREEN if c >= o else RUST for o, c in zip(d["open"], d["close"])]
    fig.add_trace(go.Bar(x=d.index, y=d["volume"], marker_color=vol_colors, name="Vol",
                         opacity=0.5), row=2, col=1)
    fig.update_layout(**{**_LAYOUT, "title": title, "height": 520, "showlegend": True,
                         "xaxis_rangeslider_visible": False})
    fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
    return fig


def signal_scanner(key: str, symbols: list, start: str, end: str, interval="1d",
                   cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> pd.DataFrame:
    """Screener: run one strategy across a basket and report each asset's CURRENT
    stance + a mechanical confidence score. This is 'potential signals across the
    whole field' — surveillance before selective strikes."""
    feed = get_feed()
    cls = single_strategies().get(key)
    rows = []
    for sym in symbols:
        try:
            d = feed.history(sym, start, end, interval=interval)
            strat = cls()
            pos = strat.generate(d)
            if key == "vol_arb":
                bt = run_backtest(strat.vol_equity_df(d), pos, cost_bps=cost_bps, slippage_bps=slippage_bps)
            else:
                bt = run_backtest(d, pos, cost_bps=cost_bps, slippage_bps=slippage_bps)
            s = bt.stats
            stance = "LONG" if pos.iloc[-1] > 0 else "SHORT" if pos.iloc[-1] < 0 else "FLAT"
            # mechanical confidence: blends historical Sharpe, profit factor, exposure
            sharpe = s.get("Sharpe", 0) or 0
            pf = s.get("ProfitFactor", 1) or 1
            conf = max(0, min(100, 50 + 25 * sharpe + 10 * (min(pf, 3) - 1)))
            rows.append({"symbol": sym, "signal": stance, "confidence": round(conf),
                         "Sharpe": round(sharpe, 2), "MaxDD": f"{s['MaxDrawdown']:.0%}",
                         "WinRate": f"{s.get('WinRate', float('nan')):.0%}" if 'WinRate' in s else "—",
                         "last_px": round(float(d['close'].iloc[-1]), 2)})
        except Exception as e:
            rows.append({"symbol": sym, "signal": "error", "confidence": 0,
                         "Sharpe": None, "MaxDD": str(e)[:24], "WinRate": "", "last_px": None})
    out = pd.DataFrame(rows)
    # rank: actionable signals first, then by confidence
    out["_act"] = out["signal"].isin(["LONG", "SHORT"]).astype(int)
    return out.sort_values(["_act", "confidence"], ascending=[False, False]).drop(columns="_act")


# ============================================================================
# Strategy <-> Asset matcher: optimal strategy per stock (and vice-versa) + WHY
# ============================================================================
def _fit_reason(key: str, stats: dict, reg) -> str:
    """Generate the 'why' — regime fit + edge rationale + honest stat read."""
    cls = _get_cls(key); cat = cls.meta.category
    ex = STRATEGY_EXPLAINERS.get(key, {})
    r = route(reg)
    if cat in r["favour"]:
        regime_bit = f"the current **{reg.label()}** regime *favours* {cat} strategies"
    elif cat in r["avoid"]:
        regime_bit = f"the current **{reg.label()}** regime is *unfavourable* for {cat} (headwind)"
    else:
        regime_bit = f"the current regime is neutral for {cat}"
    sh = stats.get("Sharpe", 0); pf = stats.get("ProfitFactor", 0) or 0
    wr = stats.get("WinRate", None)
    if wr is not None and wr < 0.40 and pf > 1:
        stat_bit = (f"Win rate is only {wr:.0%}, but that's *expected* here — the winners are far "
                    f"bigger than the losers (profit factor {pf:.2f}). Judge it on Sharpe {sh:.2f}, not win rate")
    elif pf and pf > 1:
        stat_bit = f"Sharpe {sh:.2f}, profit factor {pf:.2f}" + (f", win rate {wr:.0%}" if wr is not None else "")
    else:
        stat_bit = f"Sharpe {sh:.2f} — the edge is weak/absent on this asset in this window"
    edge = ex.get("edge", cls.meta.description)
    return f"{edge} Here, {regime_bit}. {stat_bit}."


def _verdict(fs: dict) -> tuple[str, float, bool]:
    """Judge a strategy's full-sample quality. Returns (verdict, composite_score,
    passes_gate). The gate blocks money-losers/blowups from ever being 'recommended'
    no matter what a noisy out-of-sample fold says."""
    sharpe = fs.get("Sharpe", 0) or 0
    pf = fs.get("ProfitFactor", 0) or 0
    trades = fs.get("NumTrades", 0) or 0
    total = fs.get("TotalReturn", 0) or 0
    mdd = fs.get("MaxDrawdown", 0) or 0        # negative
    # hard sanity gate: must be profitable, have a real edge, enough trades, not blown up
    passes = (total > 0) and (pf >= 1.1) and (trades >= 8) and (mdd > -0.55) and (sharpe > 0.2)
    # composite: reward Sharpe + profit factor, penalise drawdown; clipped & bounded
    dd_pen = max(0.0, (-mdd - 0.25)) * 1.5     # penalty kicks in past 25% DD
    score = 0.6 * min(sharpe, 3) + 0.25 * min(max(pf - 1, 0), 3) - dd_pen
    if not passes:
        score -= 5.0                            # sink failing strategies below all passers
    if total <= 0 or pf < 1:
        v = "❌ loses money"
    elif not passes:
        v = "⚠️ thin / fragile"
    elif trades < 15 or sharpe < 0.8:
        v = "⚠️ ok but low-confidence"
    else:
        v = "✅ solid"
    return v, score, passes


def rank_strategies_on_df(df: pd.DataFrame, metric="Sharpe", cost_bps=COST_BPS,
                          slippage_bps=SLIPPAGE_BPS, symbol: str | None = None) -> dict:
    """Walk-forward rank every single-series strategy on a given df, then rank by a
    ROBUST composite (full-sample Sharpe + profit factor − drawdown penalty), with a
    sanity gate so a money-losing strategy is never called 'optimal'. Out-of-sample
    score is shown as a consistency check, not the sole ranking."""
    reg = latest_regime(df)
    n_bars = len(df)
    rows = []
    # only rank strategies that are legitimate for this asset class — ranking a
    # commodity-seasonality model on a single stock is meaningless (and slow)
    _pool = relevant_strategies(symbol) if symbol else REGISTRY
    for key, cls in _pool.items():
        try:
            # 3 folds keeps the walk-forward honest while keeping the UI responsive
            wf = walk_forward(cls, df, n_splits=3, metric=metric,
                              cost_bps=cost_bps, slippage_bps=slippage_bps)
            fs = wf["full_sample_stats"]
            verdict, score, passes = _verdict(fs)
            rows.append({"strategy": key, "verdict": verdict, "score": round(score, 2),
                         "Sharpe": round(fs["Sharpe"], 2), "CAGR": f"{fs['CAGR']:.1%}",
                         "MaxDD": f"{fs['MaxDrawdown']:.0%}",
                         "WinRate": f"{fs.get('WinRate', float('nan')):.0%}" if 'WinRate' in fs else "—",
                         "PF": round(fs.get("ProfitFactor", float("nan")), 2),
                         "Trades": fs.get("NumTrades", 0),
                         "oos_" + metric: round(wf["oos_score"], 2),
                         "params": wf["recommended_params"], "_fs": fs, "_pass": passes})
        except Exception as e:
            rows.append({"strategy": key, "verdict": "error", "score": -99, "Sharpe": None,
                         "CAGR": f"error: {str(e)[:18]}", "MaxDD": "", "WinRate": "", "PF": None,
                         "Trades": 0, "oos_" + metric: None, "params": {}, "_fs": {}, "_pass": False})
    tbl = pd.DataFrame(rows).sort_values("score", ascending=False, na_position="last")
    any_pass = tbl["_pass"].any()
    best = tbl.iloc[0]
    conf_note = ""
    if n_bars < 400:
        conf_note = (f" ⚠️ Only {n_bars} bars in this window — walk-forward is noisy here; "
                     f"treat this as low-confidence. Widen the lookback (3–5 years) for a trustworthy read.")
    if any_pass:
        reason = _fit_reason(best["strategy"], best["_fs"], reg) + conf_note
        headline = best["strategy"]
    else:
        reason = ("**No strategy shows a robust edge on this asset in this window.** "
                  "Every option either lost money, barely traded, or drew down too hard. "
                  "The top row is just the least-bad — do NOT trade it. Try a longer lookback, a "
                  "different asset, or accept that this name isn't tradeable by these strategies right now."
                  + conf_note)
        headline = None
    return {"regime": reg.label(), "best": headline, "best_params": best["params"] if any_pass else {},
            "reason": reason, "table": tbl.drop(columns=["_fs", "_pass"]), "metric": metric,
            "any_edge": bool(any_pass)}


def optimize_for_asset(symbol: str, start: str, end: str, interval="1d", metric="Sharpe",
                       cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> dict:
    """For ONE asset, walk-forward optimise every single-series strategy, rank by
    out-of-sample metric, and explain the winner."""
    df = load_bars(symbol, start, end, interval)
    out = rank_strategies_on_df(df, metric, cost_bps, slippage_bps, symbol=symbol)
    out["symbol"] = symbol
    return out


def best_assets_for_strategy(key: str, symbols: list, start: str, end: str, interval="1d",
                             metric="Sharpe", cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> dict:
    """For ONE strategy, find which assets it fits best (walk-forward), and why.
    Answers 'where does this strategy actually work?'."""
    cls = REGISTRY.get(key) or single_strategies().get(key)
    rows = []
    for sym in symbols:
        try:
            df = load_bars(sym, start, end, interval)
            reg = latest_regime(df)
            wf = walk_forward(cls, df, n_splits=4, metric=metric,
                              cost_bps=cost_bps, slippage_bps=slippage_bps)
            fs = wf["full_sample_stats"]
            rows.append({"symbol": sym, "regime": reg.trend, "oos_" + metric: round(wf["oos_score"], 2),
                         "Sharpe": round(fs["Sharpe"], 2), "MaxDD": f"{fs['MaxDrawdown']:.0%}",
                         "PF": round(fs.get("ProfitFactor", float("nan")), 2),
                         "params": wf["recommended_params"], "_fs": fs, "_reg": reg})
        except Exception as e:
            rows.append({"symbol": sym, "regime": "?", "oos_" + metric: None, "Sharpe": None,
                         "MaxDD": str(e)[:18], "PF": None, "params": {}, "_fs": {}, "_reg": None})
    tbl = pd.DataFrame(rows)
    # rank by the same robust composite, not raw OOS
    def _sc(row):
        fs = row["_fs"]
        return _verdict(fs)[1] if isinstance(fs, dict) and fs else -99
    tbl["score"] = tbl.apply(_sc, axis=1)
    tbl["verdict"] = tbl.apply(lambda r: _verdict(r["_fs"])[0] if isinstance(r["_fs"], dict) and r["_fs"] else "error", axis=1)
    tbl = tbl.sort_values("score", ascending=False, na_position="last")
    any_pass = any(_verdict(r["_fs"])[2] for r in rows if isinstance(r["_fs"], dict) and r["_fs"])
    best = tbl.iloc[0]
    if any_pass:
        reason = (_fit_reason(key, best["_fs"], best["_reg"]) if best["_fs"] is not None and len(best["_fs"])
                  else "no valid fit found.")
        best_sym = best["symbol"]
    else:
        reason = (f"**`{key}` shows no robust edge on any asset in this universe/window.** "
                  "The top row is least-bad, not good — don't trade it. Try a longer lookback or a "
                  "different universe.")
        best_sym = None
    return {"strategy": key, "best_symbol": best_sym, "best_params": best["params"] if any_pass else {},
            "reason": reason, "table": tbl.drop(columns=["_fs", "_reg"]), "metric": metric,
            "any_edge": bool(any_pass)}


# ============================================================================
# Asset-class relevance: stop running crypto/commodity strategies on an equity
# ============================================================================
def infer_asset_class(symbol: str) -> str:
    """Infer the asset class from the ticker so we only run strategies that make
    sense for it. Ticker conventions are reliable enough for routing."""
    s = (symbol or "").upper()
    if s.endswith("-USD") or s.endswith("USDT") or "/" in s:
        return "crypto"
    if "=X" in s:
        return "fx"
    if "=F" in s:
        return "commodities"
    if s.startswith("^"):
        return "index"
    if s in {"SPY", "QQQ", "IWM", "DIA", "GLD", "SLV", "USO", "TLT", "IEF", "XLE", "XLF", "EEM", "VTI"}:
        return "etf"
    return "equities"


# which strategies are legitimate for which asset class
_CLASS_OK = {
    "equities": {"vix_rsi", "breakout", "trendline", "ichimoku", "cci", "macd",
                 "bollinger_rev", "overnight_drift"},
    "etf":      {"vix_rsi", "breakout", "trendline", "ichimoku", "cci", "macd",
                 "bollinger_rev", "overnight_drift", "commodity_trend"},
    "index":    {"vix_rsi", "breakout", "trendline", "ichimoku", "cci", "macd", "bollinger_rev"},
    "crypto":   {"breakout", "trendline", "ichimoku", "cci", "macd", "bollinger_rev"},
    "fx":       {"breakout", "trendline", "ichimoku", "cci", "macd", "bollinger_rev"},
    "commodities": {"breakout", "trendline", "commodity_trend", "ichimoku", "cci", "macd"},
}


def relevant_strategies(symbol: str) -> dict:
    """The single-series strategies that actually make sense for this ticker."""
    cls_name = infer_asset_class(symbol)
    ok = _CLASS_OK.get(cls_name, set(REGISTRY))
    return {k: v for k, v in REGISTRY.items() if k in ok}


def relevance_note(symbol: str) -> str:
    ac = infer_asset_class(symbol)
    keys = list(relevant_strategies(symbol))
    return (f"`{symbol}` detected as **{ac}** → running the {len(keys)} strategies valid for "
            f"{ac}: {', '.join(keys)}. (Strategies for other asset classes are excluded — "
            f"running commodity seasonality on an equity would be meaningless.)")


# ============================================================================
# Actionable live signals: entry / stop / target / status / why — per strategy
# ============================================================================
def actionable_signals(symbol: str, df: pd.DataFrame, capital: float = INITIAL_CAPITAL,
                       cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS,
                       spread_bps=0.0, financing_bps_annual=0.0) -> pd.DataFrame:
    """One row per RELEVANT strategy: what it says to do right now, at what price,
    with stop/target/size, whether it's already in a position or waiting for
    confirmation, and WHY. This is the trade ticket, not a sentiment read."""
    from core.indicators import atr as _atr
    from risk.sizing import vol_target_size
    from core.indicators import realized_vol as _rv

    px = float(df["close"].iloc[-1])
    atr_v = float(_atr(df, 14).iloc[-1])
    rv = float(_rv(df["close"], 20).iloc[-1])
    reg = latest_regime(df)
    gross = route(reg)["gross_exposure"]
    rows = []
    for key, cls in relevant_strategies(symbol).items():
        try:
            strat = cls()
            pos = strat.generate(df)
            now = float(pos.iloc[-1])
            prev = float(pos.iloc[-2]) if len(pos) > 1 else 0.0
            # backtest for quality context
            bt = run_backtest(df, pos, initial_capital=capital, cost_bps=cost_bps,
                              slippage_bps=slippage_bps, spread_bps=spread_bps,
                              financing_bps_annual=financing_bps_annual)
            s = bt.stats
            verdict, score, passes = _verdict(s)

            if now == 0 and prev == 0:
                status, side = "⚪ no setup", "FLAT"
            elif now != 0 and prev == 0:
                status, side = "🔔 NEW ENTRY", ("LONG" if now > 0 else "SHORT")
            elif now == 0 and prev != 0:
                status, side = "🚪 EXIT now", "CLOSE"
            elif now != prev:
                status, side = "🔄 FLIP", ("LONG" if now > 0 else "SHORT")
            else:
                status, side = "✅ in position", ("LONG" if now > 0 else "SHORT")

            d = 1 if now > 0 else -1 if now < 0 else 0
            if d != 0:
                ex = derive_exits(key, df, d, px, cost_bps, slippage_bps,
                                  spread_bps, financing_bps_annual)
                stop, targ, rr = ex["stop"], ex["target"], ex["rr"]
                expectancy, exit_basis = ex["expectancy_pct"], ex["basis"]
                lev = vol_target_size(rv, 0.15, 2.0)
                notional = capital * 0.05 * gross * lev          # respects 5% cap
                shares = round(notional / px, 2) if px > 0 else 0
            else:
                stop = targ = rr = shares = expectancy = None
                exit_basis = "no position"
                notional = 0.0

            rows.append({
                "strategy": key, "status": status, "side": side,
                "entry": round(px, 2), "stop": stop, "target": targ, "R:R": rr,
                "expect_%/trade": expectancy,
                "size_$": round(notional), "shares": shares,
                "quality": verdict, "hist_Sharpe": round(s["Sharpe"], 2),
                "levels_from": exit_basis,
                "why": strat._explain(df) if hasattr(strat, "_explain") else cls.meta.description,
            })
        except Exception as e:
            rows.append({"strategy": key, "status": "error", "side": "-", "entry": None,
                         "stop": None, "target": None, "R:R": None, "expect_%/trade": None,
                         "size_$": 0, "shares": None, "quality": "error", "hist_Sharpe": None,
                         "levels_from": "-", "why": str(e)[:60]})
    out = pd.DataFrame(rows)
    order = {"🔔 NEW ENTRY": 0, "🔄 FLIP": 1, "🚪 EXIT now": 2, "✅ in position": 3, "⚪ no setup": 4}
    out["_o"] = out["status"].map(lambda v: order.get(v, 9))
    out["_e"] = out["expect_%/trade"].fillna(-99)
    return out.sort_values(["_o", "_e"], ascending=[True, False]).drop(columns=["_o", "_e"])


def auto_route_mandates(symbols: list, start: str, end: str, interval="1d",
                        cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS) -> pd.DataFrame:
    """Auto-route: for each ticker in YOUR universe, look at its history (trend +
    regime), test only the strategies valid for its asset class, and assign the one
    with a real edge. Tickers where nothing passes the sanity gate are dropped —
    the book only trades where an edge actually exists."""
    rows = []
    for sym in symbols:
        try:
            df = load_bars(sym, start, end, interval)
            reg = latest_regime(df)
            best_key, best_score, best_stats, best_params = None, -99, None, {}
            for key, cls in relevant_strategies(sym).items():
                try:
                    wf = walk_forward(cls, df, n_splits=4, metric="Sharpe",
                                      cost_bps=cost_bps, slippage_bps=slippage_bps)
                    fs = wf["full_sample_stats"]
                    verdict, score, passes = _verdict(fs)
                    if score > best_score:
                        best_key, best_score, best_stats = key, score, fs
                        best_params, best_verdict = wf["recommended_params"], verdict
                except Exception:
                    continue
            if best_stats is None:
                rows.append({"symbol": sym, "strategy": None, "verdict": "no data",
                             "score": None, "Sharpe": None, "regime": "?", "params": {},
                             "trade": False, "why": "could not evaluate"})
                continue
            v, sc, passes = _verdict(best_stats)
            rows.append({"symbol": sym, "strategy": best_key, "verdict": v, "score": round(sc, 2),
                         "Sharpe": round(best_stats["Sharpe"], 2), "regime": reg.trend,
                         "params": best_params, "trade": bool(passes),
                         "why": _fit_reason(best_key, best_stats, reg)})
        except Exception as e:
            rows.append({"symbol": sym, "strategy": None, "verdict": "error", "score": None,
                         "Sharpe": None, "regime": "?", "params": {}, "trade": False,
                         "why": str(e)[:60]})
    out = pd.DataFrame(rows)
    return out.sort_values("score", ascending=False, na_position="last")


# ============================================================================
# Growth-optimal (Kelly) sizing lab — the honest answer to "maximise profit"
# ============================================================================
def kelly_lab(returns: pd.Series, periods_per_year: int = 252) -> dict:
    """Growth-optimal sizing analysis on a strategy's return stream.

    For a continuously-compounded book, long-run growth at leverage f is
        g(f) = f·μ − f²σ²/2
    which is maximised at the FULL-KELLY leverage
        f* = μ / σ²
    with peak growth g(f*) = μ² / (2σ²).

    The crucial asymmetry (this is the whole point):
        g(c·f*) / g(f*) = 2c − c²
      c = 0.50 → 75% of max growth at HALF the leverage/volatility
      c = 1.00 → 100% (but only if your μ,σ estimates are exact — they never are)
      c = 2.00 → 0% growth. Double Kelly earns NOTHING long-run.
      c > 2.00 → negative growth: ruin, even with a positive edge.

    Overbetting is punished far harder than underbetting. That is why fractional
    Kelly (0.25–0.5) is the growth-optimal choice ONCE you admit your edge estimate
    is noisy — you buy a large cut in variance for a small cut in growth.
    """
    r = pd.Series(returns).replace([np.inf, -np.inf], np.nan).dropna()
    if r.std() == 0:
        return {"ok": False, "msg": "return stream has zero variance — nothing to size"}
    if len(r) < 120:
        # Sizing off a tiny sample is how people blow up: mu is pure noise here, so
        # f* = mu/sigma^2 can recommend absurd leverage with total confidence.
        return {"ok": False,
                "msg": (f"only {len(r)} observations — far too few to estimate an edge. "
                        f"Kelly needs a stable mu and sigma; on this sample f* would be noise "
                        f"amplified into leverage. Load at least ~6 months of bars (120+) "
                        f"before trusting any sizing number."),
                "mu": float(r.mean() * periods_per_year),
                "sigma": float(r.std() * np.sqrt(periods_per_year)), "f_star": 0.0}
    mu = float(r.mean() * periods_per_year)          # annualised drift
    sigma = float(r.std() * np.sqrt(periods_per_year))
    if mu <= 0:
        return {"ok": False, "msg": f"negative expected return ({mu:.1%}/yr) — Kelly says bet ZERO. "
                                    f"No sizing rule turns a negative edge into profit.",
                "mu": mu, "sigma": sigma, "f_star": 0.0}
    f_star = mu / (sigma ** 2)
    g_max = mu ** 2 / (2 * sigma ** 2)
    # standard error of the mean -> how uncertain is our edge estimate?
    se_mu = float(r.std() * np.sqrt(periods_per_year) / np.sqrt(len(r)))
    t_stat = mu / se_mu if se_mu > 0 else 0.0
    fracs = np.array([0.1, 0.25, 0.33, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5])
    rows = []
    for c in fracs:
        f = c * f_star
        g = f * mu - (f ** 2) * (sigma ** 2) / 2
        vol = f * sigma
        # approximate drawdown scale & ruin proxy for a GBM with drift g, vol `vol`
        ruin = float(np.exp(-2 * g / (vol ** 2))) if g > 0 and vol > 0 else 1.0
        rows.append({"Kelly fraction": f"{c:.2f}×", "leverage": round(f, 2),
                     "growth %/yr": round(g * 100, 1),
                     "% of max growth": round(100 * (2 * c - c ** 2), 1),
                     "vol %/yr": round(vol * 100, 1),
                     "P(halving before doubling)": round(min(ruin, 1.0), 3)})
    return {"ok": True, "mu": mu, "sigma": sigma, "f_star": f_star, "g_max": g_max,
            "t_stat": t_stat, "se_mu": se_mu, "table": pd.DataFrame(rows),
            "n_obs": int(len(r)),
            "recommended_fraction": 0.25 if t_stat < 2 else 0.5,
            "note": ("Edge estimate is statistically weak (t<2) — the honest Kelly fraction is 0.25× "
                     "or less; treat f* as an upper bound you should never approach."
                     if t_stat < 2 else
                     "Edge estimate is reasonably significant (t>2), but f* still assumes μ and σ are "
                     "stable going forward. 0.5× Kelly captures 75% of the growth at half the vol.")}


def kelly_curve_fig(lab: dict) -> go.Figure:
    """The growth-vs-leverage curve: where the peak is, and the cliff past it."""
    mu, sigma, f_star = lab["mu"], lab["sigma"], lab["f_star"]
    f = np.linspace(0, 2.6 * f_star, 200)
    g = f * mu - (f ** 2) * (sigma ** 2) / 2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=f, y=g * 100, line=dict(color=AMBER, width=2.5), name="growth"))
    fig.add_vline(x=f_star, line=dict(color=GREEN, dash="dash"),
                  annotation_text="full Kelly f*", annotation_position="top")
    fig.add_vline(x=0.5 * f_star, line=dict(color=TEAL, dash="dot"),
                  annotation_text="½ Kelly (75% growth, ½ vol)", annotation_position="top left")
    fig.add_vline(x=2 * f_star, line=dict(color=RUST, dash="dash"),
                  annotation_text="2× Kelly = ZERO growth", annotation_position="top right")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    return _style(fig, title="Long-run growth vs leverage — the overbetting cliff",
                  xaxis=dict(title="leverage (f)", gridcolor=GRID),
                  yaxis=dict(title="growth %/yr", gridcolor=GRID, ticksuffix="%"))


# ============================================================================
# Per-strategy exit levels derived from that strategy's OWN trade distribution
# ============================================================================
def derive_exits(key: str, df: pd.DataFrame, direction: int, price: float,
                 cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS,
                 spread_bps=0.0, financing_bps_annual=0.0) -> dict:
    """Stop/target/R:R from how this strategy ACTUALLY behaves on THIS asset —
    not a fixed ATR multiple.

    A trend strategy wins ~35% of the time with large winners; a reversion strategy
    wins ~65% with small ones. Forcing both to the same 2×ATR / 3×ATR (R:R 1.5) is
    arbitrary. Instead we read the strategy's realised trade distribution:
        stop   = its typical LOSS when wrong   (avg losing trade)
        target = its typical WIN when right    (avg winning trade)
        R:R    = target / stop                  (an OUTPUT, not an input)
    and report EXPECTANCY = win_rate·avgWin − (1−win_rate)·avgLoss, which is the
    number that actually decides whether the trade is worth taking. Falls back to
    ATR multiples only when there aren't enough historical trades to be meaningful.
    """
    from core.indicators import atr as _atr
    cls = REGISTRY.get(key)
    if cls is None or direction == 0:
        return {"stop": None, "target": None, "rr": None, "expectancy_pct": None,
                "basis": "no position", "win_rate": None}
    try:
        pos = cls().generate(df)
        blot = trade_blotter(df, pos)
        closed = blot[blot["status"] == "closed"] if len(blot) else blot
        wins = closed[closed["pnl_pct"] > 0]["pnl_pct"] if len(closed) else pd.Series(dtype=float)
        losses = closed[closed["pnl_pct"] < 0]["pnl_pct"] if len(closed) else pd.Series(dtype=float)
        if len(wins) >= 3 and len(losses) >= 3:
            avg_win = float(wins.mean()) / 100.0
            avg_loss = abs(float(losses.mean())) / 100.0
            wr = float((closed["pnl_pct"] > 0).mean())
            stop = price * (1 - direction * avg_loss) if direction > 0 else price * (1 + avg_loss)
            target = price * (1 + direction * avg_win) if direction > 0 else price * (1 - avg_win)
            rr = avg_win / avg_loss if avg_loss > 0 else None
            exp_pct = (wr * avg_win - (1 - wr) * avg_loss) * 100
            return {"stop": round(stop, 2), "target": round(target, 2),
                    "rr": round(rr, 2) if rr else None,
                    "expectancy_pct": round(exp_pct, 3), "win_rate": round(wr, 3),
                    "basis": f"{len(closed)} historical trades on this asset",
                    "avg_win_pct": round(avg_win * 100, 2), "avg_loss_pct": round(-avg_loss * 100, 2)}
    except Exception:
        pass
    # fallback: volatility-based, clearly labelled as a fallback
    a = float(_atr(df, 14).iloc[-1])
    stop = price - direction * 2.0 * a
    target = price + direction * 3.0 * a
    return {"stop": round(stop, 2), "target": round(target, 2), "rr": 1.5,
            "expectancy_pct": None, "win_rate": None,
            "basis": "⚠️ ATR fallback (too few historical trades to derive)"}

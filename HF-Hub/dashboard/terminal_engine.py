"""
dashboard/terminal_engine.py
----------------------------
The TERMINAL hub's compute layer: a live opportunity scanner and heatmap across a
whole universe, plus "forming setup" detection (opportunities that are N candles
away from triggering, not just ones already live).

Design principle carried from the main hub: rank by EXPECTANCY, never by raw
signal count or R:R. A scanner that lists every crossover is a noise generator.
This one scores each (asset, strategy) pair on its realised edge and only
surfaces what has historically paid.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.engine import (
    load_bars, relevant_strategies, infer_asset_class, derive_exits,
    _verdict, _fit_reason, trade_blotter, blotter_summary,
    _style, AMBER, GREEN, RUST, TEAL, MUTED, GRID, BG,
    COST_BPS, SLIPPAGE_BPS, INITIAL_CAPITAL,
)
from backtest.engine import run_backtest
from core.regime import latest_regime
from strategies.registry import REGISTRY


# ====================================================================== scanner
def _proximity_to_trigger(pos: pd.Series, lookahead_hint: int = 5) -> tuple[str, int]:
    """Classify how close a strategy is to firing.

    Returns (state, bars_since_change). States:
      LIVE      — position is on right now
      JUST FIRED— flipped within the last few bars (still actionable)
      FORMING   — flat now, but flipped recently enough that conditions are near
      DORMANT   — nothing happening
    """
    p = pos.fillna(0.0)
    if len(p) == 0:
        return ("DORMANT", 999)
    last = float(p.iloc[-1])
    changes = p.diff().fillna(0) != 0
    idx = np.where(changes.values)[0]
    bars_since = int(len(p) - 1 - idx[-1]) if len(idx) else 999
    if last != 0 and bars_since <= lookahead_hint:
        return ("JUST FIRED", bars_since)
    if last != 0:
        return ("LIVE", bars_since)
    if bars_since <= lookahead_hint * 2:
        return ("FORMING", bars_since)
    return ("DORMANT", bars_since)


def scan_universe(symbols: list, start: str, end: str, interval: str = "1d",
                  capital: float = INITIAL_CAPITAL, cost_bps=COST_BPS,
                  slippage_bps=SLIPPAGE_BPS, spread_bps: float = 0.0,
                  financing_bps_annual: float = 0.0,
                  strategies: list | None = None,
                  min_expectancy: float | None = None) -> pd.DataFrame:
    """Scan every (asset x relevant strategy) combination.

    One row per combination with: current state, entry/stop/target derived from
    that strategy's own trade distribution, expectancy, win rate, and a quality
    verdict. This is the 'go through the charts until it finds something' engine.
    """
    rows = []
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        try:
            df = load_bars(sym, start, end, interval)
            if len(df) < 60:
                rows.append(dict(symbol=sym, strategy="—", state="NO DATA",
                                 asset_class=infer_asset_class(sym), expectancy=None,
                                 win_rate=None, net_bps=None, verdict="insufficient history",
                                 why=f"only {len(df)} bars"))
                continue
            reg = latest_regime(df)
            px = float(df["close"].iloc[-1])
            pool = relevant_strategies(sym)
            if strategies:
                pool = {k: v for k, v in pool.items() if k in strategies}
            for key, cls in pool.items():
                try:
                    strat = cls()
                    pos = strat.generate(df)
                    bt = run_backtest(df, pos, initial_capital=capital, cost_bps=cost_bps,
                                      slippage_bps=slippage_bps, spread_bps=spread_bps,
                                      financing_bps_annual=financing_bps_annual)
                    s = bt.stats
                    verdict, score, passes = _verdict(s)
                    state, bars_since = _proximity_to_trigger(pos)
                    last = float(pos.iloc[-1])
                    d = 1 if last > 0 else -1 if last < 0 else 0
                    ex = derive_exits(key, df, d if d != 0 else 1, px, cost_bps,
                                      slippage_bps, spread_bps, financing_bps_annual)
                    rows.append(dict(
                        symbol=sym, asset_class=infer_asset_class(sym), strategy=key,
                        state=state, bars_since=bars_since,
                        side=("LONG" if d > 0 else "SHORT" if d < 0 else "FLAT"),
                        price=round(px, 4),
                        entry=round(px, 4) if d != 0 else None,
                        stop=ex["stop"] if d != 0 else None,
                        target=ex["target"] if d != 0 else None,
                        rr=ex["rr"], expectancy=ex["expectancy_pct"],
                        win_rate=ex["win_rate"], sharpe=round(s["Sharpe"], 2),
                        max_dd=round(s["MaxDrawdown"], 3), trades=s.get("NumTrades", 0),
                        score=round(score, 2), verdict=verdict, regime=reg.trend,
                        why=_fit_reason(key, s, reg),
                    ))
                except Exception as e:
                    rows.append(dict(symbol=sym, asset_class=infer_asset_class(sym),
                                     strategy=key, state="ERROR", verdict="error",
                                     expectancy=None, win_rate=None, why=str(e)[:60]))
        except Exception as e:
            rows.append(dict(symbol=sym, asset_class=infer_asset_class(sym), strategy="—",
                             state="NO DATA", verdict="error", expectancy=None,
                             win_rate=None, why=str(e)[:60]))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if min_expectancy is not None and "expectancy" in out:
        keep = out["expectancy"].fillna(-99) >= min_expectancy
        out = out[keep | out["state"].isin(["NO DATA", "ERROR"])]
    order = {"JUST FIRED": 0, "LIVE": 1, "FORMING": 2, "DORMANT": 3, "NO DATA": 4, "ERROR": 5}
    out["_o"] = out["state"].map(lambda v: order.get(v, 9))
    out["_e"] = out["expectancy"].fillna(-99) if "expectancy" in out else -99
    return out.sort_values(["_o", "_e"], ascending=[True, False]).drop(columns=["_o", "_e"])


def actionable_only(scan: pd.DataFrame) -> pd.DataFrame:
    """The trades a desk would actually look at: fired/live, positive expectancy,
    and a strategy that isn't flagged as a money-loser."""
    if scan.empty:
        return scan
    m = (scan["state"].isin(["JUST FIRED", "LIVE", "FORMING"])
         & (scan["expectancy"].fillna(-99) > 0)
         & (~scan["verdict"].astype(str).str.startswith("❌")))
    return scan[m]


# ====================================================================== heatmap
def opportunity_heatmap(scan: pd.DataFrame, value: str = "expectancy") -> go.Figure:
    """Assets x strategies, coloured by expected profitability. Green = pays,
    red = bleeds. The whole universe at a glance."""
    if scan.empty:
        return _style(go.Figure(), title="No data to map")
    piv = scan.pivot_table(index="symbol", columns="strategy", values=value, aggfunc="mean")
    if piv.empty:
        return _style(go.Figure(), title="No data to map")
    piv = piv.reindex(piv.mean(axis=1).sort_values(ascending=False).index)
    z = piv.values.astype(float)
    lim = float(np.nanmax(np.abs(z))) if np.isfinite(z).any() else 1.0
    lim = lim if lim > 0 else 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=list(piv.columns), y=list(piv.index),
        zmin=-lim, zmax=lim,
        colorscale=[[0.0, RUST], [0.5, "#151A21"], [1.0, GREEN]],
        text=np.round(z, 3), texttemplate="%{text}",
        hovertemplate="%{y} · %{x}<br>" + value + ": %{z:.3f}<extra></extra>",
        colorbar=dict(title=value)))
    h = max(320, 34 * len(piv.index) + 120)
    return _style(fig, title=f"Opportunity heatmap — {value} by asset × strategy", height=h)


def asset_ranking(scan: pd.DataFrame) -> pd.DataFrame:
    """Rank assets most-profitable → most-lossable, aggregated across strategies."""
    if scan.empty:
        return scan
    g = scan.groupby("symbol").agg(
        best_strategy=("expectancy", lambda s: s.idxmax() if s.notna().any() else None),
        best_expectancy=("expectancy", "max"),
        avg_expectancy=("expectancy", "mean"),
        live_signals=("state", lambda s: int((s.isin(["JUST FIRED", "LIVE"])).sum())),
        solid=("verdict", lambda s: int(s.astype(str).str.startswith("✅").sum())),
        losers=("verdict", lambda s: int(s.astype(str).str.startswith("❌").sum())),
        avg_sharpe=("sharpe", "mean"),
    ).reset_index()
    # map index -> strategy name
    if "best_strategy" in g:
        g["best_strategy"] = g["best_strategy"].map(
            lambda i: scan.loc[i, "strategy"] if pd.notna(i) and i in scan.index else None)
    g["grade"] = pd.cut(g["best_expectancy"].fillna(-9),
                        bins=[-np.inf, 0, 0.15, 0.4, np.inf],
                        labels=["❌ avoid", "⚠️ marginal", "🟡 tradeable", "🟢 strong"])
    return g.sort_values("best_expectancy", ascending=False, na_position="last")


def ranking_bar(rank: pd.DataFrame, top: int = 25) -> go.Figure:
    """Horizontal bar: most profitable at the top, most lossable at the bottom."""
    if rank.empty:
        return _style(go.Figure(), title="No data")
    d = rank.head(top).iloc[::-1]
    colors = [GREEN if v > 0 else RUST for v in d["best_expectancy"].fillna(0)]
    fig = go.Figure(go.Bar(
        x=d["best_expectancy"], y=d["symbol"], orientation="h",
        marker=dict(color=colors),
        text=[f"{v:+.2f}%  ({s})" for v, s in zip(d["best_expectancy"].fillna(0),
                                                  d["best_strategy"].fillna("—"))],
        textposition="outside"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    h = max(320, 26 * len(d) + 120)
    return _style(fig, title="Assets ranked by best available edge (expectancy %/trade)",
                  height=h, xaxis=dict(gridcolor=GRID, ticksuffix="%"))


# ================================================================== arb display
def arb_scan_table(engine, books: list, vol_annual: float = 0.6,
                   pre_positioned: bool = True) -> pd.DataFrame:
    """Wrap the crypto arb engine for the terminal."""
    opps = engine.scan_spatial(books, vol_annual, pre_positioned)
    if not opps:
        return pd.DataFrame()
    return pd.DataFrame([o.to_row() for o in opps])


def arb_waterfall(opp) -> go.Figure:
    """Show exactly where an apparent arbitrage edge goes: gross → net.
    This single chart is the most useful thing in a crypto arb tool."""
    steps = ["gross", "depth slip", "fees", "transfer", "latency", "NET"]
    vals = [opp.gross_bps, -abs(opp.slippage_bps), -abs(opp.fees_bps),
            -abs(opp.transfer_bps), -abs(opp.latency_decay_bps), opp.net_bps]
    measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=steps, y=vals,
        text=[f"{v:+.1f}" for v in vals], textposition="outside",
        connector=dict(line=dict(color=MUTED)),
        increasing=dict(marker=dict(color=GREEN)),
        decreasing=dict(marker=dict(color=RUST)),
        totals=dict(marker=dict(color=AMBER if opp.net_bps > 0 else RUST))))
    return _style(fig, title=f"Where the edge goes — {opp.buy_venue} → {opp.sell_venue}",
                  yaxis=dict(gridcolor=GRID, ticksuffix="bps"), height=380)

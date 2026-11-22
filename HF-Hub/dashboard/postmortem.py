"""
dashboard/postmortem.py
-----------------------
THE LOSS POST-MORTEM LOOP.

When a strategy loses money, this asks *why* — mechanically, not vibes — then
proposes a specific fix and PROVES it out-of-sample before recommending it.

The core diagnostic is MFE/MAE excursion analysis. For every trade we record:
    MFE = max favourable excursion  (how far it ran our way before closing)
    MAE = max adverse excursion     (how far it went against us before closing)

Those two numbers separate failure modes that look identical in a P&L column:

  * High MFE on losers  -> the trade WAS working and we gave it back.
                           Fix: take profit earlier / trail sooner.
  * High MAE on winners -> we sat through pain we didn't need to.
                           Fix: better entry timing, or a wider stop is fine.
  * Low MFE + high MAE  -> entries are simply bad. Signal is wrong, not the exits.
  * MAE ~= stop on most losers -> stopped out at the noise floor.
                           Fix: widen stop relative to ATR.
  * Many tiny losses    -> whipsaw. Fix: add a trend filter / confirmation delay.
  * Costs > gross P&L   -> the edge is real but turnover eats it. Fix: slow it down.
  * One trade dominates -> tail dependence, not an edge. Fix: don't trust it.

Every proposed fix is then run through walk-forward validation. If the fix does
not improve OUT-OF-SAMPLE expectancy, it is reported as REJECTED. A post-mortem
that always finds a fix is just overfitting with extra steps.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from backtest.optimizer import walk_forward
from core.regime import latest_regime
from core.indicators import atr as _atr
from strategies.registry import REGISTRY


# ==================================================================== excursions
def trade_excursions(df: pd.DataFrame, position: pd.Series) -> pd.DataFrame:
    """Per-trade MFE / MAE table — the raw material of every diagnosis."""
    pos = position.fillna(0.0)
    hi, lo, cl = df["high"], df["low"], df["close"]
    atr_s = _atr(df, 14)
    rows, cur, i0 = [], 0.0, None
    for i in range(len(pos)):
        p = float(pos.iloc[i])
        if p != cur:
            if cur != 0.0 and i0 is not None:
                seg = slice(i0, i + 1)
                entry = float(cl.iloc[i0])
                d = np.sign(cur)
                if d > 0:
                    mfe = (float(hi.iloc[seg].max()) / entry - 1) * 100
                    mae = (float(lo.iloc[seg].min()) / entry - 1) * 100
                else:
                    mfe = (1 - float(lo.iloc[seg].min()) / entry) * 100
                    mae = (1 - float(hi.iloc[seg].max()) / entry) * 100
                pnl = (float(cl.iloc[i]) / entry - 1) * 100 * d
                a = float(atr_s.iloc[i0]) if np.isfinite(atr_s.iloc[i0]) else np.nan
                rows.append(dict(
                    entry_i=i0, exit_i=i, side="LONG" if d > 0 else "SHORT",
                    bars=i - i0, entry_px=round(entry, 4),
                    pnl_pct=round(pnl, 3), mfe_pct=round(mfe, 3), mae_pct=round(mae, 3),
                    atr_pct=round(a / entry * 100, 3) if np.isfinite(a) and entry else np.nan,
                    give_back=round(mfe - pnl, 3),      # profit reached then surrendered
                    win=pnl > 0))
            i0 = i if p != 0.0 else None
            cur = p
    return pd.DataFrame(rows)


# ==================================================================== diagnosis
def diagnose(df: pd.DataFrame, key: str, params: dict | None = None,
             cost_bps: float = 1.0, slippage_bps: float = 2.0,
             spread_bps: float = 0.0, financing_bps_annual: float = 0.0) -> dict:
    """Run the strategy, then work out WHY it made or lost money."""
    cls = REGISTRY[key]
    strat = cls(**(params or {}))
    pos = strat.generate(df)
    bt = run_backtest(df, pos, cost_bps=cost_bps, slippage_bps=slippage_bps,
                      spread_bps=spread_bps, financing_bps_annual=financing_bps_annual)
    ex = trade_excursions(df, pos)
    stats = bt.stats
    reg = latest_regime(df)

    if ex.empty:
        return {"key": key, "ok": False, "stats": stats, "excursions": ex,
                "findings": [{"code": "NO_TRADES", "severity": "high",
                              "title": "Strategy never traded",
                              "detail": "No entries fired in this window — nothing to diagnose. "
                                        "Loosen entry thresholds or lengthen the lookback."}],
                "verdict": "no trades"}

    wins, losses = ex[ex.win], ex[~ex.win]
    n = len(ex)
    gross = float(ex["pnl_pct"].sum())
    # cost drag: compare the same signal at zero cost
    bt_free = run_backtest(df, pos, cost_bps=0, slippage_bps=0, spread_bps=0,
                           financing_bps_annual=0)
    cost_drag = float(bt_free.stats["TotalReturn"] - stats["TotalReturn"]) * 100

    findings = []

    # --- 1. give-back: trades that were working and we surrendered the gain
    if len(losses):
        gb = float(losses["give_back"].mean())
        mfe_losers = float(losses["mfe_pct"].mean())
        if mfe_losers > 1.0 and gb > abs(float(losses["pnl_pct"].mean())) * 0.6:
            findings.append(dict(
                code="GIVE_BACK", severity="high",
                title="Losers were winning first — we gave the profit back",
                detail=(f"Losing trades ran **+{mfe_losers:.2f}%** in our favour on average before "
                        f"closing at **{float(losses['pnl_pct'].mean()):.2f}%**. That is "
                        f"**{gb:.2f}%** of round-trip give-back per losing trade. The entries are "
                        f"finding real moves; the EXIT is the problem."),
                fix="Take partial profit into strength or trail a stop once the trade is in profit."))

    # --- 2. stopped at the noise floor
    if len(losses) and losses["atr_pct"].notna().any():
        mae_in_atr = float((losses["mae_pct"].abs() / losses["atr_pct"]).median())
        if mae_in_atr < 1.5:
            findings.append(dict(
                code="STOP_TOO_TIGHT", severity="high",
                title="Exits are inside the noise floor",
                detail=(f"Median adverse excursion on losers is only **{mae_in_atr:.2f}×ATR**. "
                        f"Normal bar-to-bar noise is ~1×ATR, so these trades were closed by "
                        f"randomness, not by the thesis breaking."),
                fix="Widen the effective stop to ≥2×ATR, or slow the signal so it stops flipping."))

    # --- 3. whipsaw: many small losses, short holds
    if n >= 10:
        small = losses[losses["pnl_pct"].abs() < 0.5] if len(losses) else losses
        if len(losses) and len(small) / max(len(losses), 1) > 0.5 and float(ex["bars"].median()) <= 5:
            findings.append(dict(
                code="WHIPSAW", severity="high",
                title="Whipsaw — many tiny losses, very short holds",
                detail=(f"**{len(small)}/{len(losses)}** losing trades lost under 0.5%, with a "
                        f"median hold of **{float(ex['bars'].median()):.0f} bars**. The signal is "
                        f"flipping on noise and paying costs each time."),
                fix="Add a trend filter or require N-bar confirmation before entering."))

    # --- 4. cost drag
    if cost_drag > 0 and abs(cost_drag) > abs(float(stats["TotalReturn"]) * 100) * 0.5:
        findings.append(dict(
            code="COST_DRAG", severity="high",
            title="Costs are eating the edge",
            detail=(f"The same signal returns **{bt_free.stats['TotalReturn']*100:+.1f}%** at zero "
                    f"cost but **{stats['TotalReturn']*100:+.1f}%** after costs — a drag of "
                    f"**{cost_drag:.1f}%** across **{n}** trades. The edge may be real; the "
                    f"turnover is not affordable."),
            fix="Reduce trade frequency (longer lookbacks / wider thresholds), or trade a "
                "cheaper, more liquid instrument."))

    # --- 5. tail dependence
    if n >= 8:
        top = float(ex["pnl_pct"].max())
        if gross > 0 and top > gross * 0.8:
            findings.append(dict(
                code="TAIL_DEPENDENT", severity="medium",
                title="One trade carries the whole result",
                detail=(f"The single best trade made **{top:+.2f}%** out of **{gross:+.2f}%** total. "
                        f"Remove it and the strategy is flat or negative. This is not a repeatable "
                        f"edge yet — it is one lucky event."),
                fix="Require more trades before trusting this, and size it as if the tail won't repeat."))

    # --- 6. entries are simply bad
    if len(losses):
        mfe_all = float(ex["mfe_pct"].mean())
        mae_all = float(ex["mae_pct"].abs().mean())
        if mfe_all < mae_all * 0.8:
            findings.append(dict(
                code="BAD_ENTRY", severity="high",
                title="Entries are wrong — trades go against us immediately",
                detail=(f"Average favourable excursion **{mfe_all:.2f}%** vs adverse "
                        f"**{mae_all:.2f}%**. Trades rarely work even briefly, so no exit tweak "
                        f"can save this. The signal itself has no predictive content here."),
                fix="Change the signal or the asset — do not tune the exits."))

    # --- 7. side asymmetry
    for side in ("LONG", "SHORT"):
        sub = ex[ex.side == side]
        if len(sub) >= 5 and float(sub["pnl_pct"].sum()) < 0:
            other = ex[ex.side != side]
            if len(other) >= 3 and float(other["pnl_pct"].sum()) > 0:
                findings.append(dict(
                    code=f"SIDE_{side}", severity="medium",
                    title=f"{side} side loses money; the other side works",
                    detail=(f"{side} trades: **{float(sub['pnl_pct'].sum()):+.2f}%** over "
                            f"{len(sub)} trades. Opposite side: "
                            f"**{float(other['pnl_pct'].sum()):+.2f}%**."),
                    fix=f"Disable {side.lower()}s (`allow_short=False` or long-only) on this asset."))
                break

    # --- 8. regime mismatch
    cat = cls.meta.category
    if reg.trend in ("uptrend", "downtrend") and "Reversion" in cat and stats["TotalReturn"] < 0:
        findings.append(dict(
            code="REGIME_MISMATCH", severity="medium",
            title="Mean-reversion strategy in a trending market",
            detail=f"Asset regime is **{reg.trend}** but this is a **{cat}** strategy. "
                   f"Reversion fades moves that keep going.",
            fix="Switch to a trend strategy on this asset, or wait for a sideways regime."))
    if reg.trend == "sideways" and "Trend" in cat and stats["TotalReturn"] < 0:
        findings.append(dict(
            code="REGIME_MISMATCH", severity="medium",
            title="Trend strategy in a sideways market",
            detail=f"Asset regime is **sideways** but this is a **{cat}** strategy. "
                   f"Breakouts fail repeatedly in a range.",
            fix="Switch to a mean-reversion strategy here, or wait for a trending regime."))

    if not findings:
        findings.append(dict(
            code="HEALTHY", severity="low", title="No structural failure found",
            detail="Excursions, costs and trade distribution all look normal for this style.",
            fix="No change indicated. Losses here look like ordinary variance."))

    return {"key": key, "ok": True, "stats": stats, "excursions": ex, "findings": findings,
            "cost_drag_pct": round(cost_drag, 2), "regime": reg.label(),
            "n_trades": n, "gross_pct": round(gross, 2),
            "verdict": "losing" if stats["TotalReturn"] <= 0 else "profitable"}


# ================================================================ prescriptions
def _candidate_params(key: str, base: dict, findings: list) -> list:
    """Turn diagnoses into concrete parameter changes to TEST (not to assume)."""
    codes = {f["code"] for f in findings}
    cls = REGISTRY[key]
    grid = getattr(cls, "param_grid", {}) or {}
    out = []

    def add(label, changes, rationale):
        p = {**base, **changes}
        if p != base:
            out.append({"label": label, "params": p, "rationale": rationale})

    # slow the signal down (whipsaw / cost drag / tight stops)
    if codes & {"WHIPSAW", "COST_DRAG", "STOP_TOO_TIGHT"}:
        for pname in ("n", "entry_n", "slow", "kijun", "ma_n", "z_window", "senkou_b"):
            if pname in base and isinstance(base[pname], (int, float)):
                add(f"Slow down: {pname} {base[pname]} → {int(base[pname] * 1.6)}",
                    {pname: int(base[pname] * 1.6)},
                    "Longer lookback = fewer, higher-conviction signals; less noise, lower turnover.")
                break
        for pname in ("threshold", "z_entry", "k", "rsi_buy"):
            if pname in base and isinstance(base[pname], (int, float)) and base[pname] != 0:
                add(f"Demand more: {pname} → {round(base[pname] * 1.4, 4)}",
                    {pname: type(base[pname])(round(base[pname] * 1.4, 4))},
                    "A stricter entry threshold filters marginal setups that whipsaw.")
                break

    # wider trailing / stop room
    if "STOP_TOO_TIGHT" in codes or "GIVE_BACK" in codes:
        for pname in ("atr_mult", "trail_mult", "exit_n"):
            if pname in base and isinstance(base[pname], (int, float)):
                add(f"More room: {pname} {base[pname]} → {round(base[pname] * 1.5, 3)}",
                    {pname: type(base[pname])(round(base[pname] * 1.5, 3))},
                    "Wider exit keeps the trade alive through noise instead of being stopped by it.")
                break

    # kill the losing side
    for f in findings:
        if f["code"].startswith("SIDE_") and "allow_short" in base:
            if f["code"] == "SIDE_SHORT":
                add("Long-only (shorts lose here)", {"allow_short": False},
                    "The short side is net negative on this asset; removing it keeps the working half.")

    # regime mismatch -> flip the mode if the strategy has one
    if "REGIME_MISMATCH" in codes and "mode" in base:
        other = "momentum" if base.get("mode") == "reversion" else "reversion"
        add(f"Flip mode → {other}", {"mode": other},
            "The asset's regime favours the opposite behaviour to the current mode.")

    # always test the grid's neighbours as a sanity baseline
    for pname, vals in list(grid.items())[:2]:
        for v in vals:
            if pname in base and v != base[pname]:
                add(f"Grid probe: {pname}={v}", {pname: v},
                    "Neighbouring parameter value, tested for comparison.")
                break
    # dedupe
    seen, uniq = set(), []
    for c in out:
        sig = tuple(sorted(c["params"].items()))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq[:8]


def _oos_expectancy(df: pd.DataFrame, key: str, params: dict, folds: int = 4,
                    cost_bps=1.0, slippage_bps=2.0, spread_bps=0.0,
                    financing_bps_annual=0.0) -> dict:
    """Honest score: average expectancy across held-out folds. No peeking."""
    cls = REGISTRY[key]
    n = len(df)
    if n < 200:
        folds = 2
    size = n // (folds + 1)
    exps, rets, trades = [], [], 0
    for f in range(folds):
        test = df.iloc[size * (f + 1): size * (f + 2)]
        if len(test) < 40:
            continue
        pos = cls(**params).generate(test)
        bt = run_backtest(test, pos, cost_bps=cost_bps, slippage_bps=slippage_bps,
                          spread_bps=spread_bps, financing_bps_annual=financing_bps_annual)
        ex = trade_excursions(test, pos)
        if len(ex):
            exps.append(float(ex["pnl_pct"].mean()))
            trades += len(ex)
        rets.append(float(bt.stats["TotalReturn"]))
    if not exps:
        return {"oos_expectancy": None, "oos_return": None, "oos_trades": 0}
    return {"oos_expectancy": round(float(np.mean(exps)), 4),
            "oos_return": round(float(np.mean(rets)) * 100, 3),
            "oos_trades": trades}


def prescribe(df: pd.DataFrame, key: str, base_params: dict | None = None,
              cost_bps=1.0, slippage_bps=2.0, spread_bps=0.0,
              financing_bps_annual=0.0) -> dict:
    """Full loop: diagnose → propose fixes → validate each OUT-OF-SAMPLE → recommend.

    A fix is only recommended if it beats the current configuration on held-out
    data. Otherwise it is explicitly REJECTED, and if nothing works we say the
    strategy should not be traded on this asset.
    """
    cls = REGISTRY[key]
    base = {**cls.default_params, **(base_params or {})}
    diag = diagnose(df, key, base, cost_bps, slippage_bps, spread_bps, financing_bps_annual)

    cur = _oos_expectancy(df, key, base, cost_bps=cost_bps, slippage_bps=slippage_bps,
                          spread_bps=spread_bps, financing_bps_annual=financing_bps_annual)
    cands = _candidate_params(key, base, diag["findings"])

    tested = []
    for c in cands:
        try:
            sc = _oos_expectancy(df, key, c["params"], cost_bps=cost_bps,
                                 slippage_bps=slippage_bps, spread_bps=spread_bps,
                                 financing_bps_annual=financing_bps_annual)
            if sc["oos_expectancy"] is None:
                continue
            base_e = cur["oos_expectancy"] if cur["oos_expectancy"] is not None else -99
            delta = sc["oos_expectancy"] - base_e
            tested.append({**c, **sc, "delta_expectancy": round(delta, 4),
                           "improves": bool(delta > 0 and sc["oos_expectancy"] > 0)})
        except Exception:
            continue

    tested.sort(key=lambda x: x["oos_expectancy"], reverse=True)
    winners = [t for t in tested if t["improves"]]
    best = winners[0] if winners else None

    if best:
        summary = (f"**Recommended change: {best['label']}** — out-of-sample expectancy improves "
                   f"from **{cur['oos_expectancy']:+.3f}%** to **{best['oos_expectancy']:+.3f}%** "
                   f"per trade ({best['delta_expectancy']:+.3f}pp) across held-out folds. "
                   f"{best['rationale']}")
    elif tested:
        summary = ("**No change is justified.** Every proposed fix was tested on held-out data and "
                   "none improved out-of-sample expectancy. Tuning further would be fitting noise. "
                   "If this strategy is losing here, the honest move is to trade a different "
                   "strategy or a different asset — not to keep adjusting parameters.")
    else:
        summary = ("**Not enough data to validate any change.** Too few trades in the held-out "
                   "folds to test a fix. Lengthen the window before drawing conclusions.")

    return {"diagnosis": diag, "current": cur, "candidates": tested,
            "best": best, "summary": summary, "base_params": base}


def apply_fix(key: str, base: dict, fix: dict) -> dict:
    """The updated strategy definition, ready to paste or run."""
    new = fix["params"]
    changed = {k: (base.get(k), v) for k, v in new.items() if base.get(k) != v}
    return {"strategy": key, "params": new, "changed": changed,
            "code": f"{REGISTRY[key].__name__}(" +
                    ", ".join(f"{k}={v!r}" for k, v in new.items()) + ")"}

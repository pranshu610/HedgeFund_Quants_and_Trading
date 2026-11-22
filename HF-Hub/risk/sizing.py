"""
risk/sizing.py
--------------
Position sizing and trade-level risk. Answers, for a single signal:
  * how big a position (fraction of allocated capital)
  * where the stop-loss and take-profit sit
  * scale-out / trailing logic

Three sizing methods, from most to least aggressive:
  fixed_fractional : risk a constant % of capital per trade (the retail default)
  vol_target       : size so the position's expected vol hits a target — this is
                     what pods actually do; it equalises risk across regimes
  kelly            : fraction from edge/odds, ALWAYS fractional-Kelly (0.25–0.5×)
                     because full Kelly is far too volatile to run in practice
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class TradePlan:
    size_frac: float        # fraction of allocated capital to deploy (0..1+)
    stop_price: float
    target_price: float
    rr: float               # reward:risk
    method: str
    notes: str


def fixed_fractional(capital: float, entry: float, stop: float, risk_pct: float = 0.01) -> float:
    """Shares/units such that (entry-stop) loss = risk_pct * capital."""
    per_unit_risk = abs(entry - stop)
    if per_unit_risk <= 0:
        return 0.0
    dollar_risk = capital * risk_pct
    units = dollar_risk / per_unit_risk
    return units * entry / capital  # return as fraction of capital deployed


def vol_target_size(realized_vol_ann: float, target_vol_ann: float = 0.15,
                    max_leverage: float = 3.0) -> float:
    """Leverage that scales an asset's realised vol to the target. The core of
    institutional risk parity / vol control."""
    if realized_vol_ann <= 0:
        return 0.0
    return float(np.clip(target_vol_ann / realized_vol_ann, 0.0, max_leverage))


def fractional_kelly(win_rate: float, rr: float, fraction: float = 0.33) -> float:
    """
    Kelly fraction f* = W - (1-W)/RR, then scaled down. Never bet full Kelly.
    Returns 0 if edge is non-positive.
    """
    if rr <= 0:
        return 0.0
    f_star = win_rate - (1 - win_rate) / rr
    return float(max(0.0, f_star) * fraction)


def atr_bracket(entry: float, atr: float, direction: int,
                stop_mult: float = 2.0, target_mult: float = 3.0) -> tuple[float, float]:
    """ATR-based stop and target. direction: +1 long, -1 short."""
    stop = entry - direction * stop_mult * atr
    target = entry + direction * target_mult * atr
    return float(stop), float(target)


def build_trade_plan(
    capital: float, entry: float, atr: float, direction: int,
    realized_vol_ann: float, method: str = "vol_target",
    win_rate: float = 0.5, target_vol_ann: float = 0.15,
    risk_pct: float = 0.01,
) -> TradePlan:
    stop, target = atr_bracket(entry, atr, direction)
    rr = abs(target - entry) / max(abs(entry - stop), 1e-9)

    if method == "vol_target":
        size = vol_target_size(realized_vol_ann, target_vol_ann)
        notes = f"Vol-targeted to {target_vol_ann:.0%} ann. from realised {realized_vol_ann:.0%}."
    elif method == "kelly":
        size = fractional_kelly(win_rate, rr) 
        notes = f"⅓-Kelly on win_rate={win_rate:.0%}, RR={rr:.2f}."
    else:
        size = fixed_fractional(capital, entry, stop, risk_pct)
        notes = f"Fixed-fractional risking {risk_pct:.1%} of capital."

    return TradePlan(size_frac=round(float(size), 4), stop_price=round(stop, 2),
                     target_price=round(target, 2), rr=round(rr, 2),
                     method=method, notes=notes)


# ----------------------------------------------------------------------------
# Trade management: trailing stops + scale-out ("running SL / TP bit by bit")
# ----------------------------------------------------------------------------
def trade_management_plan(entry: float, atr: float, direction: int,
                          stop_mult: float = 2.0, trail_mult: float = 2.5,
                          scale_levels: tuple = (1.0, 2.0), scale_fracs: tuple = (0.33, 0.33)):
    """
    A full trade-management plan, the way a desk actually runs a position:
      * initial stop at stop_mult × ATR
      * scale OUT a fraction at each R-multiple target (lock profits bit by bit)
      * trail the REMAINDER at trail_mult × ATR behind the best price
    Returns a dict ready for display / execution.
    """
    risk = stop_mult * atr
    initial_stop = entry - direction * risk
    scale_outs = [
        {"level": round(entry + direction * risk * r, 2), "take_frac": f,
         "label": f"take {f:.0%} at +{r:g}R"}
        for r, f in zip(scale_levels, scale_fracs)
    ]
    runner_frac = round(1 - sum(scale_fracs), 2)
    return {
        "entry": round(entry, 2),
        "initial_stop": round(initial_stop, 2),
        "risk_per_unit": round(risk, 2),
        "scale_outs": scale_outs,
        "runner": {"frac": runner_frac,
                   "rule": f"trail remaining {runner_frac:.0%} at {trail_mult}×ATR "
                           f"({trail_mult * atr:.2f}) behind best price"},
        "summary": (f"Stop {initial_stop:.2f} ({stop_mult}×ATR). "
                    + "; ".join(s['label'] for s in scale_outs)
                    + f"; trail last {runner_frac:.0%} at {trail_mult}×ATR."),
    }


def trailing_stop_path(prices, entry_idx: int, direction: int, atr_series, trail_mult: float = 2.5):
    """Bar-by-bar trailing stop levels from entry onward (for charting the trail)."""
    import numpy as _np
    px = prices.iloc[entry_idx:]
    a = atr_series.iloc[entry_idx:]
    best = px.iloc[0]
    trail = []
    for p, atr in zip(px.values, a.values):
        best = max(best, p) if direction > 0 else min(best, p)
        trail.append(best - direction * trail_mult * (atr if not _np.isnan(atr) else 0))
    import pandas as _pd
    return _pd.Series(trail, index=px.index)

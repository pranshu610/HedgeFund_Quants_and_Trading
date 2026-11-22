"""
core/regime.py
--------------
Market-regime detector. Classifies each bar into a (trend, volatility, volume)
regime and — crucially — maps the CURRENT regime to which strategy categories to
favour vs avoid. This is the connective tissue that turns 16 independent
strategies into one coherent book: run trend-followers in trends, mean-reversion
in chop, and cut gross exposure when volatility spikes.

Everything here is causal (trailing windows only). No bar uses future data.

Answers the 'market regime detection' prompt:
  * Trend       — bull / bear / sideways   (100-bar regression slope + 200-MA)
  * Volatility  — low / normal / high / extreme  (rolling percentile of realised vol)
  * Volume      — rising / normal / falling  (20d vs 60d average volume)
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from core.indicators import sma, realized_vol, rolling_slope


@dataclass
class Regime:
    trend: str          # "bull" | "bear" | "sideways"
    volatility: str     # "low" | "normal" | "high" | "extreme"
    volume: str         # "rising" | "normal" | "falling"
    rv_annual: float
    slope_norm: float

    def label(self) -> str:
        return f"{self.trend} / {self.volatility}-vol / {self.volume}-volume"


def classify(df: pd.DataFrame, slope_n: int = 100, ma_n: int = 200,
             vol_n: int = 20, vol_lookback: int = 252,
             slope_thresh: float = 0.05) -> pd.DataFrame:
    """Return a per-bar DataFrame with trend / volatility / volume labels."""
    close, vol = df["close"], df["volume"]

    # --- trend ---
    slope = rolling_slope(close, slope_n) / close       # normalised slope
    ma = sma(close, ma_n)
    trend = pd.Series("sideways", index=df.index)
    trend[(slope > slope_thresh) & (close > ma)] = "bull"
    trend[(slope < -slope_thresh) & (close < ma)] = "bear"

    # --- volatility: rolling percentile rank of realised vol (causal) ---
    rv = realized_vol(close, vol_n)
    pct = rv.rolling(vol_lookback, min_periods=vol_n * 2).apply(
        lambda w: (w[-1] >= w).mean(), raw=True)
    volat = pd.Series("normal", index=df.index)
    volat[pct <= 0.25] = "low"
    volat[pct >= 0.75] = "high"
    volat[pct >= 0.95] = "extreme"

    # --- volume behaviour ---
    v_fast, v_slow = sma(vol, 20), sma(vol, 60)
    ratio = v_fast / v_slow
    volume = pd.Series("normal", index=df.index)
    volume[ratio >= 1.15] = "rising"
    volume[ratio <= 0.85] = "falling"

    return pd.DataFrame({"trend": trend, "volatility": volat, "volume": volume,
                         "rv_annual": rv, "slope_norm": slope})


def latest_regime(df: pd.DataFrame, **kw) -> Regime:
    r = classify(df, **kw).iloc[-1]
    return Regime(trend=r["trend"], volatility=r["volatility"], volume=r["volume"],
                  rv_annual=round(float(r["rv_annual"]), 4),
                  slope_norm=round(float(r["slope_norm"]), 5))


# --- regime → strategy routing -------------------------------------------------
# Which strategy CATEGORIES fit / to avoid in each regime, plus a gross-exposure
# multiplier that de-risks as volatility rises. Categories match StrategyMeta.category.
def route(regime: Regime) -> dict:
    favour, avoid = [], []

    if regime.trend in ("bull", "bear"):
        favour += ["Trend Following", "Quant Factor"]
        avoid += ["Mean Reversion"]
    else:  # sideways
        favour += ["Mean Reversion", "Statistical Arb", "Volatility"]
        avoid += ["Trend Following"]

    if regime.volatility == "extreme":
        favour = ["Volatility", "Mean Reversion"]
        avoid += ["Trend Following", "Market Making"]
    elif regime.volatility == "low":
        favour += ["Trend Following"]  # low-vol trends are the cleanest

    gross = {"low": 1.0, "normal": 1.0, "high": 0.6, "extreme": 0.35}[regime.volatility]

    return {
        "regime": regime.label(),
        "favour": sorted(set(favour)),
        "avoid": sorted(set(avoid) - set(favour)),
        "gross_exposure": gross,
        "note": f"Realised vol {regime.rv_annual:.0%} ann.; scale book gross to {gross:.0%}.",
    }


def strategy_multipliers(regime: Regime, strategy_categories: dict[str, str]) -> dict[str, float]:
    """
    Turn the routing into a per-strategy exposure multiplier (0..1) the allocator
    can apply on top of its weights. Favoured category → full; avoided → damped.
    """
    r = route(regime)
    out = {}
    for name, cat in strategy_categories.items():
        m = r["gross_exposure"]
        if cat in r["favour"]:
            m *= 1.0
        elif cat in r["avoid"]:
            m *= 0.25
        else:
            m *= 0.6
        out[name] = round(m, 3)
    return out

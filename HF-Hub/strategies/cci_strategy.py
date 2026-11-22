"""
strategies/cci_strategy.py
--------------------------
Commodity Channel Index (CCI) — a classic day-trading oscillator. CCI measures
how far price has moved from its statistical mean in units of mean deviation.
±100 are the conventional thresholds.

Two modes (configurable):
  reversion : fade extremes — long below -threshold, short above +threshold,
              exit back toward zero. Works in ranges.
  momentum  : ride breakouts — long when CCI crosses ABOVE +threshold, short
              when it crosses below -threshold. Works in trends.

Single-series (plugs into the backtester, scanner, blotter).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta
from core.indicators import sma


def cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = sma(tp, n)
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad.replace(0, np.nan))


class CCIStrategy(BaseStrategy):
    meta = StrategyMeta(
        name="CCI",
        category="Mean Reversion",
        asset_classes=["equities", "crypto", "fx", "commodities"],
        timeframe="1d",
        horizon="intraday",
        uses_short=True,
        description="Commodity Channel Index oscillator — fade extremes (reversion) or ride breakouts (momentum).",
    )
    default_params = dict(n=20, threshold=100, mode="reversion", exit=0, allow_short=True)
    param_grid = dict(n=[14, 20, 30], threshold=[80, 100, 150], mode=["reversion", "momentum"])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        c = cci(df, p["n"]).fillna(0.0).values
        thr = p["threshold"]
        out = np.zeros(len(df))
        holding = 0.0
        for i in range(len(df)):
            if p["mode"] == "reversion":
                if holding == 0.0:
                    if c[i] <= -thr:
                        holding = 1.0
                    elif p["allow_short"] and c[i] >= thr:
                        holding = -1.0
                elif (holding > 0 and c[i] >= p["exit"]) or (holding < 0 and c[i] <= p["exit"]):
                    holding = 0.0
            else:  # momentum
                if c[i] >= thr:
                    holding = 1.0
                elif p["allow_short"] and c[i] <= -thr:
                    holding = -1.0
                elif abs(c[i]) < p["exit"] if p["exit"] else False:
                    holding = 0.0
            out[i] = holding
        return pd.Series(out, index=df.index)

    def _features(self, df):
        return {"cci": round(float(cci(df, self.params["n"]).iloc[-1]), 1)}

    def _explain(self, df):
        v = self._features(df)["cci"]
        return (f"CCI({self.params['n']}) = {v} (thresholds ±{self.params['threshold']}, "
                f"mode: {self.params['mode']}). "
                f"Thesis: {'fade the extreme' if self.params['mode']=='reversion' else 'ride the breakout'}.")

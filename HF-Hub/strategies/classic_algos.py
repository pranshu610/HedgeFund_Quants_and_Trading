"""
strategies/classic_algos.py
---------------------------
Three more single-series strategies day-traders actually use:

  MACD          — trend/momentum from the MACD line vs its signal line.
  BollingerRev  — mean-reversion: fade moves to the outer Bollinger band.
  OvernightDrift— the overnight anomaly: hold from close to next open only
                  (the historical equity risk premium concentrates overnight).
                  Also the clearest illustration of overnight/financing cost.

All plug into the backtester, scanner, blotter and optimizer like the rest.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta
from core.indicators import ema, sma, bollinger


class MACD(BaseStrategy):
    meta = StrategyMeta(
        name="MACD", category="Trend Following",
        asset_classes=["equities", "crypto", "fx", "futures"], timeframe="1d",
        horizon="swing", uses_short=True,
        description="MACD line vs signal line — long when momentum turns up, short when it turns down.")
    default_params = dict(fast=12, slow=26, signal=9, allow_short=True)
    param_grid = dict(fast=[8, 12, 19], slow=[21, 26, 39], signal=[7, 9, 12])

    def _macd(self, df):
        macd = ema(df["close"], self.params["fast"]) - ema(df["close"], self.params["slow"])
        sig = ema(macd, self.params["signal"])
        return macd, sig

    def generate(self, df: pd.DataFrame) -> pd.Series:
        macd, sig = self._macd(df)
        pos = pd.Series(0.0, index=df.index)
        pos[macd > sig] = 1.0
        if self.params["allow_short"]:
            pos[macd < sig] = -1.0
        return pos.fillna(0.0)

    def _explain(self, df):
        macd, sig = self._macd(df)
        up = macd.iloc[-1] > sig.iloc[-1]
        return (f"MACD {'above' if up else 'below'} signal ({macd.iloc[-1]:.3f} vs {sig.iloc[-1]:.3f}) → "
                f"{'long, momentum rising' if up else 'short/flat, momentum falling'}.")


class BollingerRev(BaseStrategy):
    meta = StrategyMeta(
        name="Bollinger Reversion", category="Mean Reversion",
        asset_classes=["equities", "fx", "crypto"], timeframe="1d",
        horizon="swing", uses_short=True,
        description="Fade stretched moves: long below the lower band, short above the upper, exit at the mean.")
    default_params = dict(n=20, k=2.0, allow_short=True)
    param_grid = dict(n=[14, 20, 30], k=[1.5, 2.0, 2.5])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        mid, up, lo = bollinger(df["close"], self.params["n"], self.params["k"])
        c = df["close"].values
        mid_v, up_v, lo_v = mid.values, up.values, lo.values
        out = np.zeros(len(df)); hold = 0.0
        for i in range(len(df)):
            if np.isnan(up_v[i]):
                out[i] = 0.0; continue
            if hold == 0.0:
                if c[i] < lo_v[i]:
                    hold = 1.0
                elif self.params["allow_short"] and c[i] > up_v[i]:
                    hold = -1.0
            elif (hold > 0 and c[i] >= mid_v[i]) or (hold < 0 and c[i] <= mid_v[i]):
                hold = 0.0
            out[i] = hold
        return pd.Series(out, index=df.index)

    def _explain(self, df):
        mid, up, lo = bollinger(df["close"], self.params["n"], self.params["k"])
        c = float(df["close"].iloc[-1])
        where = ("below the lower band (long setup)" if c < lo.iloc[-1]
                 else "above the upper band (short setup)" if c > up.iloc[-1]
                 else "inside the bands (hold/flat)")
        return f"Price {c:.2f} is {where}. Thesis: fade the stretch back toward the {self.params['n']}-bar mean."


class OvernightDrift(BaseStrategy):
    meta = StrategyMeta(
        name="Overnight Drift", category="Event / flow",
        asset_classes=["equities", "etf"], timeframe="1d",
        horizon="overnight", uses_short=False,
        description="Hold from close to next open only — the equity premium historically concentrates overnight.")
    default_params = dict(trend_filter=True, ma=100)
    param_grid = dict(ma=[50, 100, 200])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        # long into every close (captured as next-bar exposure by the 1-bar lag),
        # optionally only when above a long-term MA (risk-off filter).
        pos = pd.Series(1.0, index=df.index)
        if self.params.get("trend_filter", True):
            ma = sma(df["close"], self.params["ma"])
            pos[df["close"] < ma] = 0.0
        return pos.fillna(0.0)

    def _explain(self, df):
        return ("Holds overnight only — historically most of the equity risk premium accrues from "
                "close to next open, not during the session. Note: financing/borrow cost matters here, "
                "so set the overnight-financing slider to see the honest net.")

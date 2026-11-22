"""
strategies/ichimoku.py
----------------------
Ichimoku Kinko Hyo — a complete trend/momentum system in one indicator, popular
in discretionary and algo day-trading. Trades the classic confluence: price above
the cloud (Kumo) AND Tenkan above Kijun = bullish; the mirror = bearish.

Components (all trailing / causal for signals):
  Tenkan-sen (conversion) : midpoint of the last `tenkan` bars' high/low
  Kijun-sen  (base)       : midpoint of the last `kijun` bars' high/low
  Senkou A   (lead 1)     : (Tenkan+Kijun)/2, projected `shift` bars ahead
  Senkou B   (lead 2)     : midpoint of last `senkou_b` bars, projected ahead
The cloud at time t is built from data `shift` bars in the past, so comparing
today's price to it is causal.

Single-series (plugs into the backtester, scanner, blotter). Configurable periods.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta


class Ichimoku(BaseStrategy):
    meta = StrategyMeta(
        name="Ichimoku",
        category="Trend Following",
        asset_classes=["equities", "crypto", "fx", "futures"],
        timeframe="1d",
        horizon="swing",
        uses_short=True,
        description="Trade cloud breakouts: long above Kumo with Tenkan>Kijun, short below.",
    )
    default_params = dict(tenkan=9, kijun=26, senkou_b=52, shift=26, allow_short=True)
    param_grid = dict(tenkan=[7, 9, 12], kijun=[22, 26, 30], senkou_b=[44, 52, 60])

    def _lines(self, df):
        p = self.params
        hi, lo = df["high"], df["low"]
        tenkan = (hi.rolling(p["tenkan"]).max() + lo.rolling(p["tenkan"]).min()) / 2
        kijun = (hi.rolling(p["kijun"]).max() + lo.rolling(p["kijun"]).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(p["shift"])
        senkou_b = ((hi.rolling(p["senkou_b"]).max() + lo.rolling(p["senkou_b"]).min()) / 2).shift(p["shift"])
        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bot = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)
        return tenkan, kijun, cloud_top, cloud_bot

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        c = df["close"]
        tenkan, kijun, top, bot = self._lines(df)
        long_ok = (c > top) & (tenkan > kijun)
        short_ok = (c < bot) & (tenkan < kijun)
        pos = pd.Series(0.0, index=df.index)
        pos[long_ok] = 1.0
        if p["allow_short"]:
            pos[short_ok] = -1.0
        return pos.fillna(0.0)

    def _features(self, df):
        tenkan, kijun, top, bot = self._lines(df)
        c = float(df["close"].iloc[-1])
        return {"close": round(c, 2), "cloud_top": round(float(top.iloc[-1]), 2),
                "cloud_bot": round(float(bot.iloc[-1]), 2),
                "tenkan>kijun": bool(tenkan.iloc[-1] > kijun.iloc[-1])}

    def _explain(self, df):
        f = self._features(df)
        where = "above" if f["close"] > f["cloud_top"] else "below" if f["close"] < f["cloud_bot"] else "inside"
        return (f"Price {f['close']} is {where} the cloud ({f['cloud_bot']}–{f['cloud_top']}); "
                f"Tenkan {'>' if f['tenkan>kijun'] else '<'} Kijun. "
                f"Thesis: trade the trend confirmed by full Ichimoku confluence.")

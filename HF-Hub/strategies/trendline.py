"""
strategies/trendline.py
-----------------------
Systematic 'trendline' following without hand-drawn geometry. Fits a trailing
least-squares line to price; the sign and magnitude of its slope (t-stat scaled)
is the position. A confirmation filter requires price to be on the correct side
of a longer moving average, which kills most whipsaw. This is the honest,
non-discretionary version of drawing trendlines on a chart.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta
from core.indicators import rolling_slope, sma


class TrendlineFollow(BaseStrategy):
    meta = StrategyMeta(
        name="Trendline Follow",
        category="Trend Following",
        asset_classes=["equities", "crypto", "fx", "futures"],
        timeframe="1d",
        horizon="swing",
        uses_short=True,
        description="Position = sign of trailing regression slope, filtered by a long MA.",
    )
    default_params = dict(slope_n=20, ma_n=100, slope_thresh=0.0, allow_short=True)
    param_grid = dict(slope_n=[15, 20, 40], ma_n=[50, 100, 200], slope_thresh=[0.0, 0.05])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        slope = rolling_slope(close, p["slope_n"]) / close  # normalise by price
        ma = sma(close, p["ma_n"])
        long_ok = close > ma
        short_ok = close < ma

        raw = np.where(slope > p["slope_thresh"], 1.0,
                       np.where(slope < -p["slope_thresh"], -1.0, 0.0))
        pos = pd.Series(raw, index=df.index)
        pos = pos.where(long_ok | (pos <= 0), 0.0)          # no longs below MA
        pos = pos.where(short_ok | (pos >= 0), 0.0)          # no shorts above MA
        if not p["allow_short"]:
            pos = pos.clip(lower=0.0)
        return pos.fillna(0.0)

    def _features(self, df):
        p = self.params
        s = float((rolling_slope(df["close"], p["slope_n"]) / df["close"]).iloc[-1])
        return {"slope_norm": round(s, 5),
                "above_ma": bool(df["close"].iloc[-1] > sma(df["close"], p["ma_n"]).iloc[-1])}

    def _explain(self, df):
        f = self._features(df)
        d = "up" if f["slope_norm"] > 0 else "down"
        return (f"Trailing {self.params['slope_n']}-bar regression slope is {d}sloping "
                f"({f['slope_norm']}); price {'above' if f['above_ma'] else 'below'} the "
                f"{self.params['ma_n']}-MA filter. Thesis: persistence of established trend.")

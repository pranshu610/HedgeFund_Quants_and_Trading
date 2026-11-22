"""
strategies/commodity_trend.py
-----------------------------
Commodity trend + seasonality. Commodities trend strongly (supply/demand shocks
persist) and have real calendar seasonality (heating oil in winter, grains at
harvest, etc.). This blends a trend signal (fast/slow MA crossover, sign) with a
month-of-year seasonal bias estimated from the asset's own history (expanding, so
no lookahead). Single-series, so it plugs straight into the standard backtester
and the dashboard.

Edge rationale: momentum + a genuine seasonal risk premium in physical
commodities. High capacity via futures.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta
from core.indicators import sma


class CommodityTrend(BaseStrategy):
    meta = StrategyMeta(
        name="Commodity Trend+Seasonal",
        category="Trend/Seasonal",
        asset_classes=["commodities", "futures"],
        timeframe="1d",
        horizon="position",
        uses_short=True,
        description="MA-crossover trend blended with an expanding month-of-year seasonal bias.",
    )
    default_params = dict(fast=20, slow=100, seasonal_w=0.3, allow_short=True)
    param_grid = dict(fast=[10, 20, 40], slow=[50, 100, 200], seasonal_w=[0.0, 0.3, 0.5])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        trend = np.sign(sma(close, p["fast"]) - sma(close, p["slow"])).fillna(0.0)

        # expanding seasonal bias: mean monthly return using only PAST data
        ret = close.pct_change()
        month = df.index.month
        seasonal = pd.Series(0.0, index=df.index)
        cum_by_month: dict[int, list] = {m: [] for m in range(1, 13)}
        vals = ret.values
        for i in range(len(df)):
            m = month[i]
            hist = cum_by_month[m]
            seasonal.iloc[i] = np.sign(np.mean(hist)) if len(hist) >= 2 else 0.0
            if not np.isnan(vals[i]):
                hist.append(vals[i])

        raw = (1 - p["seasonal_w"]) * trend + p["seasonal_w"] * seasonal
        pos = np.sign(raw)
        if not p["allow_short"]:
            pos = pos.clip(lower=0.0)
        return pos.fillna(0.0)

    def _explain(self, df):
        p = self.params
        tr = float(np.sign(sma(df["close"], p["fast"]).iloc[-1] - sma(df["close"], p["slow"]).iloc[-1]))
        return (f"Trend {'up' if tr > 0 else 'down' if tr < 0 else 'flat'} "
                f"({p['fast']}/{p['slow']} MA), blended {p['seasonal_w']:.0%} with month-{df.index[-1].month} "
                f"seasonal bias. Thesis: persistent supply/demand momentum + calendar seasonality.")

"""
strategies/vol_arb.py
---------------------
Volatility arbitrage — variance-risk-premium (VRP) harvesting. Implied vol
(what options price in) sits ABOVE subsequently-realized vol most of the time,
because option buyers pay a premium for insurance. Selling that insurance —
short variance — collects the premium: each day you receive implied variance and
pay realized variance. It's a positive-carry, negative-skew strategy: small
steady gains punctuated by sharp losses when realized vol spikes above implied
(the 2018 'Volmageddon' profile). So the model flips LONG vol when realized
exceeds implied, and the risk engine must size it small.

We can't trade options directly here, so the strategy trades a synthetic
short/long-variance book whose daily P&L = position · (implied_var − realized_var)
· vega_scale, then hands a normal price series to the backtester (costs, exec lag
all apply). Swap `df['vix']` for a per-name option-implied vol on real data.

Edge rationale: variance risk premium (a genuine, persistent premium for bearing
crash risk) — NOT free money. The negative skew is the whole point; respect it.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.indicators import realized_vol
from core.strategy import BaseStrategy, StrategyMeta


class VolArb(BaseStrategy):
    meta = StrategyMeta(
        name="Vol Arb (VRP)",
        category="Volatility",
        asset_classes=["options", "equities"],
        timeframe="1d",
        horizon="swing",
        uses_short=True,
        description="Short variance when implied>realized (harvest VRP); long vol when realized spikes.",
    )
    default_params = dict(rv_window=20, vrp_entry=0.02, vrp_exit=0.0,
                          vega_scale=10.0, allow_long_vol=True)
    param_grid = dict(rv_window=[10, 20, 30], vrp_entry=[0.0, 0.02, 0.04],
                      vega_scale=[20.0, 40.0])

    def _vrp(self, df: pd.DataFrame) -> pd.Series:
        """Variance-risk-premium signal in VOL points: implied − trailing realized."""
        implied = df["vix"] / 100 if "vix" in df.columns else realized_vol(df["close"], 30)
        rv = realized_vol(df["close"], self.params["rv_window"])
        return (implied - rv).fillna(0.0)

    def generate(self, df: pd.DataFrame) -> pd.Series:
        """Position on the short-variance book: +1 short vol, -1 long vol, 0 flat."""
        p = self.params
        vrp = self._vrp(df).values
        out = np.zeros(len(df))
        holding = 0.0
        for i in range(len(df)):
            if holding == 0.0:
                if vrp[i] >= p["vrp_entry"]:
                    holding = 1.0                      # implied rich → short vol
                elif p["allow_long_vol"] and vrp[i] <= -p["vrp_entry"]:
                    holding = -1.0                     # realized rich → long vol
            else:
                if (holding > 0 and vrp[i] <= p["vrp_exit"]) or \
                   (holding < 0 and vrp[i] >= -p["vrp_exit"]):
                    holding = 0.0
            out[i] = holding
        return pd.Series(out, index=df.index)

    def vrp_returns(self, df: pd.DataFrame) -> pd.Series:
        """Daily variance P&L per unit position (short-vol convention)."""
        p = self.params
        implied = (df["vix"] / 100) if "vix" in df.columns else realized_vol(df["close"], 30)
        iv_var_daily = (implied.shift(1) ** 2) / 252           # struck at prior implied
        rv_var_daily = df["close"].pct_change() ** 2           # today's realized
        return ((iv_var_daily - rv_var_daily) * p["vega_scale"]).fillna(0.0)

    def vol_equity_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Positive price index of the short-variance book for the backtester."""
        pos = self.generate(df)
        pnl = pos.shift(1).fillna(0.0) * self.vrp_returns(df)
        px = 100 * (1 + pnl).cumprod()
        return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                            index=df.index)

    def _features(self, df):
        return {"vrp": round(float(self._vrp(df).iloc[-1]), 4),
                "implied": round(float(df["vix"].iloc[-1]), 1) if "vix" in df else None}

    def _explain(self, df):
        f = self._features(df)
        stance = "SHORT vol (harvest premium)" if f["vrp"] > 0 else "LONG vol (realized rich)"
        return (f"VRP = implied − realized = {f['vrp']:+.3f} vol pts → {stance}. "
                f"Thesis: collect the variance risk premium; expect negative skew, size small.")

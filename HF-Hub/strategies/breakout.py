"""
strategies/breakout.py
----------------------
Donchian-channel breakout — the canonical trend-following / CTA entry. Go long
on a break of the N-bar high, exit on a break of the shorter M-bar low or an
ATR trailing stop. Works across futures, FX, crypto, equities. Thrives in strong
trends, bleeds in chop — the optimiser's job is to find channel lengths that
survive both regimes on the target asset.

Edge rationale: momentum / underreaction to new information; robust and high-
capacity, which is why it's a genuine institutional workhorse rather than a
curve-fit.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta
from core.indicators import donchian, atr


class DonchianBreakout(BaseStrategy):
    meta = StrategyMeta(
        name="Donchian Breakout",
        category="Trend Following",
        asset_classes=["futures", "crypto", "fx", "equities"],
        timeframe="1d",
        horizon="position",
        uses_short=True,
        description="Long N-bar high breakouts / short N-bar low breakdowns, ATR trailing exit.",
    )
    default_params = dict(entry_n=20, exit_n=10, atr_n=14, atr_mult=3.0, allow_short=True)
    param_grid = dict(entry_n=[20, 40, 55], exit_n=[10, 20], atr_mult=[2.0, 3.0, 4.0])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        up, _ = donchian(df, p["entry_n"])
        _, dn_exit = donchian(df, p["exit_n"])
        up_exit, _ = donchian(df, p["exit_n"])
        dn_entry = df["low"].rolling(p["entry_n"], min_periods=p["entry_n"]).min()
        a = atr(df, p["atr_n"])
        close = df["close"].values
        upv, dnv = up.shift(1).values, dn_entry.shift(1).values
        up_exv, dn_exv = up_exit.shift(1).values, dn_exit.shift(1).values
        av = a.values

        out = np.zeros(len(df))
        holding = 0.0
        entry_px = np.nan
        for i in range(len(df)):
            c = close[i]
            if holding == 0.0:
                if not np.isnan(upv[i]) and c > upv[i]:
                    holding, entry_px = 1.0, c
                elif p["allow_short"] and not np.isnan(dnv[i]) and c < dnv[i]:
                    holding, entry_px = -1.0, c
            elif holding > 0:
                trail = entry_px - p["atr_mult"] * av[i] if not np.isnan(av[i]) else -np.inf
                if c < dn_exv[i] or c < trail:
                    holding = 0.0
                else:
                    entry_px = max(entry_px, c)  # ratchet trailing anchor
            elif holding < 0:
                trail = entry_px + p["atr_mult"] * av[i] if not np.isnan(av[i]) else np.inf
                if c > up_exv[i] or c > trail:
                    holding = 0.0
                else:
                    entry_px = min(entry_px, c)
            out[i] = holding
        return pd.Series(out, index=df.index)

    def _features(self, df):
        p = self.params
        up, _ = donchian(df, p["entry_n"])
        return {"close": round(float(df["close"].iloc[-1]), 2),
                "entry_high": round(float(up.iloc[-1]), 2),
                "atr": round(float(atr(df, p["atr_n"]).iloc[-1]), 2)}

    def _explain(self, df):
        f = self._features(df)
        state = "above" if f["close"] > f["entry_high"] else "below"
        return (f"Close {f['close']} is {state} the {self.params['entry_n']}-bar high "
                f"{f['entry_high']}; ATR {f['atr']} sets a {self.params['atr_mult']}× trailing stop. "
                f"Thesis: ride established momentum, cut when the trend structure breaks.")

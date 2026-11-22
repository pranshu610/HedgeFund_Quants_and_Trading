"""
strategies/vix_rsi.py
---------------------
VIX-gated RSI mean reversion. Classic, well-documented equity-index edge:
oversold RSI dip-buys work far better when fear (VIX) is elevated, because
that's when short-term overreaction is largest. Fades stretched selloffs, exits
on mean reversion. Long-only by default (index dip-buying); short leg optional.

Edge rationale: short-horizon overreaction in index prices + volatility risk
premium. Crowded and low-capacity, so treat expectancy as modest and cost-
sensitive — exactly why the backtester charges fees/slippage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import BaseStrategy, StrategyMeta
from core.indicators import rsi


class VixRsiMeanReversion(BaseStrategy):
    meta = StrategyMeta(
        name="VIX+RSI Mean Reversion",
        category="Mean Reversion",
        asset_classes=["equities", "etf"],
        timeframe="1d",
        horizon="swing",
        uses_short=False,
        description="Dip-buy oversold RSI, gated by elevated VIX; exit on RSI recovery.",
    )
    default_params = dict(rsi_n=2, rsi_buy=10, rsi_exit=60, vix_floor=20.0, allow_short=False,
                          rsi_short=90, trend_ma=0, max_hold=0)
    # Grid kept deliberately small. A 400+ combination search over ~500 bars finds
    # noise, not edge — and makes the optimiser feel broken. These are the
    # dimensions that actually change behaviour.
    param_grid = dict(rsi_n=[2, 3], rsi_buy=[5, 10, 15], rsi_exit=[55, 60, 70],
                      vix_floor=[0.0, 22.0], trend_ma=[0, 200])

    def generate(self, df: pd.DataFrame) -> pd.Series:
        p = self.params
        close = df["close"]
        r = rsi(close, p["rsi_n"])
        vix = df["vix"] if "vix" in df.columns else pd.Series(np.nan, index=df.index)
        fear_ok = (vix >= p["vix_floor"]) | vix.isna()

        # refinement 1: trend filter — only dip-buy when price is above a long MA
        # (mean reversion inside a downtrend is a falling-knife trap)
        if p["trend_ma"] > 0:
            from core.indicators import sma
            trend_ok = (close > sma(close, p["trend_ma"])).fillna(False).values
        else:
            trend_ok = np.ones(len(df), dtype=bool)

        pos = pd.Series(0.0, index=df.index)
        holding = 0.0
        bars_held = 0
        r_arr, vix_ok_arr = r.values, fear_ok.values
        out = np.zeros(len(df))
        for i in range(len(df)):
            if holding == 0.0:
                if r_arr[i] <= p["rsi_buy"] and vix_ok_arr[i] and trend_ok[i]:
                    holding, bars_held = 1.0, 0
                elif p["allow_short"] and r_arr[i] >= p["rsi_short"] and vix_ok_arr[i]:
                    holding, bars_held = -1.0, 0
            else:
                bars_held += 1
                exit_long = holding > 0 and r_arr[i] >= p["rsi_exit"]
                exit_short = holding < 0 and r_arr[i] <= (100 - p["rsi_exit"])
                # refinement 2: time-stop — don't let a mean-reversion trade rot
                time_stop = p["max_hold"] > 0 and bars_held >= p["max_hold"]
                if exit_long or exit_short or time_stop:
                    holding, bars_held = 0.0, 0
            out[i] = holding
        pos[:] = out
        return pos

    def _features(self, df):
        p = self.params
        return {"rsi": round(float(rsi(df["close"], p["rsi_n"]).iloc[-1]), 1),
                "vix": round(float(df["vix"].iloc[-1]), 1) if "vix" in df else None}

    def _explain(self, df):
        f = self._features(df)
        vix_txt = f"VIX {f['vix']} (fear gate {'open' if (f['vix'] or 0) >= self.params['vix_floor'] else 'shut'})" \
            if f["vix"] is not None else "no VIX gate"
        return (f"RSI({self.params['rsi_n']}) = {f['rsi']} vs buy≤{self.params['rsi_buy']} / "
                f"exit≥{self.params['rsi_exit']}; {vix_txt}. "
                f"Thesis: short-term oversold snap-back amplified by elevated implied vol.")

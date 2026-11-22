"""
strategies/equity_ls.py
-----------------------
Residual-momentum equity long/short — the idiosyncratic-momentum strategy that
sophisticated shops run instead of naive price momentum. Plain 12-1 momentum is
heavily contaminated by market and sector beta, so it loads up on whatever just
rallied and crashes hard on momentum reversals. Residual momentum first strips
each stock's market beta, then ranks on the momentum of the LEFT-OVER
(idiosyncratic) return. The result has a far smoother return stream and much
smaller drawdowns than naive momentum (Blitz, Huij & Martens, 2011).

Cross-sectional, dollar-neutral, long-top / short-bottom decile. Panel strategy:
exposes `generate_panel(bars)` like the multi-factor model.

Edge rationale: underreaction to firm-specific information. Distinct from the
`multifactor` model (which blends momentum with value/low-vol/trend) — this is a
pure, beta-neutralised momentum sleeve.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta


class ResidualMomentumLS:
    meta = StrategyMeta(
        name="Residual Momentum L/S",
        category="Equity L/S",
        asset_classes=["equities"],
        timeframe="1d",
        horizon="swing",
        uses_short=True,
        description="Beta-neutralised (idiosyncratic) 12-1 momentum, long-top/short-bottom decile.",
    )
    default_params = dict(beta_window=126, mom_lookback=252, mom_skip=21,
                          rebalance="ME", top_frac=0.2, vol_scale=True)
    param_grid = dict(beta_window=[63, 126, 252], mom_lookback=[126, 252],
                      top_frac=[0.1, 0.2, 0.3])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def _residual_returns(self, rets: pd.DataFrame) -> pd.DataFrame:
        """Remove each name's market beta via a rolling regression on the
        equal-weight market, return the idiosyncratic residual."""
        p = self.params
        market = rets.mean(axis=1)
        w = p["beta_window"]
        var_m = market.rolling(w, min_periods=w // 2).var()
        resid = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
        for col in rets.columns:
            cov = rets[col].rolling(w, min_periods=w // 2).cov(market)
            beta = cov / var_m
            resid[col] = rets[col] - beta * market
        return resid

    def generate_panel(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = self.params
        closes = pd.DataFrame({s: b["close"] for s, b in bars.items()}).dropna(how="all")
        rets = closes.pct_change()
        resid = self._residual_returns(rets)

        # residual momentum = trailing sum of residuals, skipping the last month
        cum_resid = resid.rolling(p["mom_lookback"], min_periods=p["mom_lookback"] // 2).sum()
        score = cum_resid.shift(p["mom_skip"])
        if p["vol_scale"]:                       # risk-adjust the momentum signal
            idio_vol = resid.rolling(p["mom_lookback"], min_periods=60).std()
            score = score / idio_vol.replace(0, np.nan)

        rebal = closes.resample(p["rebalance"]).last().index
        top = p["top_frac"]
        weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        for dt in rebal:
            if dt not in score.index:
                continue
            row = score.loc[dt].dropna()
            if len(row) < 5:
                continue
            k = max(1, int(len(row) * top))
            target = pd.Series(0.0, index=closes.columns)
            target[row.nlargest(k).index] = 0.5 / k
            target[row.nsmallest(k).index] = -0.5 / k
            weights.loc[dt] = target
        return weights.ffill().fillna(0.0)

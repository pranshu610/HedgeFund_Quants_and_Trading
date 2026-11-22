"""
strategies/multifactor.py
-------------------------
Cross-sectional multi-factor equity model — the strategy the MM pods actually
run at scale. Ranks a UNIVERSE of names each rebalance on a weighted blend of
factors, goes long the top decile / short the bottom, dollar-neutral. This one
operates on a panel (dict of symbol -> bars), not a single series, so it exposes
`generate_panel` instead of `generate`.

Factors (all causal, all z-scored cross-sectionally):
  momentum : 12-1 month total return (skip most recent month)
  value    : proxy = -1 * 12m price change normalised (cheap = fell/lagged); swap
             in real fundamentals (E/P, B/P) when you wire a fundamentals feed
  low_vol  : -1 * trailing realised vol (low-vol premium)
  trend    : sign-scaled 100d slope

Default weights are a sane institutional starting point; the optimiser can tune
them and the rebalance frequency.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.indicators import realized_vol, rolling_slope
from core.strategy import StrategyMeta


class MultiFactorEquity:
    meta = StrategyMeta(
        name="Multi-Factor Equity",
        category="Quant Factor",
        asset_classes=["equities"],
        timeframe="1d",
        horizon="position",
        uses_short=True,
        description="Dollar-neutral long-top/short-bottom decile on momentum/value/low-vol/trend blend.",
    )
    default_params = dict(
        weights=dict(momentum=0.4, value=0.2, low_vol=0.2, trend=0.2),
        rebalance="ME", top_frac=0.2,
    )
    param_grid = dict(rebalance=["W", "ME", "QE"], top_frac=[0.1, 0.2, 0.3])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    @staticmethod
    def _z(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / (s.std() + 1e-9)

    def _factor_panel(self, closes: pd.DataFrame) -> pd.DataFrame:
        mom = closes.pct_change(252).shift(21)                 # 12-1 momentum
        value = -closes.pct_change(252)                        # value proxy: 12-month reversal until a fundamentals feed is added
        lowvol = -closes.apply(lambda c: realized_vol(c, 60))
        trend = closes.apply(lambda c: np.sign(rolling_slope(c, 100)))
        w = self.params["weights"]
        score = (w["momentum"] * mom.apply(self._z, axis=1)
                 + w["value"] * value.apply(self._z, axis=1)
                 + w["low_vol"] * lowvol.apply(self._z, axis=1)
                 + w["trend"] * trend.apply(self._z, axis=1))
        return score

    def generate_panel(self, bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a weights DataFrame (index=dates, cols=symbols) summing ~0 (dollar-neutral)."""
        closes = pd.DataFrame({s: b["close"] for s, b in bars.items()}).dropna(how="all")
        score = self._factor_panel(closes)
        rebal_dates = closes.resample(self.params["rebalance"]).last().index
        top = self.params["top_frac"]

        # write a COMPLETE target vector (zeros for unpicked) only on rebalance
        # dates, leave other rows NaN, then ffill whole rows. This preserves
        # dollar-neutrality between rebalances instead of forward-filling stale
        # individual picks.
        weights = pd.DataFrame(np.nan, index=closes.index, columns=closes.columns)
        for dt in rebal_dates:
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

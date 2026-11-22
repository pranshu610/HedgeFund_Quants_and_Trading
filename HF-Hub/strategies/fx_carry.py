"""
strategies/fx_carry.py
----------------------
FX carry — the canonical macro cross-sectional strategy. Long the highest-yield
currencies, short the lowest, funded in the base. You earn the interest-rate
differential ('carry') as long as spot doesn't move against you. It works most
of the time and then loses a lot fast in risk-off events ('picking up pennies in
front of a steamroller'), so a momentum overlay + vol-scaling are standard
risk mitigants and are built in here.

Signal blend (cross-sectional, dollar-neutral, monthly rebalance):
  carry     : short-rate differential vs the panel mean
  momentum  : trailing spot return (dampens entering a currency that's already
              rolling over — the classic carry+momentum combination)

Edge rationale: forward-rate bias / uncovered-interest-parity failure + a risk
premium for bearing crash risk. High capacity, heavily studied, crowded.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta


class FXCarry:
    meta = StrategyMeta(
        name="FX Carry",
        category="Macro",
        asset_classes=["fx"],
        timeframe="1d",
        horizon="position",
        uses_short=True,
        description="Long high-yield / short low-yield currencies, momentum-filtered, vol-scaled.",
    )
    default_params = dict(carry_w=0.7, mom_w=0.3, mom_lookback=63, rebalance="ME",
                          top_frac=0.34, vol_scale=True, vol_window=63)
    param_grid = dict(carry_w=[0.5, 0.7, 1.0], mom_lookback=[21, 63, 126], top_frac=[0.25, 0.34])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def generate_carry(self, spots: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
        """spots/rates: DataFrames (dates x currencies). Returns daily weights."""
        p = self.params
        rets = spots.pct_change()

        carry = rates.sub(rates.mean(axis=1), axis=0)           # rate differential
        mom = spots.pct_change(p["mom_lookback"])
        score = p["carry_w"] * self._z(carry) + p["mom_w"] * self._z(mom)

        rebal = spots.resample(p["rebalance"]).last().index
        top = p["top_frac"]
        weights = pd.DataFrame(np.nan, index=spots.index, columns=spots.columns)
        for dt in rebal:
            if dt not in score.index:
                continue
            row = score.loc[dt].dropna()
            if len(row) < 4:
                continue
            k = max(1, int(len(row) * top))
            target = pd.Series(0.0, index=spots.columns)
            target[row.nlargest(k).index] = 0.5 / k
            target[row.nsmallest(k).index] = -0.5 / k
            weights.loc[dt] = target
        weights = weights.ffill().fillna(0.0)

        if p["vol_scale"]:                                       # equalise position risk
            vol = rets.rolling(p["vol_window"], min_periods=20).std()
            inv = (1 / vol).replace([np.inf, -np.inf], np.nan)
            weights = (weights * inv).fillna(0.0)
            gross = weights.abs().sum(axis=1).replace(0, np.nan)
            weights = weights.div(gross, axis=0).fillna(0.0)
        return weights

    @staticmethod
    def _z(df: pd.DataFrame) -> pd.DataFrame:
        return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    def carry_returns(self, spots: pd.DataFrame, rates: pd.DataFrame) -> pd.Series:
        """Book P&L = weighted spot returns + accrued carry on held positions."""
        w = self.generate_carry(spots, rates).shift(1).fillna(0.0)
        spot_pnl = (w * spots.pct_change().fillna(0)).sum(axis=1)
        carry_pnl = (w * (rates / 100 / 252)).sum(axis=1)        # daily accrued rate
        return spot_pnl + carry_pnl

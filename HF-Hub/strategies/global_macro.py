"""
strategies/global_macro.py
--------------------------
Rules-based global macro. Reads the macro state (growth, inflation, policy rate,
yield curve) and tilts a multi-asset book — equities, bonds, gold, USD — toward
whatever the current regime rewards. This is the systematic version of a
discretionary macro book: transparent if-this-then-that rules on economic data,
not price signals.

For each factor (rates / inflation /
growth) it states how it affects each asset and produces entry/exit tilts.

Macro data is monthly and RELEASED WITH A LAG, so every indicator is shifted one
period before use — you cannot trade on a GDP print before it's published. This
lag discipline is what separates a real macro model from a lookahead fantasy.

Edge rationale: cross-asset risk premia are conditional on the cycle. Modest,
slow-moving, high-capacity — a diversifier, not a Sharpe monster.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta


class GlobalMacro:
    meta = StrategyMeta(
        name="Global Macro",
        category="Macro",
        asset_classes=["equities", "rates", "commodities", "fx"],
        timeframe="1d",
        horizon="position",
        uses_short=True,
        description="Rules on growth/inflation/rates tilt a multi-asset book (equity/bonds/gold/USD).",
    )
    default_params = dict(release_lag=1, gross=1.0, smooth=21)
    param_grid = dict(release_lag=[1, 2], smooth=[5, 21, 42])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def generate_macro(self, macro: pd.DataFrame, assets: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return daily target weights (index=dates, cols=asset names)."""
        p = self.params
        idx = next(iter(assets.values())).index
        # lag macro for publication delay, then align to daily and smooth
        m = macro.shift(p["release_lag"]).reindex(idx, method="ffill")
        g = self._z(m["growth"])                       # growth momentum
        dgrowth = self._z(m["growth"].diff())          # accelerating vs slowing
        infl = self._z(m["inflation"])
        dinfl = self._z(m["inflation"].diff())
        drate = self._z(m["policy_rate"].diff())       # hiking vs cutting
        curve = self._z(m["yield_10y"] - m["policy_rate"])  # steepness

        # --- the rule book: each asset's tilt is a sum of factor effects ---
        w = pd.DataFrame(index=idx)
        w["equity"] = 0.6 * dgrowth + 0.3 * g - 0.3 * infl - 0.2 * drate
        w["bonds"] = -0.5 * dgrowth - 0.4 * drate - 0.3 * dinfl + 0.2 * curve
        w["gold"] = 0.5 * dinfl + 0.3 * infl - 0.3 * drate
        w["usd"] = 0.4 * drate + 0.2 * g

        w = w.reindex(columns=list(assets.keys()), fill_value=0.0)
        w = w.rolling(p["smooth"], min_periods=1).mean()          # de-noise tilts
        # normalise to unit gross, apply gross scaler
        gross = w.abs().sum(axis=1).replace(0, np.nan)
        w = w.div(gross, axis=0).fillna(0.0) * p["gross"]
        return w

    @staticmethod
    def _z(s: pd.Series, n: int = 36) -> pd.Series:
        mean = s.rolling(n, min_periods=6).mean()
        sd = s.rolling(n, min_periods=6).std()
        return ((s - mean) / sd).fillna(0.0)

    def explain(self, macro: pd.DataFrame, assets: dict) -> str:
        w = self.generate_macro(macro, assets).iloc[-1]
        last = macro.iloc[-1]
        tilt = ", ".join(f"{k} {v:+.0%}" for k, v in w.items() if abs(v) > 0.02)
        return (f"Macro: growth {last['growth']:.0f}, inflation {last['inflation']:.1f}%, "
                f"policy {last['policy_rate']:.1f}%, 10Y {last['yield_10y']:.1f}%. "
                f"Book tilt → {tilt or 'flat'}.")

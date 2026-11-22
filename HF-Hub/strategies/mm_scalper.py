"""
strategies/mm_scalper.py
------------------------
Market-making scalper (inventory simulation). Real market making is
infrastructure — colocated servers, tick data, microsecond latency — so it can't
be 'backtested' on daily OHLCV. What we CAN model honestly is the economics: quote
a two-sided market around mid, earn the half-spread on each fill, and carry
inventory risk (an accumulated position marked to market against you). This
Avellaneda-Stoikov-lite simulation exposes the core tradeoff every MM lives:
capture spread vs. get run over by adverse selection and inventory.

P&L per bar = (fills · half-spread)  −  (inventory · price move)  −  fees.
Fill probability falls as you widen the quote and rises with volatility/flow.

Edge rationale: liquidity provision earns the spread, but only if inventory and
adverse selection are controlled. This is a simulator/teaching model, NOT a
deployable daily strategy — flagged as such.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta
from core.indicators import realized_vol


class MMScalper:
    meta = StrategyMeta(
        name="MM Scalper (sim)",
        category="Market Making",
        asset_classes=["crypto"],
        timeframe="1m",
        horizon="scalp",
        uses_short=True,
        description="Inventory-aware market-making simulation: earn half-spread, manage inventory risk.",
    )
    default_params = dict(spread_bps=8.0, base_fill_prob=0.35, inv_limit=20.0,
                          skew=0.5, fee_bps=0.5, seed=0)
    param_grid = dict(spread_bps=[5.0, 8.0, 12.0], skew=[0.0, 0.5, 1.0])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def simulate(self, df: pd.DataFrame) -> pd.Series:
        """Return the MM book's per-bar return series."""
        p = self.params
        rng = np.random.default_rng(p["seed"])
        px = df["close"].values
        ret = df["close"].pct_change().fillna(0).values
        vol = realized_vol(df["close"], 20).fillna(0.2).values
        half = p["spread_bps"] / 2e4

        inv = 0.0
        pnl = np.zeros(len(df))
        for i in range(1, len(df)):
            # inventory skew: when long, lower fill prob on buys (push to sell)
            flow = min(1.0, vol[i] / 0.2) * p["base_fill_prob"]
            skew = p["skew"] * inv / p["inv_limit"]
            buy_fill = float(rng.random() < np.clip(flow * (1 - skew), 0, 1))
            sell_fill = float(rng.random() < np.clip(flow * (1 + skew), 0, 1))

            spread_capture = (buy_fill + sell_fill) * half              # earn half-spread per fill
            fee = (buy_fill + sell_fill) * p["fee_bps"] / 1e4
            inv += (buy_fill - sell_fill)
            inv = float(np.clip(inv, -p["inv_limit"], p["inv_limit"]))
            inventory_pnl = inv * ret[i]                                # marked-to-market inventory
            pnl[i] = spread_capture - fee + inventory_pnl * 0.01        # scale inventory risk to book

        return pd.Series(pnl, index=df.index)

    def equity_df(self, df: pd.DataFrame) -> pd.DataFrame:
        r = self.simulate(df)
        px = 100 * (1 + r).cumprod()
        return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                            index=df.index)

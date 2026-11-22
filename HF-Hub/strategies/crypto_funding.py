"""
strategies/crypto_funding.py
----------------------------
Crypto funding-rate carry (cash-and-carry basis trade). Perpetual swaps pay a
periodic funding rate to keep the perp pinned to spot. When funding is positive
(the usual state in bull markets — longs pay shorts), you SHORT the perp and go
LONG spot: you're market-neutral on price and collect the funding as carry. When
funding flips negative you reverse. The classic delta-neutral crypto yield trade.

We model the delta-neutral book so its P&L is essentially the harvested funding
minus a small basis-noise term — no directional crypto exposure by construction.

Edge rationale: a structural funding/basis premium from leverage demand. Real,
persistent, but capacity-limited and it compresses as more capital arrives.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta, Signal


class CryptoFunding:
    meta = StrategyMeta(
        name="Crypto Funding Carry",
        category="Crypto Quant",
        asset_classes=["crypto"],
        timeframe="1d",
        horizon="intraday",
        uses_short=True,
        description="Delta-neutral perp/spot basis: short perp & long spot to harvest positive funding.",
    )
    default_params = dict(entry=0.0002, exit=0.0, smooth=3, basis_noise=0.0003)
    param_grid = dict(entry=[0.0, 0.0002, 0.0005], smooth=[1, 3, 5])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def generate(self, funding: pd.Series) -> pd.Series:
        """Position in {-1,0,+1}: +1 = collect positive funding (short perp/long spot)."""
        p = self.params
        f = funding.rolling(p["smooth"], min_periods=1).mean()
        pos = pd.Series(0.0, index=funding.index)
        pos[f >= p["entry"]] = 1.0
        pos[f <= -p["entry"]] = -1.0
        return pos

    def carry_returns(self, funding: pd.Series, seed: int = 0) -> pd.Series:
        """Delta-neutral book return = position · funding − small basis noise."""
        rng = np.random.default_rng(seed)
        pos = self.generate(funding).shift(1).fillna(0.0)
        noise = rng.normal(0, self.params["basis_noise"], len(funding))
        return pos * funding - np.abs(pos) * np.abs(noise)

    def equity_df(self, funding: pd.Series) -> pd.DataFrame:
        px = 100 * (1 + self.carry_returns(funding)).cumprod()
        return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                            index=funding.index)

    def latest_signal(self, funding: pd.Series, symbol: str) -> Signal:
        f = float(funding.iloc[-1])
        pos = self.generate(funding)
        d = int(pos.iloc[-1])
        stance = "short perp / long spot (collect funding)" if d > 0 else \
                 "long perp / short spot" if d < 0 else "flat"
        return Signal(timestamp=funding.index[-1], symbol=symbol, strategy=self.meta.name,
                      direction=d, strength=min(abs(f) / max(self.params["entry"], 1e-9), 1.0),
                      reason=f"Funding {f*100:.3f}%/day → {stance}. Delta-neutral carry.",
                      features={"funding_daily": round(f, 5)})

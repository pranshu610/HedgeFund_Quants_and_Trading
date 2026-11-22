"""
strategies/merger_arb.py
------------------------
Merger (risk) arbitrage. When Acquirer bids for Target in cash, the target trades
BELOW the offer until the deal closes — that gap is the arb spread, compensation
for deal risk. You buy the target on announcement and earn the spread as it
converges to the offer at close. If the deal BREAKS, the target collapses back
toward its pre-bid level and you take a large loss. Classic positive-carry,
negative-skew payoff: many small wins, occasional large loss.

Models a diversified book of simultaneous deals (equal risk each), producing a
daily return stream. Real feed: a deal database for terms + a live price feed for
the target.

Edge rationale: a genuine risk premium for underwriting deal-completion risk.
Capacity-limited and correlated in risk-off (deals break together in crises).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta


class MergerArb:
    meta = StrategyMeta(
        name="Merger Arb",
        category="Event Driven",
        asset_classes=["equities"],
        timeframe="1d",
        horizon="position",
        uses_short=False,
        description="Buy cash-deal targets below offer; earn the spread on close, lose if it breaks.",
    )
    default_params = dict(max_concurrent=10, break_loss=0.5)
    param_grid = dict(max_concurrent=[5, 10, 20], break_loss=[0.4, 0.5, 0.6])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def book_returns(self, deals: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.Series:
        """Daily equal-risk book return across all active deals over `calendar`."""
        p = self.params
        daily = pd.DataFrame(0.0, index=calendar, columns=["ret", "n"])
        for _, d in deals.iterrows():
            span = calendar[(calendar >= d["announce"]) & (calendar <= d["close"])]
            if len(span) < 2:
                continue
            spread = d["offer"] / d["pre"] - 1.0                 # total gain if it closes
            if d["completes"]:
                per_day = (1 + spread) ** (1 / len(span)) - 1     # accrue spread to close
                path = np.full(len(span), per_day)
            else:
                # drift up early, then gap down on break near the end
                per_day = (1 + spread * 0.5) ** (1 / len(span)) - 1
                path = np.full(len(span), per_day)
                path[-1] = -p["break_loss"]                       # the break gap
            daily.loc[span, "ret"] += path
            daily.loc[span, "n"] += 1
        # equal-risk average across active deals, capped concurrency
        n = daily["n"].clip(upper=p["max_concurrent"]).replace(0, np.nan)
        return (daily["ret"] / n).fillna(0.0)

    def equity_df(self, deals: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        r = self.book_returns(deals, calendar)
        px = 100 * (1 + r).cumprod()
        return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                            index=calendar)

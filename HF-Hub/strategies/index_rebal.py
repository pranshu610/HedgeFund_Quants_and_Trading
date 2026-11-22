"""
strategies/index_rebal.py
-------------------------
Index-rebalancing flow capture. When an index provider adds a stock, every fund
tracking that index is FORCED to buy it by the effective date — predictable,
price-insensitive demand that pushes the added name up into the event and often
reverses after. Deletions are the mirror. The strategy front-runs: long additions
from announcement to just after effective, short deletions, then exit.

Takes the underlying's bars + an events table (announce/effective/action) and
produces a position series, so it backtests through the standard engine.

Real feed: index provider rebalance announcements (S&P/Russell/MSCI).
Edge rationale: predictable forced flow from price-insensitive index funds — a
market-structure inefficiency, not a behavioural one. Crowded around the big
indices; more edge in smaller/less-covered ones.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta


class IndexRebal:
    meta = StrategyMeta(
        name="Index Rebalance",
        category="Flow",
        asset_classes=["equities", "etf"],
        timeframe="1d",
        horizon="event",
        uses_short=True,
        description="Long index additions / short deletions from announcement through effective date.",
    )
    default_params = dict(exit_lag=2, allow_short=True)
    param_grid = dict(exit_lag=[0, 2, 5])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def generate(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
        p = self.params
        pos = pd.Series(0.0, index=df.index)
        for _, e in events.iterrows():
            a, eff = e["announce"], e["effective"]
            exit_dt = eff + pd.Timedelta(days=p["exit_lag"])
            window = df.index[(df.index >= a) & (df.index <= exit_dt)]
            if len(window) == 0:
                continue
            sign = 1.0 if e["action"] == "add" else (-1.0 if p["allow_short"] else 0.0)
            pos.loc[window] = sign
        return pos

    def latest_event(self, events: pd.DataFrame, asof: pd.Timestamp) -> str:
        past = events[events["announce"] <= asof]
        if len(past) == 0:
            return "no index events yet"
        e = past.iloc[-1]
        return f"{e['action']} announced {e['announce'].date()}, effective {e['effective'].date()}"

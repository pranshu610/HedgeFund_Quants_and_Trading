"""
strategies/event_earnings.py
----------------------------
Post-Earnings-Announcement Drift (PEAD) — the most durable event-driven anomaly.
Stocks that beat earnings keep drifting up (and misses keep drifting down) for
weeks after the announcement, because the market underreacts to the information.
The rule: on a positive standardised earnings surprise (SUE), go long for a
holding window; on a negative surprise, go short.

Causal by construction — you only trade AFTER the announcement is public, and the
position lasts a fixed drift window. Real feed: earnings calendar + estimate data
for the surprise.

Edge rationale: underreaction / limited attention. Well documented, decaying as
it gets arbitraged, and concentrated around a few dates per name — so per-name
capacity is low but it diversifies across a universe.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta, Signal


class EarningsDrift:
    meta = StrategyMeta(
        name="PEAD (Earnings Drift)",
        category="Event Driven",
        asset_classes=["equities"],
        timeframe="1d",
        horizon="event",
        uses_short=True,
        description="Trade post-earnings drift: long positive surprises / short negative, fixed window.",
    )
    default_params = dict(sue_threshold=0.5, hold_days=40, allow_short=True)
    param_grid = dict(sue_threshold=[0.0, 0.5, 1.0], hold_days=[20, 40, 60])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    def generate(self, df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
        """events: DataFrame indexed by announce date with a 'surprise' column."""
        p = self.params
        pos = pd.Series(0.0, index=df.index)
        for dt, row in events.iterrows():
            if dt not in df.index:
                # snap to next available trading day
                later = df.index[df.index >= dt]
                if len(later) == 0:
                    continue
                dt = later[0]
            sue = row["surprise"]
            if sue >= p["sue_threshold"]:
                sign = 1.0
            elif p["allow_short"] and sue <= -p["sue_threshold"]:
                sign = -1.0
            else:
                continue
            start = df.index.get_loc(dt)
            end = min(start + p["hold_days"], len(df))
            pos.iloc[start:end] = sign          # enter day AFTER announcement (drift window)
        return pos

    def latest_signal(self, df: pd.DataFrame, events: pd.DataFrame, symbol: str) -> Signal:
        pos = self.generate(df, events)
        d = int(np.sign(pos.iloc[-1]))
        recent = events[events.index <= df.index[-1]]
        sue = float(recent["surprise"].iloc[-1]) if len(recent) else 0.0
        return Signal(timestamp=df.index[-1], symbol=symbol, strategy=self.meta.name,
                      direction=d, strength=min(abs(sue), 1.0),
                      reason=f"Last SUE {sue:+.2f} (thresh ±{self.params['sue_threshold']}); "
                             f"{self.params['hold_days']}d drift window "
                             f"{'active' if d != 0 else 'elapsed'}. Thesis: underreaction to earnings.",
                      features={"last_sue": round(sue, 2)})

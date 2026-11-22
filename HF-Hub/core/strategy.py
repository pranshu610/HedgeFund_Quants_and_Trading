"""
core/strategy.py
----------------
The contract every strategy implements. A strategy's ONLY job is to turn bars
into a target position series in [-1, 1] (fraction of the strategy's allocated
capital, sign = long/short). It does NOT decide dollar size, stops, or leverage
— that's the risk engine's job. This separation is what lets one backtester and
one risk model serve every strategy.

Each strategy also declares metadata (asset class, native timeframe, holding
period, whether shorting is used) and a `default_params` / `param_grid` so the
walk-forward optimiser can search settings and recommend the optimal config.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import pandas as pd


@dataclass
class StrategyMeta:
    name: str
    category: str            # e.g. "Mean Reversion", "Trend Following", "Volatility"
    asset_classes: list      # ["equities", "crypto", "fx", ...]
    timeframe: str           # native bar interval: "1d", "1h", "1m"
    horizon: str             # "scalp", "intraday", "swing", "position", "event"
    uses_short: bool
    description: str


@dataclass
class Signal:
    """A single forecast the platform can report and (paper-)trade on."""
    timestamp: pd.Timestamp
    symbol: str
    strategy: str
    direction: int           # +1 long, -1 short, 0 flat
    strength: float          # model conviction in [0, 1]
    reason: str              # human-readable why, for the forecast report
    features: dict = field(default_factory=dict)


class BaseStrategy(ABC):
    meta: StrategyMeta
    default_params: dict = {}
    param_grid: dict = {}    # {param: [values]} for optimiser

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.Series:
        """
        Return a target-position Series aligned to df.index, values in [-1, 1].
        MUST be causal: position at time t may only use information up to t.
        The backtester applies a 1-bar execution lag on top of this.
        """
        ...

    def latest_signal(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Turn the most recent bar into a reportable Signal with reasoning."""
        pos = self.generate(df)
        ts = df.index[-1]
        d = pos.iloc[-1]
        direction = int(round(d and d / abs(d))) if d != 0 else 0
        return Signal(
            timestamp=ts,
            symbol=symbol,
            strategy=self.meta.name,
            direction=direction,
            strength=min(abs(float(d)), 1.0),
            reason=self._explain(df),
            features=self._features(df),
        )

    # strategies override these two to power the forecast report
    def _explain(self, df: pd.DataFrame) -> str:
        return f"{self.meta.name}: position derived from latest bar."

    def _features(self, df: pd.DataFrame) -> dict:
        return {}

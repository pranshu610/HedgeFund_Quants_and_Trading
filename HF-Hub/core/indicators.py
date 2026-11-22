"""
core/indicators.py
------------------
Vectorised, dependency-light technical indicators. Every function takes/returns
pandas objects and does NOT peek into the future (all rolling windows are
trailing). Keeping these correct is the single most important defence against
lookahead bias in the whole platform.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range — the workhorse for volatility-scaled stops/sizing."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return mid, mid + k * sd, mid - k * sd


def donchian(df: pd.DataFrame, n: int = 20):
    """Upper/lower channel used for breakout entries."""
    upper = df["high"].rolling(n, min_periods=n).max()
    lower = df["low"].rolling(n, min_periods=n).min()
    return upper, lower


def realized_vol(close: pd.Series, n: int = 20, annualize: int = 252) -> pd.Series:
    return close.pct_change().rolling(n, min_periods=n).std() * np.sqrt(annualize)


def zscore(s: pd.Series, n: int = 20) -> pd.Series:
    mean = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std()
    return (s - mean) / sd


def rolling_slope(s: pd.Series, n: int = 20) -> pd.Series:
    """
    Slope of a least-squares line over a trailing window — a cheap, robust
    proxy for a 'trendline' without hand-drawn geometry.
    """
    x = np.arange(n)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def _slope(win):
        y = win
        return ((x - x_mean) * (y - y.mean())).sum() / denom

    return s.rolling(n, min_periods=n).apply(_slope, raw=True)

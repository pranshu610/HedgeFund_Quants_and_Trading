"""
risk/portfolio.py
-----------------
Allocate capital ACROSS strategies (the 'how many trades per strategy / how much
exposure to each' question). Given each strategy's backtested return stream, we
weight them by inverse-volatility (risk parity) with an optional correlation
haircut, then cap any single strategy's share. This is how a multi-strat book
decides sleeve sizes: equalise risk contribution, don't equalise dollars.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def inverse_vol_weights(returns: pd.DataFrame, cap: float = 0.4) -> pd.Series:
    """Risk-parity-lite: weight ∝ 1/vol, capped, renormalised."""
    vol = returns.std().replace(0, np.nan)
    w = (1 / vol).fillna(0.0)
    if w.sum() == 0:
        return pd.Series(1 / len(returns.columns), index=returns.columns)
    w = w / w.sum()
    w = w.clip(upper=cap)
    return w / w.sum()


def correlation_adjusted_weights(returns: pd.DataFrame, cap: float = 0.4) -> pd.Series:
    """
    Down-weight strategies that are highly correlated with the rest of the book —
    a crude but effective diversification tilt. Uses inverse (avg |corr|) as an
    extra multiplier on inverse-vol weights.
    """
    base = inverse_vol_weights(returns, cap=1.0)
    corr = returns.corr().abs()
    cm = corr.to_numpy(copy=True)
    np.fill_diagonal(cm, 0.0)
    corr = pd.DataFrame(cm, index=corr.index, columns=corr.columns)
    avg_corr = corr.mean().replace(0, 0.01)
    tilt = (1 / avg_corr)
    w = base * tilt
    w = w / w.sum()
    w = w.clip(upper=cap)
    return w / w.sum()


def combine_book(strategy_returns: dict[str, pd.Series], method: str = "corr_adj",
                 cap: float = 0.4) -> tuple[pd.Series, pd.Series]:
    """
    strategy_returns: {name -> daily return series}
    Returns (weights, blended_book_returns).
    """
    rets = pd.DataFrame(strategy_returns).dropna(how="all").fillna(0.0)
    w = correlation_adjusted_weights(rets, cap) if method == "corr_adj" else inverse_vol_weights(rets, cap)
    book = (rets * w).sum(axis=1)
    return w, book

"""
backtest/metrics.py
-------------------
Standard performance / risk statistics computed from an equity curve or a
returns series. These are the numbers a PM actually reads.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    sd = excess.std()
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0].std()
    return float(excess.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else 0.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def avg_recovery_days(equity: pd.Series) -> float:
    """Mean number of bars from a new peak-trough to reclaiming the prior peak."""
    peak = equity.cummax()
    in_dd = equity < peak
    if not in_dd.any():
        return 0.0
    runs, count = [], 0
    for flag in in_dd:
        if flag:
            count += 1
        elif count:
            runs.append(count); count = 0
    if count:
        runs.append(count)
    return float(np.mean(runs)) if runs else 0.0


def win_rate(trade_returns: pd.Series) -> float:
    if len(trade_returns) == 0:
        return 0.0
    return float((trade_returns > 0).mean())


def summary(equity: pd.Series, returns: pd.Series, trade_returns: pd.Series | None = None,
            position: pd.Series | None = None) -> dict:
    mdd = max_drawdown(equity)
    cg = cagr(equity)
    out = {
        "CAGR": cg,
        "Sharpe": sharpe(returns),
        "Sortino": sortino(returns),
        "Calmar": float(cg / abs(mdd)) if mdd < 0 else 0.0,
        "MaxDrawdown": mdd,
        "AvgRecoveryBars": avg_recovery_days(equity),
        "Vol(ann)": float(returns.std() * np.sqrt(TRADING_DAYS)),
        "TotalReturn": float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0,
    }
    if position is not None and len(position):
        out["Exposure"] = float((position != 0).mean())        # fraction of time in market
    if trade_returns is not None and len(trade_returns):
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        out["WinRate"] = win_rate(trade_returns)
        out["NumTrades"] = int(len(trade_returns))
        out["AvgWin"] = float(wins.mean()) if len(wins) else 0.0
        out["AvgLoss"] = float(losses.mean()) if len(losses) else 0.0
        out["ProfitFactor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf
    return out

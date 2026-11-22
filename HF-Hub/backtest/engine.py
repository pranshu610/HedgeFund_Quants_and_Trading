"""
backtest/engine.py
------------------
Single-instrument backtester for target-position strategies. Deliberately
conservative and honest:

  * 1-bar execution lag  -> position from signal at t is TRADED at t+1's close,
    so today's bar can never be acted on before it exists (no lookahead).
  * transaction costs    -> commission + slippage charged on TURNOVER
    (abs change in position), in basis points.
  * volatility targeting hook -> the risk engine can pass a size multiplier.

Returns a BacktestResult with the equity curve, daily returns, per-trade
returns, and the position series, ready for metrics + Monte Carlo.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from backtest.metrics import summary


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    position: pd.Series
    trade_returns: pd.Series
    stats: dict
    df: pd.DataFrame


def run_backtest(
    df: pd.DataFrame,
    target_pos: pd.Series,
    initial_capital: float = 100_000,
    cost_bps: float = 1.0,          # commission per side, basis points of notional
    slippage_bps: float = 2.0,      # slippage per side, basis points
    size: pd.Series | float = 1.0,  # risk-engine size multiplier in [0,1+]
    spread_bps: float = 0.0,        # bid/ask half-spread paid per side, bps
    financing_bps_annual: float = 0.0,  # overnight borrow/margin cost on exposure, bps/yr
    periods_per_year: int = 252,
) -> BacktestResult:
    df = df.sort_index()
    px = df["close"]
    ret = px.pct_change().fillna(0.0)

    # apply size scaling then the crucial 1-bar execution lag
    pos = (target_pos * size).clip(-3, 3)
    exec_pos = pos.shift(1).fillna(0.0)

    # per-trade costs on turnover: commission + slippage + bid/ask spread (all per side)
    turnover = exec_pos.diff().abs().fillna(exec_pos.abs())
    cost = turnover * (cost_bps + slippage_bps + spread_bps) / 1e4

    # overnight financing / borrow: charged on held exposure every bar (annual -> per bar)
    financing = exec_pos.abs() * (financing_bps_annual / 1e4) / periods_per_year

    strat_ret = exec_pos * ret - cost - financing
    equity = initial_capital * (1 + strat_ret).cumprod()

    # per-trade returns: segment by sign changes of executed position
    trade_returns = _trade_returns(exec_pos, ret, cost + financing)

    stats = summary(equity, strat_ret, trade_returns, position=exec_pos)
    stats["FinalEquity"] = float(equity.iloc[-1])
    stats["Turnover(ann)"] = float(turnover.sum() / max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9))
    return BacktestResult(equity, strat_ret, exec_pos, trade_returns, stats, df)


def _trade_returns(pos: pd.Series, ret: pd.Series, cost: pd.Series) -> pd.Series:
    """Compound bar returns within each contiguous same-sign holding period."""
    trades = []
    cur, start_val = 0.0, 1.0
    for i in range(len(pos)):
        p = pos.iloc[i]
        bar = p * ret.iloc[i] - cost.iloc[i]
        if p != cur:
            if cur != 0.0:
                trades.append(start_val - 1.0)
            start_val = 1.0
            cur = p
        if cur != 0.0:
            start_val *= (1 + bar)
    if cur != 0.0:
        trades.append(start_val - 1.0)
    return pd.Series(trades, dtype=float)

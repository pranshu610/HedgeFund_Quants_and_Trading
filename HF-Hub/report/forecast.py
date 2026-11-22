"""
report/forecast.py
------------------
Produces the per-trade forecast report: for a given strategy + latest bars, it
returns (and pretty-prints) the signal, the reasoning, the risk-managed trade
plan, and a robustness read. This is the "what is the forecast and why, per
trade" deliverable — one object your dashboard, CLI, or a PDF exporter can render.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from core.indicators import atr, realized_vol
from risk.sizing import build_trade_plan
from backtest.engine import run_backtest
from montecarlo.simulator import bootstrap_strategy


@dataclass
class TradeForecast:
    symbol: str
    strategy: str
    direction: str
    conviction: float
    entry: float
    stop: float
    target: float
    rr: float
    size_frac: float
    sizing_method: str
    horizon: str
    reasoning: str
    risk_notes: str
    robustness: str
    prob_loss: float


def forecast_trade(strategy, df: pd.DataFrame, symbol: str, capital: float = 100_000,
                   sizing_method: str = "vol_target") -> TradeForecast:
    sig = strategy.latest_signal(df, symbol)
    entry = float(df["close"].iloc[-1])
    a = float(atr(df, 14).iloc[-1])
    rv = float(realized_vol(df["close"], 20).iloc[-1])

    # win rate from this strategy's own backtest, for Kelly sizing
    bt = run_backtest(df, strategy.generate(df))
    wr = bt.stats.get("WinRate", 0.5)

    direction = sig.direction if sig.direction != 0 else 1
    plan = build_trade_plan(capital, entry, a, direction, rv,
                            method=sizing_method, win_rate=wr)

    # robustness on the strategy's realised returns
    try:
        rob = bootstrap_strategy(bt.returns, initial_value=capital, n_sims=2000, seed=1)
        rob_txt, prob_loss = rob.verdict, rob.prob_loss
    except Exception:
        rob_txt, prob_loss = "insufficient history for bootstrap", float("nan")

    dir_word = "FLAT / no trade" if sig.direction == 0 else ("LONG" if sig.direction > 0 else "SHORT")
    return TradeForecast(
        symbol=symbol, strategy=sig.strategy, direction=dir_word,
        conviction=round(sig.strength, 2), entry=round(entry, 2),
        stop=plan.stop_price, target=plan.target_price, rr=plan.rr,
        size_frac=plan.size_frac, sizing_method=plan.method,
        horizon=strategy.meta.horizon, reasoning=sig.reason,
        risk_notes=plan.notes, robustness=rob_txt, prob_loss=round(prob_loss, 3),
    )


def render(fc: TradeForecast) -> str:
    L = [
        f"┌─ TRADE FORECAST · {fc.symbol} · {fc.strategy}",
        f"│  Direction    : {fc.direction}   (conviction {fc.conviction})",
        f"│  Horizon      : {fc.horizon}",
        f"│  Entry        : {fc.entry}",
        f"│  Stop / Target: {fc.stop} / {fc.target}   (R:R {fc.rr})",
        f"│  Size         : {fc.size_frac:.2%} of sleeve  [{fc.sizing_method}]",
        f"│  Why          : {fc.reasoning}",
        f"│  Risk         : {fc.risk_notes}",
        f"│  Robustness   : {fc.robustness}",
        f"│  P(loss) MC   : {fc.prob_loss}",
        "└" + "─" * 60,
    ]
    return "\n".join(L)

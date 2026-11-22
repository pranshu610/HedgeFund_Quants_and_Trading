"""
montecarlo/simulator.py
-----------------------
Two complementary Monte Carlo tools:

1. run_simulation / run_multi_asset_simulation  — your original parametric GBM
   engine (unchanged in spirit): forward-simulate a portfolio's VALUE given
   assumed drift/vol, for VaR/CVaR/ruin. Good for portfolio-level what-ifs.

2. bootstrap_strategy  — the STRATEGY-robustness test that answers prompt #9
   ("is this strategy robust or fragile?"). It resamples the strategy's realised
   per-bar returns (block bootstrap to preserve autocorrelation) thousands of
   times to build a distribution of terminal equity, max drawdown, and the
   probability of ending in a loss. This is more honest than GBM for judging a
   backtested strategy, because it uses the strategy's actual return shape rather
   than an assumed normal one.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# 1) Parametric GBM engine (your original, condensed) — portfolio value MC
# ----------------------------------------------------------------------------
@dataclass
class SimulationResult:
    paths: np.ndarray
    final_values: np.ndarray
    var_95: float; var_99: float
    cvar_95: float; cvar_99: float
    prob_ruin: float; prob_profit: float
    expected_return: float; median_return: float
    best_case: float; worst_case: float; sharpe: float
    initial_value: float; n_simulations: int; n_days: int
    annual_return: float; annual_vol: float; ruin_threshold: float


def run_simulation(initial_value=100_000, annual_return=0.08, annual_vol=0.15,
                   n_simulations=10_000, n_days=252, ruin_threshold=0.20, seed=None):
    rng = np.random.default_rng(seed)
    daily_ret = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    drift = daily_ret - 0.5 * daily_vol**2                      # Ito correction
    shocks = rng.standard_normal((n_simulations, n_days))
    log_ret = drift + daily_vol * shocks
    price_rel = np.exp(np.cumsum(log_ret, axis=1))
    paths = np.hstack([np.ones((n_simulations, 1)), price_rel]) * initial_value
    final = paths[:, -1]

    var_95 = initial_value - np.percentile(final, 5)
    var_99 = initial_value - np.percentile(final, 1)
    t95 = final[final <= np.percentile(final, 5)]
    t99 = final[final <= np.percentile(final, 1)]
    cvar_95 = initial_value - t95.mean() if len(t95) else var_95
    cvar_99 = initial_value - t99.mean() if len(t99) else var_99
    ruin_level = initial_value * (1 - ruin_threshold)
    std_daily = log_ret.std()
    sharpe = (log_ret.mean() * 252) / (std_daily * np.sqrt(252)) if std_daily > 0 else 0.0

    return SimulationResult(
        paths=paths, final_values=final, var_95=var_95, var_99=var_99,
        cvar_95=cvar_95, cvar_99=cvar_99,
        prob_ruin=(final < ruin_level).mean(), prob_profit=(final > initial_value).mean(),
        expected_return=final.mean(), median_return=np.median(final),
        best_case=np.percentile(final, 95), worst_case=np.percentile(final, 5),
        sharpe=sharpe, initial_value=initial_value, n_simulations=n_simulations,
        n_days=n_days, annual_return=annual_return, annual_vol=annual_vol,
        ruin_threshold=ruin_threshold,
    )


# ----------------------------------------------------------------------------
# 2) Block-bootstrap robustness test on a STRATEGY's realised returns
# ----------------------------------------------------------------------------
@dataclass
class RobustnessResult:
    prob_loss: float
    p05_terminal: float; p50_terminal: float; p95_terminal: float
    median_maxdd: float; p95_maxdd: float
    verdict: str
    terminal_dist: np.ndarray
    maxdd_dist: np.ndarray


def bootstrap_strategy(returns: pd.Series, initial_value: float = 100_000,
                       n_sims: int = 5_000, block: int = 10, horizon: int | None = None,
                       seed: int | None = None) -> RobustnessResult:
    """
    Block bootstrap of realised per-bar returns. `block` preserves short-run
    autocorrelation (momentum/mean-reversion) that an i.i.d. bootstrap destroys.
    """
    r = returns.dropna().values
    if len(r) < block * 2:
        raise ValueError("Not enough return history to bootstrap.")
    rng = np.random.default_rng(seed)
    H = horizon or len(r)
    n_blocks = int(np.ceil(H / block))

    terminals = np.empty(n_sims)
    maxdds = np.empty(n_sims)
    starts_pool = len(r) - block
    for s in range(n_sims):
        starts = rng.integers(0, starts_pool, n_blocks)
        sample = np.concatenate([r[i:i + block] for i in starts])[:H]
        equity = initial_value * np.cumprod(1 + sample)
        terminals[s] = equity[-1]
        peak = np.maximum.accumulate(equity)
        maxdds[s] = ((equity - peak) / peak).min()

    prob_loss = float((terminals < initial_value).mean())
    median_maxdd = float(np.median(maxdds))
    p95_maxdd = float(np.percentile(maxdds, 5))  # 5th pct = worst 5% drawdown

    # simple robustness verdict
    if prob_loss < 0.35 and median_maxdd > -0.20:
        verdict = "ROBUST — positive skew survives resampling, drawdowns contained."
    elif prob_loss < 0.5:
        verdict = "FRAGILE-ish — edge present but drawdown/loss risk is material; size down."
    else:
        verdict = "FRAGILE — resampled paths lose more often than not. Do not deploy."

    return RobustnessResult(
        prob_loss=prob_loss,
        p05_terminal=float(np.percentile(terminals, 5)),
        p50_terminal=float(np.percentile(terminals, 50)),
        p95_terminal=float(np.percentile(terminals, 95)),
        median_maxdd=median_maxdd, p95_maxdd=p95_maxdd,
        verdict=verdict, terminal_dist=terminals, maxdd_dist=maxdds,
    )

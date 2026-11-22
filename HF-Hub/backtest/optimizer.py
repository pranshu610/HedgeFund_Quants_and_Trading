"""
backtest/optimizer.py
---------------------
Walk-forward parameter search. Grid-searches a strategy's `param_grid`, but
scores each config on OUT-OF-SAMPLE walk-forward windows, not on the same data
it was fitted on. This is the single most important guard against the overfit
backtests that look great in-sample and fail live.

For each split: optimise on the in-sample window, then record the config's
performance on the next (unseen) out-of-sample window. The recommended config is
the one with the best average out-of-sample objective across all folds — which
is a far more honest number than a single in-sample Sharpe.
"""
from __future__ import annotations
from itertools import product
import numpy as np
import pandas as pd
from backtest.engine import run_backtest


def _objective(stats: dict, metric: str) -> float:
    v = stats.get(metric, 0.0)
    return -1e9 if (v is None or np.isnan(v) or np.isinf(v)) else float(v)


def grid_configs(param_grid: dict) -> list[dict]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, combo)) for combo in product(*[param_grid[k] for k in keys])]


def walk_forward(
    strategy_cls,
    df: pd.DataFrame,
    n_splits: int = 4,
    metric: str = "Sharpe",
    cost_bps: float = 1.0,
    slippage_bps: float = 2.0,
    fixed_params: dict | None = None,
    max_configs: int = 60,
) -> dict:
    """`max_configs` caps the grid search. Searching 300+ combinations on a few
    hundred bars does not find a better strategy — it finds the combination that
    best fits the noise, and it makes the UI feel broken. When the grid is larger
    than the cap we take a deterministic evenly-spaced subsample, which preserves
    coverage of the parameter space without the combinatorial blow-up."""
    fixed_params = fixed_params or {}
    configs = grid_configs(strategy_cls.param_grid)
    if max_configs and len(configs) > max_configs:
        step = len(configs) / max_configs
        configs = [configs[int(i * step)] for i in range(max_configs)]
    folds = np.array_split(np.arange(len(df)), n_splits + 1)

    oos_scores = {i: [] for i in range(len(configs))}
    for k in range(n_splits):
        is_idx = folds[k]
        oos_idx = folds[k + 1]
        is_df, oos_df = df.iloc[is_idx], df.iloc[oos_idx]

        # pick best config in-sample this fold, then evaluate it out-of-sample
        best_ci, best_is = None, -np.inf
        for ci, cfg in enumerate(configs):
            strat = strategy_cls(**{**fixed_params, **cfg})
            r = run_backtest(is_df, strat.generate(is_df), cost_bps=cost_bps, slippage_bps=slippage_bps)
            score = _objective(r.stats, metric)
            if score > best_is:
                best_is, best_ci = score, ci
        strat = strategy_cls(**{**fixed_params, **configs[best_ci]})
        roos = run_backtest(oos_df, strat.generate(oos_df), cost_bps=cost_bps, slippage_bps=slippage_bps)
        oos_scores[best_ci].append(_objective(roos.stats, metric))

    # recommend the config with the best average realised OOS score
    ranked = sorted(
        [(ci, np.mean(s)) for ci, s in oos_scores.items() if s],
        key=lambda x: x[1], reverse=True,
    )
    best_ci = ranked[0][0] if ranked else 0
    best_cfg = {**fixed_params, **configs[best_ci]}

    # final full-sample backtest with the recommended config (for reporting)
    final = run_backtest(df, strategy_cls(**best_cfg).generate(df),
                         cost_bps=cost_bps, slippage_bps=slippage_bps)
    return {
        "recommended_params": best_cfg,
        "oos_score": ranked[0][1] if ranked else 0.0,
        "metric": metric,
        "full_sample_stats": final.stats,
        "n_configs_tested": len(configs),
    }

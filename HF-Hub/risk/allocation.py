"""
risk/allocation.py
------------------
The 'mathematically optimal, risk-weighted' capital allocator. Given a matrix of
strategy (or asset) return streams, it produces target weights under several
objectives. The honest headline: EVERY closed-form 'optimal' portfolio is only
optimal in-sample. Mean-variance in particular is notoriously fragile — it
levers up on the noisiest estimates. So this module ships the robust methods
(shrinkage covariance, Hierarchical Risk Parity) as the DEFAULTS, and treats raw
max-Sharpe as a reference point, not the recommendation.

Methods
  min_variance      : lowest-vol long-only portfolio (most robust MV variant)
  max_sharpe        : tangency portfolio (highest in-sample Sharpe; use w/ caution)
  risk_parity       : equal risk contribution (no return estimates needed)
  hrp               : Hierarchical Risk Parity (López de Prado) — clusters, then
                      allocates; robust to estimation error, no matrix inversion
  kelly             : growth-optimal weights from mean/cov, fractional-scaled

Covariance is Ledoit-Wolf shrunk by default because the sample covariance of
~16 strategies over a few years of data is not invertible-with-a-straight-face.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

TRADING_DAYS = 252


# ---------------------------------------------------------------- covariance
def shrink_cov(returns: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage toward a scaled identity. Falls back gracefully."""
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(returns.values).covariance_ * TRADING_DAYS
    except Exception:
        S = returns.cov().values * TRADING_DAYS
        mu = np.trace(S) / len(S)
        shrink = 0.1
        return (1 - shrink) * S + shrink * mu * np.eye(len(S))


def _ann_mean(returns: pd.DataFrame) -> np.ndarray:
    return returns.mean().values * TRADING_DAYS


# ---------------------------------------------------------------- optimisers
def min_variance(returns: pd.DataFrame, cap: float = 0.4) -> pd.Series:
    cov = shrink_cov(returns)
    n = len(cov)
    w0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, cap)] * n
    res = minimize(lambda w: w @ cov @ w, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=returns.columns)


def max_sharpe(returns: pd.DataFrame, rf: float = 0.0, cap: float = 0.4) -> pd.Series:
    cov = shrink_cov(returns)
    mu = _ann_mean(returns) - rf
    n = len(cov)
    w0 = np.repeat(1 / n, n)
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0, cap)] * n

    def neg_sharpe(w):
        vol = np.sqrt(w @ cov @ w)
        return -(w @ mu) / vol if vol > 0 else 0.0

    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons)
    return pd.Series(res.x, index=returns.columns)


def risk_parity(returns: pd.DataFrame) -> pd.Series:
    """Equal risk contribution — no expected-return estimate, so far more stable."""
    cov = shrink_cov(returns)
    n = len(cov)

    def obj(w):
        port_var = w @ cov @ w
        mrc = cov @ w
        rc = w * mrc
        target = port_var / n
        return ((rc - target) ** 2).sum()

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(obj, np.repeat(1 / n, n), method="SLSQP",
                   bounds=[(1e-4, 1)] * n, constraints=cons)
    return pd.Series(res.x / res.x.sum(), index=returns.columns)


def hrp(returns: pd.DataFrame) -> pd.Series:
    """
    Hierarchical Risk Parity. Clusters strategies by correlation, then recursively
    splits capital inversely to cluster variance. No covariance inversion, so it's
    robust exactly where max-Sharpe blows up. This is the DEFAULT recommendation.
    """
    corr = returns.corr().fillna(0.0)
    cm = corr.to_numpy(copy=True)
    np.fill_diagonal(cm, 1.0)
    corr = pd.DataFrame(cm, index=corr.index, columns=corr.columns)
    cov = returns.cov().fillna(0.0)
    dist = np.sqrt(((1 - corr) / 2).clip(lower=0)).fillna(1.0)
    dv = dist.to_numpy(copy=True)
    dv[~np.isfinite(dv)] = 1.0
    np.fill_diagonal(dv, 0.0)
    link = linkage(squareform(dv, checks=False), method="single")
    order = leaves_list(link)
    ordered = returns.columns[order].tolist()

    w = pd.Series(1.0, index=ordered)
    clusters = [ordered]
    while clusters:
        new = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            var_l = _cluster_var(cov, left)
            var_r = _cluster_var(cov, right)
            alpha = 1 - var_l / (var_l + var_r)
            for s in left:
                w[s] *= alpha
            for s in right:
                w[s] *= (1 - alpha)
            new += [left, right]
        clusters = new
    return (w / w.sum()).reindex(returns.columns)


def _cluster_var(cov: pd.DataFrame, items: list) -> float:
    sub = cov.loc[items, items].values
    ivp = 1 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def kelly(returns: pd.DataFrame, fraction: float = 0.33, cap: float = 0.4) -> pd.Series:
    """
    Growth-optimal weights w* = Σ⁻¹ μ, scaled by `fraction` (never full Kelly).
    Uses shrunk covariance; clipped to long-only within cap for sanity.
    """
    cov = shrink_cov(returns)
    mu = _ann_mean(returns)
    try:
        raw = np.linalg.solve(cov, mu)
    except np.linalg.LinAlgError:
        raw = np.linalg.pinv(cov) @ mu
    w = np.clip(raw * fraction, 0, cap)
    if w.sum() == 0:
        return pd.Series(1 / len(mu), index=returns.columns)
    return pd.Series(w / w.sum(), index=returns.columns)


ALLOCATORS = {
    "hrp": hrp, "risk_parity": risk_parity, "min_variance": min_variance,
    "max_sharpe": max_sharpe, "kelly": kelly,
}


def _usable(returns: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns that are finite and actually move (non-zero variance).
    A flat/all-zero strategy return stream makes correlation matrices non-finite
    and blows up HRP/optimisers — drop those before allocating."""
    r = returns.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    good = [c for c in r.columns if r[c].std() > 1e-9]
    return r[good]


def allocate(returns: pd.DataFrame, method: str = "hrp", **kw) -> pd.Series:
    if method not in ALLOCATORS:
        raise KeyError(f"Unknown method '{method}'. Options: {list(ALLOCATORS)}")
    all_cols = list(returns.columns)
    r = _usable(returns)
    # degenerate cases: 0 or 1 usable strategy → equal-weight across all originals
    if r.shape[1] < 2:
        return pd.Series(1.0 / max(len(all_cols), 1), index=all_cols)
    fn = ALLOCATORS[method]
    import inspect
    params = inspect.signature(fn).parameters
    kw = {k: v for k, v in kw.items() if k in params}
    w = fn(r, **kw).clip(lower=0)
    w = (w / w.sum()) if w.sum() > 0 else pd.Series(1.0 / r.shape[1], index=r.columns)
    return w.reindex(all_cols).fillna(0.0)


def compare_allocations(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Run every allocator and report each portfolio's in-sample ann. return, vol,
    and Sharpe — so you can SEE why HRP/risk-parity are recommended over the
    'optimal' max-Sharpe once you account for how concentrated it gets.
    """
    rows = {}
    for name, fn in ALLOCATORS.items():
        try:
            w = fn(returns)
        except Exception as e:
            rows[name] = {"error": str(e)[:40]}
            continue
        port = (returns * w).sum(axis=1)
        vol = port.std() * np.sqrt(TRADING_DAYS)
        ret = port.mean() * TRADING_DAYS
        rows[name] = {
            "ann_return": round(ret, 4),
            "ann_vol": round(vol, 4),
            "sharpe": round(ret / vol, 3) if vol > 0 else 0.0,
            "max_weight": round(float(w.max()), 3),
            "n_effective": round(1 / (w ** 2).sum(), 1),   # diversification
        }
    return pd.DataFrame(rows).T

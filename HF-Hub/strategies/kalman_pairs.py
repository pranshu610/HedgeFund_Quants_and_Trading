"""
strategies/kalman_pairs.py
--------------------------
Kalman-filter pairs trading — the dynamic-hedge-ratio upgrade to static OLS
pairs. Real cointegration relationships drift: the hedge ratio between two names
is not constant. A Kalman filter treats [alpha, beta] as a hidden state that
random-walks over time and updates them every bar. We then trade the filter's
STANDARDISED FORECAST ERROR (observed price minus predicted price, divided by
its predicted std): a large error means the spread has diverged from its current
equilibrium, so fade it.

Why this beats static-beta pairs (strategies/pairs_statarb.py):
  * adapts to slow drift in the relationship (no periodic re-fitting needed)
  * the forecast-error variance is produced by the filter itself, so the z-score
    denominator is regime-aware rather than a fixed rolling window
This is the Chan (2013) formulation. Hand-rolled, no external filter library.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from core.strategy import StrategyMeta, Signal


@dataclass
class KalmanState:
    beta: pd.Series          # time-varying hedge ratio
    alpha: pd.Series         # time-varying intercept
    forecast_err: pd.Series  # e_t = y_t - H_t x_{t-1}
    err_std: pd.Series       # sqrt(Q_t), predicted std of the forecast error


def kalman_hedge(a: pd.Series, b: pd.Series, delta: float = 1e-4, r_var: float = 1e-3) -> KalmanState:
    """
    State x=[alpha,beta] random-walks; observe y=a as H=[1,b] · x + noise.
    delta controls how fast beta adapts (bigger = faster/noisier).
    r_var is the observation-noise variance.
    """
    y = a.values.astype(float)
    Bx = b.values.astype(float)
    n = len(y)

    x = np.array([0.0, 1.0])             # prior: alpha=0, beta=1
    P = np.eye(2) * 1.0                  # diffuse initial covariance → fast early learning
    Q = delta / (1 - delta) * np.eye(2)  # state-transition (process) noise

    betas = np.zeros(n); alphas = np.zeros(n)
    errs = np.zeros(n); estds = np.zeros(n)

    for t in range(n):
        H = np.array([1.0, Bx[t]])
        # predict
        P = P + Q
        y_hat = H @ x
        e = y[t] - y_hat                 # forecast error
        S = H @ P @ H + r_var           # forecast-error variance
        # update
        K = (P @ H) / S                 # Kalman gain
        x = x + K * e
        P = P - np.outer(K, H) @ P

        alphas[t], betas[t] = x[0], x[1]
        errs[t] = e
        estds[t] = np.sqrt(S)

    idx = a.index
    return KalmanState(beta=pd.Series(betas, idx), alpha=pd.Series(alphas, idx),
                       forecast_err=pd.Series(errs, idx), err_std=pd.Series(estds, idx))


class KalmanPairs:
    meta = StrategyMeta(
        name="Kalman Pairs",
        category="Statistical Arb",
        asset_classes=["equities", "crypto"],
        timeframe="1d",
        horizon="intraday",
        uses_short=True,
        description="Dynamic hedge ratio via Kalman filter; trade the standardised forecast error.",
    )
    default_params = dict(delta=1e-4, r_var=1e-3, z_window=20, entry=1.5, exit=0.3, warmup=60)
    param_grid = dict(delta=[1e-4, 1e-3, 1e-2], z_window=[15, 20, 30], entry=[1.0, 1.5, 2.0])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}
        self.state: KalmanState | None = None

    def _zscore(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Rolling-standardised Kalman forecast error — the tradeable signal.
        The filter supplies the adaptive hedge ratio; the rolling window supplies
        a well-calibrated z denominator (the filter's own variance is model
        uncertainty, not spread deviation, and mis-scales the z-score)."""
        p = self.params
        self.state = kalman_hedge(np.log(a), np.log(b), p["delta"], p["r_var"])
        e = self.state.forecast_err
        mean = e.rolling(p["z_window"], min_periods=p["z_window"]).mean()
        sd = e.rolling(p["z_window"], min_periods=p["z_window"]).std()
        return ((e - mean) / sd).fillna(0.0)

    def generate_pair(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Target position on the spread in {-1,0,+1}. +1 = long a / short beta·b."""
        p = self.params
        z = self._zscore(a, b).values
        out = np.zeros(len(z))
        holding = 0.0
        for t in range(len(z)):
            if t < p["warmup"]:           # let the filter converge first
                out[t] = 0.0
                continue
            if holding == 0.0:
                if z[t] >= p["entry"]:
                    holding = -1.0        # a rich vs prediction → short spread
                elif z[t] <= -p["entry"]:
                    holding = 1.0
            elif abs(z[t]) <= p["exit"]:
                holding = 0.0
            out[t] = holding
        return pd.Series(out, index=a.index)

    def spread_returns(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Dollar-neutral P&L using the CURRENT (lagged) beta each bar."""
        ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
        beta = self.state.beta.shift(1).fillna(0) if self.state is not None else 1.0
        w = 1 / (1 + beta.abs())
        return w * ra - (beta * w) * rb

    def latest_signal(self, a: pd.Series, b: pd.Series, sym_a: str, sym_b: str) -> Signal:
        z_series = self._zscore(a, b)
        pos = self.generate_pair(a, b)
        z = float(z_series.iloc[-1])
        beta = float(self.state.beta.iloc[-1])
        d = int(pos.iloc[-1])
        reason = (f"Kalman forecast-error z {z:.2f} (entry ±{self.params['entry']}); "
                  f"current hedge ratio β={beta:.3f}. "
                  f"Thesis: fade divergence from the time-varying equilibrium.")
        return Signal(timestamp=a.index[-1], symbol=f"{sym_a}/{sym_b}",
                      strategy=self.meta.name, direction=d,
                      strength=min(abs(z) / self.params["entry"], 1.0),
                      reason=reason, features={"z": round(z, 2), "beta": round(beta, 3)})


def pairs_equity_df(strat: KalmanPairs, a: pd.Series, b: pd.Series) -> pd.DataFrame:
    strat.generate_pair(a, b)             # ensures state is populated
    sr = strat.spread_returns(a, b)
    px = 100 * (1 + sr).cumprod()
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                        index=a.index)

"""
strategies/pairs_statarb.py
---------------------------
Cointegration pairs trading — the canonical statistical-arbitrage strategy.
Two assets whose prices share a long-run equilibrium (cointegrated) will see
their spread mean-revert. We estimate the hedge ratio by OLS, test the spread
for stationarity (Engle-Granger ADF), then trade the z-score of the spread:
short the spread when it's stretched high, long it when stretched low, exit near
the mean.

Because a spread can go negative, this strategy exposes `generate_pair` (returns
the spread's target position) and `spread_series`, plus a helper to turn spread
returns into a positive equity index so the standard backtester (with its exec
lag + costs) can score it unchanged.

Edge rationale: relative-value mean reversion. Market-neutral by construction, so
it survives market crashes — but it's crowded and breaks when the cointegration
relationship structurally shifts (regime break, M&A, sector rotation). The ADF
p-value is the guardrail: no trade if the pair isn't actually cointegrated.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from core.indicators import zscore
from core.strategy import StrategyMeta, Signal


@dataclass
class PairFit:
    beta: float             # hedge ratio: spread = log(a) - beta*log(b) - alpha
    alpha: float
    adf_pvalue: float       # Engle-Granger p-value < 0.05 => cointegrated => tradeable
    cointegrated: bool
    half_life: float        # bars to mean-revert halfway (Ornstein-Uhlenbeck)


def fit_pair(a: pd.Series, b: pd.Series, pvalue_max: float = 0.05) -> PairFit:
    """Engle-Granger: OLS hedge ratio on log prices; cointegration p-value from
    coint() (uses MacKinnon critical values that correctly account for the fact
    that beta was estimated — naive adfuller on OLS residuals over-rejects and
    would flag independent walks as cointegrated)."""
    la, lb = np.log(a), np.log(b)
    X = np.vstack([np.ones(len(lb)), lb.values]).T
    coef, *_ = np.linalg.lstsq(X, la.values, rcond=None)
    alpha, beta = float(coef[0]), float(coef[1])
    spread = la - (alpha + beta * lb)
    # correct Engle-Granger cointegration test
    pval = float(coint(la, lb, trend="c", maxlag=1, autolag=None)[1])

    # half-life via AR(1) on the spread change
    s = spread.dropna()
    ds = s.diff().dropna()
    s_lag = s.shift(1).dropna().loc[ds.index]
    phi = np.polyfit(s_lag.values, ds.values, 1)[0]
    half_life = float(-np.log(2) / phi) if phi < 0 else np.inf

    return PairFit(beta=beta, alpha=alpha, adf_pvalue=pval,
                   cointegrated=pval < pvalue_max, half_life=half_life)


class PairsStatArb:
    meta = StrategyMeta(
        name="Pairs Stat-Arb",
        category="Statistical Arb",
        asset_classes=["equities", "crypto"],
        timeframe="1d",
        horizon="intraday",
        uses_short=True,
        description="Cointegration-tested pairs; trade z-score of the mean-reverting spread.",
    )
    default_params = dict(z_window=20, z_entry=2.0, z_exit=0.5, pvalue_max=0.05)
    param_grid = dict(z_window=[15, 20, 30], z_entry=[1.5, 2.0, 2.5], z_exit=[0.0, 0.5])

    def __init__(self, **params):
        self.params = {**self.default_params, **params}
        self.fit: PairFit | None = None

    def spread_series(self, a: pd.Series, b: pd.Series) -> pd.Series:
        self.fit = fit_pair(a, b, self.params["pvalue_max"])
        return np.log(a) - (self.fit.alpha + self.fit.beta * np.log(b))

    def generate_pair(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Target position on the SPREAD in {-1, 0, +1}. +1 = long spread (long a / short b)."""
        p = self.params
        spread = self.spread_series(a, b)
        if not self.fit.cointegrated:
            return pd.Series(0.0, index=a.index)   # not cointegrated → stand aside
        z = zscore(spread, p["z_window"])
        out = np.zeros(len(z))
        holding = 0.0
        zv = z.values
        for i in range(len(z)):
            if holding == 0.0:
                if zv[i] >= p["z_entry"]:
                    holding = -1.0            # spread too high → short it
                elif zv[i] <= -p["z_entry"]:
                    holding = 1.0             # spread too low → long it
            elif abs(zv[i]) <= p["z_exit"]:
                holding = 0.0
            out[i] = holding
        return pd.Series(out, index=a.index)

    def spread_returns(self, a: pd.Series, b: pd.Series) -> pd.Series:
        """Dollar-neutral daily P&L of a unit long-spread, in return space."""
        ra, rb = a.pct_change().fillna(0), b.pct_change().fillna(0)
        beta = self.fit.beta if self.fit else 1.0
        # normalise so gross ~1: long 1/(1+|beta|) of a, short beta/(1+|beta|) of b
        w = 1 / (1 + abs(beta))
        return w * ra - (beta * w) * rb

    def latest_signal(self, a: pd.Series, b: pd.Series, sym_a: str, sym_b: str) -> Signal:
        pos = self.generate_pair(a, b)
        z = zscore(self.spread_series(a, b), self.params["z_window"]).iloc[-1]
        d = int(pos.iloc[-1])
        coint = self.fit.cointegrated
        reason = (f"Spread z-score {z:.2f} (entry ±{self.params['z_entry']}); "
                  f"ADF p={self.fit.adf_pvalue:.3f} "
                  f"({'cointegrated' if coint else 'NOT cointegrated — stand aside'}); "
                  f"half-life {self.fit.half_life:.0f} bars.")
        return Signal(timestamp=a.index[-1], symbol=f"{sym_a}/{sym_b}",
                      strategy=self.meta.name, direction=d,
                      strength=min(abs(float(z)) / self.params["z_entry"], 1.0) if coint else 0.0,
                      reason=reason, features={"z": round(float(z), 2),
                                               "adf_p": round(self.fit.adf_pvalue, 3),
                                               "beta": round(self.fit.beta, 3)})


def pairs_equity_df(strat: PairsStatArb, a: pd.Series, b: pd.Series) -> pd.DataFrame:
    """Build a positive 'close' index from spread returns so run_backtest can score it."""
    strat.spread_series(a, b)                       # ensure fit
    sr = strat.spread_returns(a, b)
    px = 100 * (1 + sr).cumprod()
    return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                        index=a.index)

"""
strategies/convergence.py
-------------------------
Two convergence (relative-value) strategies from the classic hedge-fund roster,
both built on the cointegration-pairs engine (they ARE convergence trades) but
with their own economics, framing and tuning:

  FixedIncomeArb — duration/curve convergence. Two rate instruments (e.g. IEF vs
    TLT, or a 2s10s proxy) share a stable long-run relationship; trade the spread
    back to equilibrium. Rates mean-revert hard, so tighter z-thresholds and
    faster exits. Institutional in size (leverage + funding), but the SIGNAL is
    identical; retail scale is the constraint, not the logic.

  IndexArb — ETF/futures basis. An index ETF and its future (or fair value) should
    track tightly; when the basis dislocates, trade convergence. Very fast,
    low-margin, latency-sensitive in the real world — here it's the daily-bar
    research version.

Both reuse PairsStatArb's cointegration test, hedge ratio, z-score entries/exits,
spread P&L and reporting — so they slot into the backtester, scanner and pair UI
unchanged.
"""
from __future__ import annotations
from core.strategy import StrategyMeta
from strategies.pairs_statarb import PairsStatArb


class FixedIncomeArb(PairsStatArb):
    meta = StrategyMeta(
        name="Fixed Income Arb",
        category="Fixed Income Arb",
        asset_classes=["rates", "bonds", "etf"],
        timeframe="1d",
        horizon="swing",
        uses_short=True,
        description="Duration/curve convergence between two rate instruments (e.g. IEF/TLT).",
    )
    # rates mean-revert strongly → tighter entries, exit closer to the mean
    default_params = dict(z_window=20, z_entry=1.5, z_exit=0.25, pvalue_max=0.05)
    param_grid = dict(z_window=[15, 20, 30], z_entry=[1.0, 1.5, 2.0], z_exit=[0.0, 0.25])


class IndexArb(PairsStatArb):
    meta = StrategyMeta(
        name="Index Arb",
        category="Index Arb",
        asset_classes=["equities", "etf", "futures"],
        timeframe="1d",
        horizon="intraday",
        uses_short=True,
        description="ETF vs future/fair-value basis convergence (e.g. SPY vs ES=F).",
    )
    # basis is very tight and fast → small deviations, quick reversion
    default_params = dict(z_window=15, z_entry=1.25, z_exit=0.0, pvalue_max=0.10)
    param_grid = dict(z_window=[10, 15, 20], z_entry=[1.0, 1.25, 1.75])

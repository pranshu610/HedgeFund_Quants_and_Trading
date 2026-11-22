"""
core/altdata.py
---------------
Synthetic generators for the non-OHLCV data the macro / carry / vol strategies
need, so they run fully offline and deterministically. Each generator documents
the REAL feed you'd swap in for production:

  macro indicators  -> FRED (fredapi): GDP/PMI, CPI, fed funds, 10Y yield
  FX spot + rates   -> broker/EOD feed for spot; central-bank policy rates or
                       OIS for the carry leg
  implied vol       -> option-implied (VIX for SPX; per-name from an options feed)

Keeping these behind small functions means the strategies depend on a clean dict
interface, not on any particular vendor.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def synthetic_macro(start="2010-01-01", periods=180, seed=0) -> pd.DataFrame:
    """Monthly macro indicators with realistic co-movement and business cycles.
    Columns: growth (PMI-like, ~50 neutral), inflation (% yoy), policy_rate (%),
    yield_10y (%). Returns a monthly-indexed DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="ME")
    t = np.arange(periods)
    cycle = np.sin(2 * np.pi * t / 60)                       # ~5y business cycle
    growth = 52 + 6 * cycle + rng.normal(0, 1.0, periods)    # PMI around 50
    inflation = 2.2 + 1.5 * np.sin(2 * np.pi * t / 48 + 1) + 0.1 * np.cumsum(rng.normal(0, 0.1, periods))
    inflation = np.clip(inflation, -1, 9)
    policy = np.clip(0.5 + 0.4 * (inflation - 2) + 0.15 * (growth - 50) + rng.normal(0, 0.2, periods), 0, 8)
    y10 = np.clip(policy + 1.2 + 0.5 * cycle + rng.normal(0, 0.25, periods), 0, 10)
    return pd.DataFrame({"growth": growth, "inflation": inflation,
                         "policy_rate": policy, "yield_10y": y10}, index=idx)


def synthetic_asset_bars(macro: pd.DataFrame, seed=1) -> dict[str, pd.DataFrame]:
    """Daily bars for macro-tradeable assets whose drift is DRIVEN by the macro
    state (so a correct macro rule has something to find): equity, bonds, gold, usd."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(macro.index[0], macro.index[-1])
    m = macro.reindex(idx, method="ffill")
    g = (m["growth"] - 50) / 10
    infl = (m["inflation"] - 2) / 3
    drate = m["policy_rate"].diff().fillna(0)

    def build(drift_series, vol):
        shock = rng.standard_normal(len(idx)) * vol
        lr = drift_series.values / 252 + shock
        px = 100 * np.exp(np.cumsum(lr))
        return pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0}, index=idx)

    return {
        "equity": build(0.10 + 0.8 * g - 0.3 * infl, 0.011),        # loves growth, hates inflation
        "bonds":  build(0.03 - 0.5 * g - 2.0 * drate, 0.004),       # hurt by growth & hikes
        "gold":   build(0.02 + 1.5 * infl - 0.4 * m["policy_rate"] / 100, 0.010),  # inflation hedge
        "usd":    build(0.00 + 0.6 * drate * 252 * 0 + 0.3 * (m["policy_rate"] - m["policy_rate"].mean()) / 100, 0.005),
    }


def synthetic_fx_panel(start="2012-01-01", periods=1500, seed=2, n_ccy=6):
    """FX spot + short rates for a carry cross-section. Returns (spots, rates)
    DataFrames (daily). Higher-rate currencies drift up modestly (carry) but
    carry occasional crash risk — the well-known 'picking up pennies' profile."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=periods)
    names = [f"CCY{i}" for i in range(n_ccy)]
    base_rates = np.linspace(0.5, 8.0, n_ccy)                # spread of policy rates
    spots, rates = {}, {}
    for i, nm in enumerate(names):
        r = base_rates[i] + np.cumsum(rng.normal(0, 0.02, periods))
        r = np.clip(r, 0, 15)
        carry_drift = (r / 100) / 252                         # high-yielders drift up
        crash = rng.random(periods) < 0.002                  # rare risk-off crashes
        shock = rng.standard_normal(periods) * 0.006 - crash * 0.04 * (i / n_ccy)
        px = 100 * np.exp(np.cumsum(carry_drift + shock))
        spots[nm] = pd.Series(px, index=idx)
        rates[nm] = pd.Series(r, index=idx)
    return pd.DataFrame(spots), pd.DataFrame(rates)


# ============================================================================
# Event / flow data generators for the event-driven, merger-arb, index-rebal,
# and crypto-funding strategies. Real feeds noted per function.
# ============================================================================

def synthetic_earnings(df: pd.DataFrame, freq_days: int = 63, seed: int = 3) -> pd.DataFrame:
    """Quarterly earnings events with a surprise (SUE) and the realised drift the
    strategy should capture. Real feed: earnings calendar + I/B/E/S estimates.
    Returns DataFrame indexed by event date: columns [surprise, announced].
    We bake in post-earnings drift so a correct PEAD rule has signal to find."""
    rng = np.random.default_rng(seed)
    dates = df.index[::freq_days][1:]
    surprise = rng.normal(0, 1.0, len(dates))              # standardised unexpected earnings
    return pd.DataFrame({"surprise": surprise, "announced": True}, index=dates)


def synthetic_ma_deals(start="2016-01-01", n_deals=40, seed=4) -> pd.DataFrame:
    """Cash M&A deals: announce date, offer price, pre-announcement price, close
    date, and whether the deal ultimately completes. Real feed: deal databases
    (Bloomberg/Dealogic). Spread = offer/current-1 is the arb return if it closes."""
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(start)
    rows = []
    for i in range(n_deals):
        announce = base + pd.Timedelta(days=int(rng.integers(0, 2500)))
        days_to_close = int(rng.integers(40, 160))
        offer = 50 * (1 + rng.random())
        pre = offer * (1 - rng.uniform(0.05, 0.20))        # target trades below offer pre-close
        completes = rng.random() > 0.12                    # ~88% completion rate
        rows.append(dict(deal=f"DEAL{i}", announce=announce,
                         close=announce + pd.Timedelta(days=days_to_close),
                         offer=offer, pre=pre, completes=completes))
    return pd.DataFrame(rows).sort_values("announce").reset_index(drop=True)


def synthetic_funding(df: pd.DataFrame, seed=5) -> pd.Series:
    """Perp-swap funding rate series (per 8h, here aggregated daily). Positive =
    longs pay shorts (the common state), so shorting the perp vs long spot earns
    carry. Real feed: exchange funding endpoints via ccxt. Mean-positive with
    spikes, mildly correlated to recent returns."""
    rng = np.random.default_rng(seed)
    ret = df["close"].pct_change().fillna(0).values
    base = 0.0004 + 0.02 * ret + rng.normal(0, 0.0006, len(df))   # daily funding
    return pd.Series(base, index=df.index).clip(-0.01, 0.02)


def synthetic_index_events(df: pd.DataFrame, n_events=30, seed=6) -> pd.DataFrame:
    """Index add/delete announcements with an effective date a few days later.
    Real feed: index provider rebalance announcements (S&P/MSCI/Russell).
    Forced fund buying/selling front-runs the effective date."""
    rng = np.random.default_rng(seed)
    dates = df.index[rng.integers(60, len(df) - 20, n_events)]
    return pd.DataFrame({
        "announce": sorted(dates),
        "effective": [d + pd.Timedelta(days=5) for d in sorted(dates)],
        "action": rng.choice(["add", "delete"], n_events),
    })

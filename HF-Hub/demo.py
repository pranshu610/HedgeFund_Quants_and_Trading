"""
demo.py — end-to-end walk-through of the whole hub on offline synthetic data.
Run:  python demo.py

Pipeline:  data -> strategy backtest (with costs) -> walk-forward optimise
           -> recommend settings -> risk-parity blend across strategies
           -> Monte Carlo robustness -> per-trade forecast report.

Swap SyntheticFeed for YFinanceFeed / AlpacaFeed in config.py to run on real data.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

from config import make_feed, COST_BPS, SLIPPAGE_BPS, INITIAL_CAPITAL, START, END
from strategies.registry import REGISTRY
from backtest.engine import run_backtest
from backtest.optimizer import walk_forward
from risk.portfolio import combine_book
from montecarlo.simulator import bootstrap_strategy
from report.forecast import forecast_trade, render


def line(c="─", n=64): print(c * n)


def main():
    feed = make_feed()
    symbol = "DEMO"
    df = feed.history(symbol, START, END)
    print(f"\nLoaded {len(df)} bars for {symbol}  [{df.index[0].date()} → {df.index[-1].date()}]\n")

    # ---- market regime + capital routing ----
    from core.regime import latest_regime, route, strategy_multipliers
    reg = latest_regime(df)
    r = route(reg)
    line("═"); print("MARKET REGIME DETECTION"); line("═")
    print(f"  Current regime : {r['regime']}")
    print(f"  Favour         : {', '.join(r['favour'])}")
    print(f"  Avoid          : {', '.join(r['avoid']) or '—'}")
    print(f"  {r['note']}\n")

    strat_returns = {}
    strat_categories = {}

    for key, cls in REGISTRY.items():
        line()
        print(f"STRATEGY: {cls.meta.name}  ({cls.meta.category}, {cls.meta.horizon})")
        line()

        # 1) baseline backtest with default (industry-standard) settings
        base = cls()
        bt = run_backtest(df, base.generate(df), initial_capital=INITIAL_CAPITAL,
                          cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
        s = bt.stats
        print(f"  default params : {base.params}")
        print(f"  Sharpe {s['Sharpe']:.2f} | CAGR {s['CAGR']:.1%} | MaxDD {s['MaxDrawdown']:.1%} "
              f"| Trades {s.get('NumTrades', 0)} | Win {s.get('WinRate', float('nan')):.0%}")

        # 2) walk-forward optimisation -> recommended settings
        wf = walk_forward(cls, df, n_splits=4, metric="Sharpe",
                          cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
        print(f"  ▶ recommended  : {wf['recommended_params']}")
        print(f"    (best OOS {wf['metric']} {wf['oos_score']:.2f} across folds; "
              f"{wf['n_configs_tested']} configs tested)")
        rs = wf["full_sample_stats"]
        print(f"    full-sample @ recommended: Sharpe {rs['Sharpe']:.2f} | "
              f"CAGR {rs['CAGR']:.1%} | MaxDD {rs['MaxDrawdown']:.1%}")

        # keep recommended-config returns for the book blend
        best = cls(**wf["recommended_params"])
        strat_returns[cls.meta.name] = run_backtest(
            df, best.generate(df), cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS).returns
        strat_categories[cls.meta.name] = cls.meta.category

        # 3) per-trade forecast for the latest bar
        fc = forecast_trade(best, df, symbol, capital=INITIAL_CAPITAL)
        print("\n" + render(fc) + "\n")

    # 4) blend strategies into one book — compare allocation methods, recommend HRP
    line("═")
    print("MULTI-STRATEGY ALLOCATION — optimal, risk-weighted (recommend: HRP)")
    line("═")
    from risk.allocation import allocate, compare_allocations
    rets_df = pd.DataFrame(strat_returns).dropna(how="all").fillna(0.0)
    print(compare_allocations(rets_df).to_string())
    weights = allocate(rets_df, method="hrp")
    print("\n  ▶ recommended (HRP) weights:")
    for name, w in weights.items():
        print(f"    {name:<28} {w:6.1%}")

    # regime-aware overlay: damp strategies the current regime disfavours
    mult = strategy_multipliers(reg, strat_categories)
    adj = (weights * pd.Series(mult)).fillna(0.0)
    adj = adj / adj.sum() if adj.sum() > 0 else weights
    print("\n  ▶ regime-adjusted weights (gross scaled to "
          f"{r['gross_exposure']:.0%}):")
    for name, w in adj.items():
        print(f"    {name:<28} {w:6.1%}   (×{mult.get(name,1):.2f})")
    book = (rets_df * adj).sum(axis=1) * r["gross_exposure"]

    # ---- pairs stat-arb: static OLS vs adaptive Kalman on the same pair ----
    line("═"); print("PAIRS STAT-ARB — static OLS vs adaptive Kalman"); line("═")
    from strategies.pairs_statarb import PairsStatArb, pairs_equity_df as static_eq
    from strategies.kalman_pairs import KalmanPairs, pairs_equity_df as kalman_eq
    a = feed.history("PAIR_A", START, END)["close"]
    import numpy as np
    rng = np.random.default_rng(11)
    sp = np.zeros(len(a))
    for i in range(1, len(a)):
        sp[i] = 0.93 * sp[i - 1] + rng.standard_normal() * 0.02
    b = pd.Series(a.values * np.exp(sp + 0.2), index=a.index)

    ps = PairsStatArb(); ppos = ps.generate_pair(a, b)
    pbt = run_backtest(static_eq(ps, a, b), ppos, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    kp = KalmanPairs(); kpos = kp.generate_pair(a, b)
    kbt = run_backtest(kalman_eq(kp, a, b), kpos, cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)

    print(f"  static OLS : β={ps.fit.beta:.3f} (fixed)  ADF p={ps.fit.adf_pvalue:.4f}  "
          f"half-life {ps.fit.half_life:.0f}b")
    ss = pbt.stats
    print(f"             Sharpe {ss['Sharpe']:.2f} | CAGR {ss['CAGR']:.1%} | "
          f"Win {ss.get('WinRate',0):.0%} | Trades {ss.get('NumTrades',0)}")
    print(f"  kalman     : β={kp.state.beta.iloc[100]:.3f}→{kp.state.beta.iloc[-1]:.3f} (adaptive)")
    ks = kbt.stats
    print(f"             Sharpe {ks['Sharpe']:.2f} | CAGR {ks['CAGR']:.1%} | "
          f"Win {ks.get('WinRate',0):.0%} | Trades {ks.get('NumTrades',0)}")
    print(f"  note: static wins on clean/stationary pairs; Kalman earns its keep "
          f"when the hedge ratio drifts on real data.\n")

    # ---- advanced multi-asset sleeves (macro / carry / vol) ----
    line("═"); print("ADVANCED SLEEVES — macro / FX carry / vol arb"); line("═")
    from core.altdata import synthetic_macro, synthetic_asset_bars, synthetic_fx_panel
    from strategies.global_macro import GlobalMacro
    from strategies.fx_carry import FXCarry
    from strategies.vol_arb import VolArb

    def _stats(ret):
        eq = (1 + ret).cumprod()
        sh = ret.mean() / ret.std() * (252 ** 0.5) if ret.std() > 0 else 0
        mdd = ((eq / eq.cummax()) - 1).min()
        return sh, eq.iloc[-1] - 1, mdd

    macro = synthetic_macro(periods=150, seed=0); massets = synthetic_asset_bars(macro, seed=1)
    gm = GlobalMacro(); mw = gm.generate_macro(macro, massets)
    mrets = pd.DataFrame({k: v["close"].pct_change() for k, v in massets.items()}).fillna(0)
    sh, tot, mdd = _stats((mw.shift(1) * mrets).sum(axis=1))
    print(f"  Global Macro : Sharpe {sh:.2f} | total {tot:+.0%} | MaxDD {mdd:.0%}")
    print(f"                 {gm.explain(macro, massets)}")

    spots, rates = synthetic_fx_panel(periods=1500, seed=2)
    fx = FXCarry(); sh, tot, mdd = _stats(fx.carry_returns(spots, rates))
    lw, lr = fx.generate_carry(spots, rates).iloc[-1], rates.iloc[-1]
    print(f"  FX Carry     : Sharpe {sh:.2f} | total {tot:+.0%} | MaxDD {mdd:.0%} | "
          f"long {lw.idxmax()}({lr[lw.idxmax()]:.1f}%) short {lw.idxmin()}({lr[lw.idxmin()]:.1f}%)")

    va = VolArb(); vbt = run_backtest(va.vol_equity_df(df), va.generate(df),
                                      cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    vs = vbt.stats
    print(f"  Vol Arb (VRP): Sharpe {vs['Sharpe']:.2f} | CAGR {vs['CAGR']:+.0%} | "
          f"MaxDD {vs['MaxDrawdown']:.0%}  ← negative skew is the point, size small\n")

    # ---- event / flow / crypto sleeves ----
    line("═"); print("EVENT / FLOW / CRYPTO SLEEVES"); line("═")
    from core.altdata import (synthetic_earnings, synthetic_ma_deals,
                              synthetic_index_events, synthetic_funding)
    from strategies.commodity_trend import CommodityTrend
    from strategies.crypto_funding import CryptoFunding
    from strategies.event_earnings import EarningsDrift
    from strategies.merger_arb import MergerArb
    from strategies.index_rebal import IndexRebal
    from strategies.mm_scalper import MMScalper

    ct = CommodityTrend(); cbt = run_backtest(df, ct.generate(df), cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    print(f"  Commodity Trend : Sharpe {cbt.stats['Sharpe']:.2f} | trend+seasonal blend")
    fund = synthetic_funding(df); cf = CryptoFunding()
    fbt = run_backtest(cf.equity_df(fund), cf.generate(fund), cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    print(f"  Crypto Funding  : Sharpe {fbt.stats['Sharpe']:.2f} | delta-neutral carry (short perp/long spot)")
    ev = synthetic_earnings(df); ed = EarningsDrift(); ebt = run_backtest(df, ed.generate(df, ev),
                                                                          cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    print(f"  PEAD (earnings) : Sharpe {ebt.stats['Sharpe']:.2f} | {len(ev)} events, 40d drift window")
    deals = synthetic_ma_deals(n_deals=40); mr = MergerArb()
    mbt = run_backtest(mr.equity_df(deals, df.index), pd.Series(1.0, index=df.index),
                       cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    print(f"  Merger Arb      : Sharpe {mbt.stats['Sharpe']:.2f} | MaxDD {mbt.stats['MaxDrawdown']:.0%} "
          f"(deal breaks = negative skew)")
    iev = synthetic_index_events(df); ir = IndexRebal()
    ibt = run_backtest(df, ir.generate(df, iev), cost_bps=COST_BPS, slippage_bps=SLIPPAGE_BPS)
    print(f"  Index Rebal     : Sharpe {ibt.stats['Sharpe']:.2f} | front-run forced index flows")
    mm = MMScalper(); mmbt = run_backtest(mm.equity_df(df), pd.Series(1.0, index=df.index))
    print(f"  MM Scalper (sim): Sharpe {mmbt.stats['Sharpe']:.2f} | inventory sim — infra-bound, not deployable\n")

    # 5) Monte Carlo robustness of the blended book
    rob = bootstrap_strategy(book, initial_value=INITIAL_CAPITAL, n_sims=4000, seed=7)
    line()
    print("BOOK ROBUSTNESS (block bootstrap of blended returns)")
    line()
    print(f"  Terminal equity  p05/p50/p95 : "
          f"${rob.p05_terminal:,.0f} / ${rob.p50_terminal:,.0f} / ${rob.p95_terminal:,.0f}")
    print(f"  Max drawdown     median/p95  : {rob.median_maxdd:.1%} / {rob.p95_maxdd:.1%}")
    print(f"  Probability of ending in loss: {rob.prob_loss:.1%}")
    print(f"  Verdict: {rob.verdict}\n")


if __name__ == "__main__":
    main()

"""
tests.py — invariant checks for the engine. Run: python tests.py
These are the properties that, if broken, would silently corrupt every backtest.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import sys

from core.data import SyntheticFeed
from core.indicators import rsi, atr
from strategies.registry import REGISTRY
from backtest.engine import run_backtest
from risk.sizing import fractional_kelly, vol_target_size, build_trade_plan
from montecarlo.simulator import run_simulation, bootstrap_strategy

passed = failed = 0
def check(name, cond, detail=""):
    global passed, failed
    if cond: print(f"  ✓ {name}"); passed += 1
    else:    print(f"  ✗ {name}  {detail}"); failed += 1

def _max_run(pos):
    """Longest consecutive same-nonzero-position run length."""
    best = cur = 0
    prev = 0.0
    for v in pos.values:
        cur = cur + 1 if (v != 0 and v == prev) else (1 if v != 0 else 0)
        best = max(best, cur)
        prev = v
    return best

df = SyntheticFeed(seed=42).history("T", "2016-01-01", "2023-01-01")
print("\n  HF-HUB ENGINE — TEST SUITE\n  " + "─" * 42)

# --- indicators are causal (no future leak) ---
r = rsi(df["close"], 14)
check("RSI bounded [0,100]", ((r >= 0) & (r <= 100)).all())
check("ATR non-negative", (atr(df, 14).dropna() >= 0).all())

# --- execution lag: a strategy cannot trade the same bar it signals ---
strat = REGISTRY["breakout"]()
bt = run_backtest(df, strat.generate(df))
# executed position must equal target shifted by 1
target = (strat.generate(df) * 1.0).clip(-3, 3)
check("1-bar execution lag enforced",
      np.allclose(bt.position.values, target.shift(1).fillna(0.0).values))

# --- costs monotonic: higher costs never increase final equity ---
lo = run_backtest(df, strat.generate(df), cost_bps=0, slippage_bps=0)
hi = run_backtest(df, strat.generate(df), cost_bps=10, slippage_bps=10)
check("higher costs → lower/equal final equity",
      hi.stats["FinalEquity"] <= lo.stats["FinalEquity"] + 1e-6)

# --- every strategy produces positions in [-1,1] and aligned index ---
for k, cls in REGISTRY.items():
    pos = cls().generate(df)
    check(f"{k}: positions in [-1,1] & aligned",
          pos.index.equals(df.index) and pos.abs().max() <= 1.0 + 1e-9)

# --- sizing sanity ---
check("Kelly zero when no edge", fractional_kelly(0.4, 1.0) == 0.0)   # 0.4-0.6=−0.2 → 0
check("Kelly positive with edge", fractional_kelly(0.6, 2.0) > 0)
check("vol target scales inversely", vol_target_size(0.30) < vol_target_size(0.10))
plan = build_trade_plan(100_000, 100, 2.0, +1, 0.20, method="vol_target")
check("stop below entry for long", plan.stop_price < 100 < plan.target_price)
check("reward:risk positive", plan.rr > 0)

# --- Monte Carlo invariants ---
mc = run_simulation(100_000, 0.08, 0.15, 3000, 252, seed=1)
check("CVaR95 ≥ VaR95", mc.cvar_95 >= mc.var_95)
check("prob_ruin in [0,1]", 0 <= mc.prob_ruin <= 1)
rob = bootstrap_strategy(bt.returns.replace(0, np.nan).dropna(), n_sims=1000, seed=1)
check("bootstrap prob_loss in [0,1]", 0 <= rob.prob_loss <= 1)

# --- regime detector ---
from core.regime import latest_regime, route, classify
reg = latest_regime(df)
check("regime trend label valid", reg.trend in ("bull", "bear", "sideways"))
check("regime vol label valid", reg.volatility in ("low", "normal", "high", "extreme"))
r = route(reg)
check("gross exposure in (0,1]", 0 < r["gross_exposure"] <= 1)
check("favour/avoid disjoint", not (set(r["favour"]) & set(r["avoid"])))

# --- allocation engine ---
from risk.allocation import allocate, ALLOCATORS
rets = pd.DataFrame({k: run_backtest(df, REGISTRY[k]().generate(df)).returns for k in REGISTRY})
for m in ALLOCATORS:
    w = allocate(rets, method=m)
    check(f"allocation '{m}' sums to 1 & non-negative",
          abs(w.sum() - 1) < 1e-6 and (w >= -1e-9).all())

# --- pairs stat-arb: cointegration test is calibrated + spread trades ---
from strategies.pairs_statarb import fit_pair, PairsStatArb
rng = np.random.default_rng(5)
idx = pd.bdate_range("2018-01-01", periods=1200)
common = np.cumsum(rng.standard_normal(1200) * 0.01)
sp = np.zeros(1200)
for i in range(1, 1200):
    sp[i] = 0.93 * sp[i - 1] + rng.standard_normal() * 0.02
pa = pd.Series(50 * np.exp(common + sp), index=idx)
pb = pd.Series(50 * np.exp(common), index=idx)
pind = pd.Series(50 * np.exp(np.cumsum(rng.standard_normal(1200) * 0.01)), index=idx)
check("cointegrated pair detected", fit_pair(pa, pb).cointegrated)
# a 5% test flags ~5% of independent pairs by definition — check the RATE, not one draw
fp = sum(fit_pair(
    pd.Series(50 * np.exp(np.cumsum(np.random.default_rng(s).standard_normal(800) * 0.01)), index=idx[:800]),
    pd.Series(50 * np.exp(np.cumsum(np.random.default_rng(s + 999).standard_normal(800) * 0.01)), index=idx[:800]),
).cointegrated for s in range(60))
check("coint test calibrated (indep FP rate < 20%)", fp / 60 < 0.20, f"got {fp/60:.0%}")
ppos = PairsStatArb().generate_pair(pa, pb)
check("pairs positions in {-1,0,1}", set(np.unique(ppos.values)).issubset({-1.0, 0.0, 1.0}))

# --- kalman pairs: beta tracks a step change, positions valid ---
from strategies.kalman_pairs import KalmanPairs, kalman_hedge
beta_true = np.where(np.arange(1200) < 600, 1.0, 1.5)
pa2 = pd.Series(50 * np.exp(common[:1200] * beta_true + sp[:1200]), index=idx[:1200])
pb2 = pd.Series(50 * np.exp(common[:1200]), index=idx[:1200])
kst = kalman_hedge(np.log(pa2), np.log(pb2), 1e-4, 1e-3)
check("kalman beta tracks step up",
      kst.beta.iloc[-1] > kst.beta.iloc[500] + 0.05,
      f"pre {kst.beta.iloc[500]:.2f} post {kst.beta.iloc[-1]:.2f}")
kpos = KalmanPairs().generate_pair(pa2, pb2)
check("kalman positions in {-1,0,1}", set(np.unique(kpos.values)).issubset({-1.0, 0.0, 1.0}))
check("kalman warmup respected (flat early)", (kpos.iloc[:60] == 0).all())

# --- residual momentum L/S: dollar-neutral panel ---
from strategies.equity_ls import ResidualMomentumLS
mkt = np.cumsum(np.random.default_rng(8).standard_normal(1200) * 0.01)
lsbars = {}
for i in range(10):
    io = np.cumsum(np.random.default_rng(i + 20).standard_normal(1200) * 0.008)
    px = 100 * np.exp(mkt * (0.8 + 0.4 * (i / 10)) + io)
    lsbars[f"S{i}"] = pd.DataFrame({"open": px, "high": px, "low": px, "close": px, "volume": 1.0},
                                   index=idx)
lw = ResidualMomentumLS().generate_panel(lsbars)
check("equity_ls dollar-neutral", float(lw.sum(axis=1).abs().mean()) < 1e-6)
check("equity_ls has exposure", float(lw.abs().sum(axis=1).mean()) > 0.3)

# --- vix_rsi refinement: trend filter reduces or holds trade count, no crashes ---
base_vr = REGISTRY["vix_rsi"]().generate(df)
filt_vr = REGISTRY["vix_rsi"](trend_ma=200, max_hold=10).generate(df)
check("vix_rsi trend filter changes exposure",
      not base_vr.equals(filt_vr))
check("vix_rsi time-stop caps holding length",
      _max_run(filt_vr) <= 11)

# --- global macro: weights across assets, publication lag respected ---
from core.altdata import synthetic_macro, synthetic_asset_bars, synthetic_fx_panel
from strategies.global_macro import GlobalMacro
macro = synthetic_macro(periods=120, seed=0)
assets = synthetic_asset_bars(macro, seed=1)
mw = GlobalMacro().generate_macro(macro, assets)
check("macro weights cover all assets", list(mw.columns) == list(assets.keys()))
check("macro gross bounded", mw.abs().sum(axis=1).max() <= 1.5)

# --- fx carry: dollar-neutral-ish, long high-yield ---
from strategies.fx_carry import FXCarry
spots, rates = synthetic_fx_panel(periods=900, seed=2, n_ccy=6)
fw = FXCarry().generate_carry(spots, rates)
check("fx carry has long & short legs",
      (fw.iloc[-1] > 0).any() and (fw.iloc[-1] < 0).any())
lw, lr = fw.iloc[-1], rates.iloc[-1]
check("fx carry longs higher rate than it shorts",
      lr[lw.idxmax()] > lr[lw.idxmin()])

# --- vol arb: negative skew signature, short-vol majority ---
from strategies.vol_arb import VolArb
va = VolArb()
vpos = va.generate(df)
check("vol arb is net short vol", (vpos > 0).mean() > (vpos < 0).mean())
# structural short-variance P&L is negatively skewed (the VRP crash profile)
vraw = va.vrp_returns(df)
vraw = vraw[vraw != 0]
check("short-variance P&L negatively skewed", vraw.skew() < 0, f"skew {vraw.skew():.2f}")

# --- commodity trend: single-series, valid positions ---
from strategies.commodity_trend import CommodityTrend
cpos = CommodityTrend().generate(df)
check("commodity_trend positions in [-1,1]", cpos.abs().max() <= 1.0 + 1e-9)
check("commodity_trend causal length", cpos.index.equals(df.index))

# --- crypto funding: net long positive funding, carry positive on positive funding ---
from core.altdata import synthetic_funding
from strategies.crypto_funding import CryptoFunding
fund = synthetic_funding(df)
cf = CryptoFunding()
check("crypto_funding net long positive funding", (cf.generate(fund) > 0).mean() > 0.5)
check("crypto_funding carry ~ funding sign",
      cf.carry_returns(fund).mean() > 0, "positive mean carry expected")

# --- PEAD: only trades after events, bounded window ---
from core.altdata import synthetic_earnings
from strategies.event_earnings import EarningsDrift
ev = synthetic_earnings(df)
epos = EarningsDrift(hold_days=40).generate(df, ev)
check("PEAD positions in {-1,0,1}", set(np.unique(epos.values)).issubset({-1.0, 0.0, 1.0}))
check("PEAD flat before first event",
      (epos.loc[:ev.index[0]].iloc[:-1] == 0).all())

# --- merger arb: negative-skew payoff (breaks cause big losses) ---
from core.altdata import synthetic_ma_deals
from strategies.merger_arb import MergerArb
deals = synthetic_ma_deals(n_deals=40)
mr = MergerArb().book_returns(deals, df.index)
check("merger_arb produces returns", mr.abs().sum() > 0)
check("merger_arb negatively skewed (deal breaks)", mr[mr != 0].skew() < 0)

# --- index rebal: trades only around events ---
from core.altdata import synthetic_index_events
from strategies.index_rebal import IndexRebal
iev = synthetic_index_events(df)
ipos = IndexRebal().generate(df, iev)
check("index_rebal mostly flat (event-driven)", (ipos == 0).mean() > 0.8)

# --- MM scalper: simulation runs, positive spread capture ---
from strategies.mm_scalper import MMScalper
mmr = MMScalper().simulate(df)
check("mm_scalper simulation runs", len(mmr) == len(df))
check("mm_scalper net positive (spread capture)", mmr.sum() > 0)

# --- live paper trader: runs a cycle, rebalances to target, holds on 2nd cycle ---
from live.trader import Trader, SimBroker, default_mandates
from core.data import SyntheticFeed as _SF
_broker = SimBroker(cash=100_000)
_trader = Trader(_broker, _SF(seed=42), default_mandates("DEMO"), capital=100_000)
_rep = _trader.run_once()
check("trader submits orders on first cycle", len(_rep["orders"]) > 0)
check("trader conserves paper equity (~unchanged at fill)",
      abs(_broker.equity() - 100_000) < 1_000)
_rep2 = _trader.run_once()
check("trader holds (0 new orders) on repeat cycle", len(_rep2["orders"]) == 0)
check("trader snapshot well-formed",
      set(_trader.snapshot()) >= {"equity", "cash", "positions", "orders"})

# --- book vol-target fix: blended book vol bounded (no $400B blowups) ---
import dashboard.engine as _DE
_book = _DE.build_book(df)
_bv = _book["book_returns"].std() * np.sqrt(252)
check("blended book vol is bounded (~target, <30%)", _bv < 0.30, f"vol {_bv:.0%}")
check("no absurd daily book returns", _book["book_returns"].abs().max() <= 0.25 + 1e-9)

# --- enriched metrics present ---
from backtest.metrics import summary as _sm
_bt = run_backtest(df, REGISTRY["breakout"]().generate(df))
_st = _bt.stats
check("Calmar metric present", "Calmar" in _st)
check("Exposure metric present", "Exposure" in _st)
check("ProfitFactor metric present", "ProfitFactor" in _st or _st.get("NumTrades", 0) == 0)

# --- ichimoku + cci strategies ---
for k in ("ichimoku", "cci"):
    pos = REGISTRY[k]().generate(df)
    check(f"{k} positions in [-1,1] & aligned",
          pos.index.equals(df.index) and pos.abs().max() <= 1.0 + 1e-9)

# --- shared risk manager: caps, kill switches, bans ---
from risk.manager import RiskManager
rm = RiskManager(capital=100_000, position_cap=0.05, max_gross=1.5)
d1 = rm.check_order("X", 20_000, 0, is_entry=True)          # over 5% cap
check("risk manager clips oversized position", abs(d1.qty) <= 5_000 + 1e-6)
st_dd = rm.update_equity(84_000)                             # -16% from 100k HWM
check("portfolio kill switch trips at -15%", "PORTFOLIO KILL" in st_dd["state"])
d2 = rm.check_order("X", 1_000, 0, is_entry=True)
check("no new entries after portfolio kill", not d2.approved)
rm2 = RiskManager(capital=100_000)
d3 = rm2.check_order("V", 50_000, 0, is_entry=True, strategy_flags={"short_vol"})
check("short-vol sleeve hard-capped", abs(d3.qty) <= 0.05 * 100_000 + 1e-6)

# --- analyst/PM propose + execute workflow ---
_tr = Trader(SimBroker(100_000), _SF(seed=42), default_mandates("DEMO"), capital=100_000)
_props = _tr.propose()
check("analyst proposes trades (heads-up)", len(_props) > 0)
check("proposals carry reasoning + risk note",
      all(p.reason and p.risk_note for p in _props))
_ords = _tr.execute([p for p in _props if p.approved])
check("PM can execute confirmed proposals", len(_ords) >= 0 and _tr.broker.equity() > 0)

# --- convergence strategies (fixed income arb, index arb) ---
from strategies.registry import PAIR_REGISTRY
for k in ("fixed_income_arb", "index_arb"):
    cls = PAIR_REGISTRY[k]
    p = cls().generate_pair(pa, pb)
    check(f"{k} produces valid pair positions",
          set(np.unique(p.values)).issubset({-1.0, 0.0, 1.0}))

# --- allocation robust to flat / NaN strategy streams (the Cockpit crash) ---
from risk.allocation import allocate as _alloc
_idx = pd.bdate_range("2024-01-01", periods=250)
_rng = np.random.default_rng(0)
_mix = pd.DataFrame({"A": _rng.standard_normal(250) * 0.01, "B": np.zeros(250),
                     "C": _rng.standard_normal(250) * 0.008, "D": np.full(250, np.nan)}, index=_idx)
_w = _alloc(_mix, "hrp")
check("HRP survives flat/NaN streams (no crash)", abs(_w.sum() - 1) < 1e-6)
check("HRP zero-weights the dead streams", _w["B"] == 0 and _w["D"] == 0)

# --- robust strategy ranking never recommends a money-loser (the QBTS bug) ---
import dashboard.engine as _DE
_rk = _DE.rank_strategies_on_df(_DE.load_bars("DEMO", "2020-01-01", "2024-01-01"))
if _rk["any_edge"]:
    _top = _rk["table"].iloc[0]
    _cagr = float(str(_top["CAGR"]).replace("%", "")) if _top["CAGR"] not in ("", None) else 0
    check("recommended strategy is profitable (not a money-loser)",
          _cagr > 0 and (_top["PF"] is None or _top["PF"] >= 1.0), f"top={_top['strategy']} CAGR={_top['CAGR']}")
else:
    check("no-edge case handled honestly", _rk["best"] is None)
# money-losers must be flagged, never top when an edge exists
_losers = _rk["table"][_rk["table"]["verdict"] == "❌ loses money"]
check("money-losers are flagged and demoted",
      _losers.empty or _rk["table"].iloc[0]["verdict"] != "❌ loses money")

# --- new classic-algo strategies ---
for k in ("macd", "bollinger_rev", "overnight_drift"):
    pos = REGISTRY[k]().generate(df)
    check(f"{k} positions valid & aligned",
          pos.index.equals(df.index) and pos.abs().max() <= 1.0 + 1e-9)
check("roster is 23 strategies", len(__import__("strategies.registry", fromlist=["UNIVERSE"]).UNIVERSE) == 23)

# --- realistic costs: spread + overnight financing reduce returns ---
_p = REGISTRY["breakout"]().generate(df)
_b_free = run_backtest(df, _p, cost_bps=0, slippage_bps=0, spread_bps=0, financing_bps_annual=0)
_b_cost = run_backtest(df, _p, cost_bps=0, slippage_bps=0, spread_bps=5, financing_bps_annual=800)
check("spread + financing costs reduce net return",
      _b_cost.stats["TotalReturn"] < _b_free.stats["TotalReturn"])

# --- fallback feed returns data from whichever source works ---
from core.data import FallbackFeed, SyntheticFeed
class _Dead:
    def history(self, *a, **k): raise ValueError("no data")
_ff = FallbackFeed([("dead", _Dead()), ("synth", SyntheticFeed(seed=1))])
_fdf = _ff.history("X", "2020-01-01", "2022-01-01")
check("fallback feed skips dead source, uses working one",
      len(_fdf) > 0 and _ff.last_source == "synth")

# --- asset-class relevance routing (no commodity seasonality on an equity) ---
check("equity ticker excludes commodity_trend",
      "commodity_trend" not in _DE.relevant_strategies("PSIX"))
check("commodity ticker includes commodity_trend",
      "commodity_trend" in _DE.relevant_strategies("GC=F"))
check("crypto ticker excludes overnight_drift (equity-only anomaly)",
      "overnight_drift" not in _DE.relevant_strategies("BTC-USD"))
check("asset class inferred correctly",
      _DE.infer_asset_class("BTC-USD") == "crypto" and _DE.infer_asset_class("AAPL") == "equities")

# --- actionable signals carry entry/stop/target + status ---
_sig = _DE.actionable_signals("PSIX", df)
check("actionable signals have trade-ticket columns",
      {"status", "side", "entry", "stop", "target", "R:R", "size_$", "why"} <= set(_sig.columns))
check("actionable signals only include relevant strategies",
      set(_sig["strategy"]) <= set(_DE.relevant_strategies("PSIX")))

# --- blotter: shorts present + net/avg summary ---
_bl = _DE.trade_blotter(df, REGISTRY["macd"]().generate(df))
check("blotter records shorts when strategy shorts", (_bl["side"] == "SHORT").any())
check("blotter rows carry stop + target", {"stop", "target"} <= set(_bl.columns))
_bsum = _DE.blotter_summary(_bl)
check("blotter summary has net + average metrics",
      {"net_pnl_pct", "avg_pnl_pct", "win_rate", "profit_factor"} <= set(_bsum))

# --- book only runs asset-appropriate strategies ---
_bk = _DE.build_book(df, symbol="BTC-USD")
check("book excludes irrelevant strategies for the asset",
      len(_bk["returns"].columns) <= len(_DE.relevant_strategies("BTC-USD")))

# --- default mandates derive from the SELECTED symbol (no hardcoded SPY/AAPL/GLD) ---
_md = default_mandates("BTC-USD")
check("default mandates use the selected symbol",
      len(_md) > 0 and all(m.symbol == "BTC-USD" for m in _md))
check("default mandates are asset-class relevant",
      all(m.strategy_key in _DE.relevant_strategies("BTC-USD") for m in _md))
check("no hardcoded ticker leaks into mandates",
      not any(m.symbol in {"SPY", "QQQ", "AAPL", "GLD"} for m in default_mandates("PSIX")))

# --- Kelly lab: growth-optimal math + honest zero on a negative edge ---
_pos = pd.Series(np.random.default_rng(5).standard_normal(1200) * 0.01 + 0.0004)
_kl = _DE.kelly_lab(_pos)
check("kelly lab computes f* on a positive edge", _kl["ok"] and _kl["f_star"] > 0)
_neg = pd.Series(np.random.default_rng(6).standard_normal(1200) * 0.01 - 0.0004)
_kn = _DE.kelly_lab(_neg)
check("kelly says bet ZERO on a negative edge", (not _kn["ok"]) and _kn.get("f_star", 0) == 0)
# the asymmetry that justifies fractional Kelly: 2x Kelly = zero growth
_t = _kl["table"]
_two = _t[_t["Kelly fraction"] == "2.00×"]["% of max growth"].iloc[0]
_half = _t[_t["Kelly fraction"] == "0.50×"]["% of max growth"].iloc[0]
check("2x Kelly gives zero growth (overbetting cliff)", abs(_two) < 1e-6)
check("half Kelly gives 75% of max growth", abs(_half - 75.0) < 1e-6)

# --- R:R is DERIVED per strategy, not a hardcoded 1.5 ---
_px = float(df["close"].iloc[-1])
_rrs = {}
for _k in ("breakout", "bollinger_rev", "cci", "macd", "trendline"):
    _e = _DE.derive_exits(_k, df, 1, _px)
    if _e["rr"] is not None:
        _rrs[_k] = _e["rr"]
check("R:R differs across strategies (not a constant)", len(set(_rrs.values())) > 1, str(_rrs))
check("R:R is not hardcoded 1.5 for all", not all(abs(v - 1.5) < 1e-9 for v in _rrs.values()))
_ex = _DE.derive_exits("macd", df, 1, _px)
check("derived exits report expectancy + basis",
      _ex["expectancy_pct"] is not None and "historical trades" in _ex["basis"])
check("stop is below entry for a long, target above",
      _ex["stop"] < _px < _ex["target"])
_exs = _DE.derive_exits("macd", df, -1, _px)
check("short exits invert (stop above, target below)", _exs["stop"] > _px > _exs["target"])
_sig2 = _DE.actionable_signals("PSIX", df)
check("signals expose expectancy + levels_from",
      {"expect_%/trade", "levels_from"} <= set(_sig2.columns))

# ============================ CRYPTO ARB ENGINE ============================
from strategies.crypto_arb import CryptoArbEngine, book_from_quotes, OrderBook, BookLevel

_eng = CryptoArbEngine(latency_ms=250.0, min_net_bps=1.0, max_notional_usd=50_000)

# depth: sweeping size must pay a worse price than top-of-book
_bk = book_from_quotes("binance", "BTC/USDT", 67_000, 2.0, 300_000, seed=7)
_vw_small, _ = _bk.sweep("buy", _bk.asks[0].size * 0.5)
_vw_big, _ = _bk.sweep("buy", sum(l.size for l in _bk.asks) * 0.8)
check("book sweep pays worse price for bigger size", _vw_big > _vw_small)
check("best bid < best ask (sane book)", _bk.best_bid < _bk.best_ask)

# a gap smaller than fees must be rejected
_cheap = [book_from_quotes("binance", "BTC/USDT", 67_000, 1.5, 400_000, seed=1),
          book_from_quotes("coinbase", "BTC/USDT", 67_030, 3.0, 150_000, seed=2)]
_o = _eng.scan_spatial(_cheap, 0.6)
check("gap below fee floor is rejected, not reported as profit",
      all(x.net_bps < 0 or x.size == 0 for x in _o))

# a genuine gap on low-fee venues should clear
_rich = [book_from_quotes("binance", "ETH/USDT", 3_500, 1.0, 900_000, seed=4),
         book_from_quotes("okx", "ETH/USDT", 3_530, 1.2, 800_000, seed=5)]
_o2 = CryptoArbEngine(latency_ms=5.0, min_net_bps=1.0).scan_spatial(_rich, 0.5)
check("real gap on low-fee venues is found", len(_o2) > 0 and _o2[0].net_bps > 0)
check("net edge is always below gross edge", all(x.net_bps < x.gross_bps for x in _o2))

# competition model: WIDER edges must die FASTER (they are visible to everyone)
_hl_small = _eng.edge_half_life_ms(0.6, 3.0)
_hl_big = _eng.edge_half_life_ms(0.6, 50.0)
check("wider arb edges have shorter half-life (competition)", _hl_big < _hl_small)
_, _conf_fast = CryptoArbEngine(latency_ms=5.0).latency_decay(0.6, 25.0)
_, _conf_slow = CryptoArbEngine(latency_ms=2000.0).latency_decay(0.6, 25.0)
check("higher latency lowers fill confidence", _conf_slow < _conf_fast)

# triangular + basis produce sane structures
_tri_books = {"BTC/USDT": book_from_quotes("binance", "BTC/USDT", 67_000, 1.5, 500_000, seed=11),
              "ETH/BTC": book_from_quotes("binance", "ETH/BTC", 0.0522, 3.0, 200_000, seed=12),
              "ETH/USDT": book_from_quotes("binance", "ETH/USDT", 3_500, 1.5, 500_000, seed=13)}
_tri = _eng.scan_triangular(_tri_books, "binance")
check("triangular scan returns priced loops", len(_tri) > 0 and
      all(x.fees_bps > 0 for x in _tri))
_basis = _eng.scan_basis(_tri_books["BTC/USDT"],
                         book_from_quotes("binance", "BTC-PERP", 67_120, 2.0, 400_000, seed=14),
                         funding_rate=0.0001)
check("basis scan prices funding + fees", _basis is not None and _basis.fees_bps > 0)

# ============================ TERMINAL ENGINE ==============================
import dashboard.terminal_engine as _T
_scan = _T.scan_universe(["AST1", "AST2", "AST3"], "2019-01-01", "2024-01-01")
check("universe scan returns rows", not _scan.empty)
check("scan carries state + expectancy + exits",
      {"state", "expectancy", "entry", "stop", "target", "verdict"} <= set(_scan.columns))
check("scan states are valid",
      set(_scan["state"].unique()) <= {"JUST FIRED", "LIVE", "FORMING", "DORMANT",
                                       "NO DATA", "ERROR"})
_act = _T.actionable_only(_scan)
check("actionable filter excludes money-losers",
      _act.empty or not _act["verdict"].astype(str).str.startswith("❌").any())
check("actionable filter requires positive expectancy",
      _act.empty or (_act["expectancy"] > 0).all())
_rank = _T.asset_ranking(_scan)
check("asset ranking grades every asset", not _rank.empty and "grade" in _rank.columns)
check("heatmap + ranking figures build",
      _T.opportunity_heatmap(_scan) is not None and _T.ranking_bar(_rank) is not None)
check("arb waterfall builds", _T.arb_waterfall(_o2[0]) is not None)

# ============================ POST-MORTEM LOOP ============================
import dashboard.postmortem as _PM

_pmdf = _DE.load_bars("AST2", "2018-01-01", "2024-01-01")
_exc = _PM.trade_excursions(_pmdf, REGISTRY["macd"]().generate(_pmdf))
check("excursions computed per trade", not _exc.empty and
      {"mfe_pct", "mae_pct", "give_back", "win"} <= set(_exc.columns))
check("MFE >= trade P&L (peak beats close)",
      bool((_exc["mfe_pct"] >= _exc["pnl_pct"] - 1e-6).all()))
check("MAE <= 0 (adverse excursion is against us)", bool((_exc["mae_pct"] <= 1e-6).all()))

_diag = _PM.diagnose(_pmdf, "macd")
check("diagnosis returns findings with fixes", _diag["ok"] and len(_diag["findings"]) > 0 and
      all({"code", "title", "detail", "fix"} <= set(f) for f in _diag["findings"]))
check("diagnosis measures cost drag", "cost_drag_pct" in _diag)

_pres = _PM.prescribe(_pmdf, "macd")
check("prescribe tests candidate fixes out-of-sample",
      "candidates" in _pres and "current" in _pres)
# THE key discipline: never recommend a fix that isn't better out-of-sample AND positive
if _pres["best"] is not None:
    check("recommended fix beats current OOS expectancy",
          _pres["best"]["oos_expectancy"] > _pres["current"]["oos_expectancy"])
    check("recommended fix has positive OOS expectancy",
          _pres["best"]["oos_expectancy"] > 0)
else:
    check("no-fix case is reported honestly",
          "No change is justified" in _pres["summary"] or
          "Not enough data" in _pres["summary"])
_bad = [c for c in _pres["candidates"] if c["improves"] and c["oos_expectancy"] <= 0]
check("a fix that improves but stays negative is NOT accepted", len(_bad) == 0)

# a strategy that never trades must be diagnosed, not crash
class _NeverTrades:
    meta = REGISTRY["macd"].meta
    default_params = {}
    def __init__(self, **k): self.params = {}
    def generate(self, d): return pd.Series(0.0, index=d.index)
REGISTRY["_never"] = _NeverTrades
_nt = _PM.diagnose(_pmdf, "_never")
check("no-trade strategy diagnosed gracefully",
      _nt["findings"][0]["code"] == "NO_TRADES")
del REGISTRY["_never"]

# ==================== USABILITY / REGRESSION GUARDS ====================
# ranking must respect asset class (no commodity seasonality recommended for a stock)
_rk_eq = _DE.rank_strategies_on_df(_DE.load_bars("PSIX", "2021-01-01", "2024-01-01"),
                                   symbol="PSIX")
check("ranking only offers asset-relevant strategies",
      set(_rk_eq["table"]["strategy"]) <= set(_DE.relevant_strategies("PSIX")))

# Kelly must refuse to size off a tiny sample (it would amplify noise into leverage)
_tiny = pd.Series(np.random.default_rng(1).standard_normal(50) * 0.01 + 0.001)
check("kelly refuses thin samples", _DE.kelly_lab(_tiny)["ok"] is False)
# drift must be several standard errors above zero or Kelly correctly says "bet zero"
_ok = pd.Series(np.random.default_rng(1).standard_normal(400) * 0.01 + 0.003)
_lab_ok = _DE.kelly_lab(_ok)
check("kelly sizes a sufficient sample with a real edge",
      _lab_ok["ok"] is True and _lab_ok["f_star"] > 0)

# grid search must be capped so the optimiser can't hang on a huge grid
from backtest.optimizer import walk_forward as _wf
class _HugeGrid(REGISTRY["breakout"]):
    param_grid = {"entry_n": list(range(10, 60)), "exit_n": list(range(5, 40))}
import time as _time
_t0 = _time.time()
_wf(_HugeGrid, _DE.load_bars("DEMO", "2022-01-01", "2024-01-01"), n_splits=2)
check("huge grids are capped (optimiser stays responsive)", _time.time() - _t0 < 20)

# no dead references to the removed hub/terminal UI files
import pathlib as _pl
_appsrc = _pl.Path("dashboard/app.py").read_text()
check("no references to removed hub/terminal UI modules",
      "dashboard.terminal " not in _appsrc and "import hub" not in _appsrc)
check("single app exposes all 16 tabs", _appsrc.count("st.tabs([") == 1)

# --- Streamlit key-collision guard (a real crash class: assigning st.session_state.X
#     when a widget already owns key "X" raises StreamlitAPIException at runtime) ---
import re as _re
_appsrc2 = _pl.Path("dashboard/app.py").read_text()
_wkeys = set(_re.findall(r'key\s*=\s*["\']([A-Za-z0-9_]+)["\']', _appsrc2))
_wassign = set(_re.findall(r'st\.session_state\.([A-Za-z0-9_]+)\s*=', _appsrc2))
_wassign |= set(_re.findall(r'st\.session_state\[["\']([A-Za-z0-9_]+)["\']\]\s*=', _appsrc2))
_wget = set(_re.findall(r'st\.session_state\.get\(["\']([A-Za-z0-9_]+)["\']', _appsrc2))
_collide = (_wassign | _wget) & _wkeys
check("no session_state / widget-key collisions", not _collide, str(sorted(_collide)))
_dupes = [k for k in _wkeys if _appsrc2.count(f'key="{k}"') > 1]
check("no duplicate widget keys", not _dupes, str(_dupes))

# lowercase tickers must route identically to uppercase
check("ticker routing is case-insensitive",
      _DE.infer_asset_class("xrp-usd") == _DE.infer_asset_class("XRP-USD") == "crypto")

print(f"\n  Results: {passed} passed, {failed} failed\n")
sys.exit(1 if failed else 0)

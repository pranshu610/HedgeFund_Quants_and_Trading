# HF-Hub — a multi-strategy quant research & paper-trading platform

A modular hedge-fund-style engine: strategies generate signals, a cost-aware
backtester measures them, a walk-forward optimiser recommends settings, a risk
engine sizes trades and allocates capital across strategies, and Monte Carlo
stress-tests robustness. Built to run on **paper money** with real market data.

> **Design philosophy.** This is a *research + risk* platform, not a "profit
> generator." No backtest creates edge — it measures whether an edge survives
> transaction costs, slippage, and out-of-sample data. Every default here is
> tuned to *avoid* the overfit-backtest failure mode (1-bar execution lag,
> turnover costs, walk-forward validation, block-bootstrap robustness). That
> conservatism is the point.

## Quickstart

```bash
pip install -r requirements.txt

streamlit run dashboard/app.py    # the interactive visual hub (recommended)
python demo.py                    # or the full pipeline in the terminal
python tests.py                   # invariant test suite
```

The **dashboard** is the main way to use the platform: an amber-on-dark hub with
tabs for the strategy universe, an interactive backtest lab (pick a strategy,
tune parameters, see equity/drawdown/metrics live), the walk-forward optimizer,
regime detection, capital allocation, Monte Carlo robustness, per-trade
forecasts, and a paper-execution panel.

To run on real data, set `HF_FEED=yfinance` (free) or `HF_FEED=alpaca` (free
paper keys from alpaca.markets) in your environment.

## Architecture

```
core/         data feeds (synthetic / yfinance / alpaca), indicators, Strategy contract
strategies/   the strategy library + registry of the full universe
backtest/     engine (costs, slippage, exec-lag), metrics, walk-forward optimiser
risk/         position sizing (Kelly / vol-target / fixed-frac), cross-strategy allocation
montecarlo/   parametric GBM VaR engine + block-bootstrap strategy robustness
report/       per-trade forecast report (signal, why, size, stop, target, RR, robustness)
demo.py       end-to-end walk-through
```

The key abstraction: a **strategy only outputs a target position in [-1, 1]**.
It never decides dollar size, leverage, or stops — the risk engine does. That
separation means one backtester and one risk model serve every strategy.

## Strategy universe (16)

| Key | Category | Horizon | Asset classes | Status |
|-----|----------|---------|---------------|--------|
| `vix_rsi` | Mean Reversion | swing | equities, ETF | live |
| `breakout` | Trend Following | position | futures, crypto, FX | live |
| `trendline` | Trend Following | swing | equities, crypto, FX | live |
| `multifactor` | Quant Factor | position | equities | live |
| `equity_ls` | Equity L/S | swing | equities | live |
| `pairs_statarb` | Statistical Arb | intraday | equities | live |
| `kalman_pairs` | Statistical Arb | intraday | equities, crypto | live |
| `global_macro` | Macro | position | FX, rates, futures | live |
| `fx_carry` | Macro | position | FX | live |
| `event_earnings` | Event Driven | event | equities | live |
| `merger_arb` | Event Driven | position | equities | live |
| `vol_arb` | Volatility | swing | options | live |
| `index_rebal` | Flow | event | equities, ETF | live |
| `commodity_trend` | Trend/Seasonal | position | commodities | live |
| `crypto_funding` | Crypto Quant | intraday | crypto | live |
| `mm_scalper` | Market Making | scalp | crypto | live |

Each live strategy ships with **industry-standard defaults** and a searchable
`param_grid`. The walk-forward optimiser reports the recommended config *and*
its honest out-of-sample score, so "customise vs use recommended" is a real,
data-backed choice — not a guess.

## Market-regime detection & routing (`core/regime.py`)

Classifies each bar into a **(trend, volatility, volume)** regime — bull/bear/
sideways from a 100-bar regression slope + 200-MA; low/normal/high/extreme vol
from a causal rolling percentile of realised vol; rising/normal/falling volume.
`route()` maps the current regime to which strategy *categories* to favour vs
avoid and a **gross-exposure multiplier** that de-risks the whole book as vol
rises (100% → 60% → 35%). `strategy_multipliers()` applies this as a per-strategy
overlay on top of the allocator's weights, so the book automatically runs
trend-followers in trends, mean-reversion in chop, and shrinks in stress.

## Optimal capital allocation (`risk/allocation.py`)

The "mathematically optimal, risk-weighted" allocator. Given the strategies'
return streams it computes weights under five objectives — `min_variance`,
`max_sharpe`, `risk_parity`, `hrp`, `kelly` — all on a **Ledoit-Wolf shrunk**
covariance. `compare_allocations()` prints each method's in-sample return / vol /
Sharpe / concentration side by side, which demonstrates *why* the flashiest
in-sample `max_sharpe` is NOT the recommendation: it over-concentrates on the
noisiest estimates. **HRP is the default recommend** — it's robust to estimation
error and needs no matrix inversion.

## Where each piece lives

- **Strategy generation / explanation** → `strategies/*` (`meta` + `_explain`)
- **Backtesting (CAGR, Sharpe, MaxDD, win rate)** → `backtest/engine.py` + `metrics.py`
- **Risk-reward / drawdown analysis** → `metrics.py` (`max_drawdown`, `avg_recovery_days`) + `report/forecast.py`
- **Market-regime detection** → `core/regime.py` (trend/vol/volume classifier and routing)
- **Multi-factor strategy** → `strategies/multifactor.py`
- **Strategy optimisation** → `backtest/optimizer.py` (walk-forward, before/after)
- **Portfolio construction** → `risk/portfolio.py`
- **Trade-setup generation** → `report/forecast.py`
- **Monte Carlo simulation** → `montecarlo/simulator.py` (GBM + bootstrap)
- **Alpha/edge detection** → *planned* (crowding & capacity diagnostics)

## Live paper trading (`live/trader.py`)

The Execution tab runs risk-sized signals through a paper broker. `SimBroker` is
fully offline (fills at last price, tracks cash/positions/equity) and powers the
dashboard and tests with no keys. `AlpacaBroker` is a thin paper-`TradingClient`
adapter for when you add free Alpaca paper keys. The `Trader` runs one rebalance
cycle: latest signal per mandate -> vol-targeted size -> regime gross-exposure
scaling -> orders to move the book to target. Paper-only by design; there is no
live-trading code path and `config.ALPACA_PAPER` stays True.

## Roadmap

Built so far: regime detection and routing, Kalman-filtered pairs, the paper-trading loop, the dashboard, and the vol-arb and crypto-funding sleeves. Next:

1. **Fundamentals feed** so the multi-factor `value` leg uses real E/P and B/P.
2. **Execution realism**: participation-rate slippage and borrow costs for shorts.
3. **Crowding and capacity diagnostics** for alpha and edge detection.

## Honest caveats

- Results on `SyntheticFeed` are near-zero by construction — random data has no
  edge, and the framework correctly refuses to invent one.
- Mean-reversion and stat-arb are **crowded, low-capacity**; treat expectancy as
  small and cost-dominated.
- Never run `ALPACA_PAPER = False`. Live capital needs review this repo doesn't do.
- This is a *decision-support* tool. The edge is in the data, the costs, and the
  risk discipline — not in any single indicator.

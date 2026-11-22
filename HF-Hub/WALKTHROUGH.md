# HF-Hub — Walkthrough (how to actually use it)

This is the plain-English guide. Read the first two sections; the rest is reference.

---

## 1. The one thing to understand first: synthetic vs real data

Out of the box the platform runs on **synthetic (random) data**. You'll see this
in the top-left of the dashboard: `data feed: synthetic`.

Random data has **no real edge in it**. So when you see negative Sharpe ratios,
an 86% "probability of loss", or a "FRAGILE — do not deploy" verdict — that is
**not you losing money**. It's the platform *correctly refusing to invent profit
where none exists*. That honesty is the point: a tool that showed fake profits on
random data would be worse than useless.

**You are never trading real money.** The only execution path is a paper
(simulated) account.

To see results on **real market data** instead, close the app and relaunch with:

```bash
HF_FEED=yfinance streamlit run dashboard/app.py
```

Then type a real symbol (`AAPL`, `SPY`, `BTC-USD`) in the sidebar. This pulls free
daily data — no keys needed. Expect *modest* numbers even then: the simple
strategies (RSI, breakout) are the most crowded signals in the world, so they
won't mint money. The value here is the machinery and the discipline, not a money
printer. (I've been consistent about this from day one.)

---

## 2. Launching it

```bash
cd hf-hub
pip install -r requirements.txt      # one time
streamlit run dashboard/app.py       # opens in your browser
```

That's the whole setup. Everything runs offline on synthetic data by default.

---

## 3. The tabs, one by one

**🏠 Cockpit** — the TLDR: current regime, the recommended strategy for it, the
book's Sharpe/drawdown, and a LIVE SIGNALS table (every strategy's current
stance). Below it, the overview and how-to.

**💼 Fund** — the desk view: fund NAV, P&L attribution (which strategy made or
lost the money), VaR/CVaR risk panel, strategy correlation heatmap, the TRADE
BLOTTER (every historical trade: entry, exit, side, duration, P&L — including the
open one), a beta-hedge calculator ("how much SPY to short to be neutral"), a
trailing-stop/scale-out trade plan, a multi-asset scan (one strategy across a
basket of tickers), and a one-click downloadable daily fund report.

**🏠 Overview (below Cockpit)** — what the platform is, all 16 strategies grouped by type, and
this how-to. Start here.

**🧪 Strategies** — the heart of it. Pick any of the **16** strategies from the
dropdown. You get its equity curve, drawdown, key metrics (Sharpe, CAGR, max
drawdown, win rate), and a plain-English read of what it's signalling *right now*.
Single-asset strategies (vix_rsi, breakout, trendline, commodity_trend, vol_arb)
let you tune parameters live and watch the results change. The others (baskets,
pairs, macro, FX, events, crypto) run on their own appropriate datasets.

**🎛️ Optimizer** — pick a strategy, hit run. It searches the parameter grid using
*walk-forward* validation (tune on past data, score on unseen data) and recommends
the best settings — with the honest out-of-sample number, not an overfit one. This
is the "industry-standard vs optimal settings" feature you asked for.

**🌡️ Regime** — detects the current market regime (trend / volatility / volume),
shows which strategy types to favour or avoid right now, and how much to scale the
whole book's exposure (it de-risks automatically when volatility spikes).

**⚖️ Allocation** — how to split capital *across* strategies. Compares five methods
(HRP, risk-parity, min-variance, max-Sharpe, Kelly). **HRP is the recommended
default** — it's robust; max-Sharpe looks best in-sample but over-concentrates.
Also shows the regime-adjusted weights (allocation × regime overlay).

**🎲 Monte Carlo** — takes a strategy's (or the blended book's) returns and
resamples them thousands of times to ask: is this robust or fragile? Shows the
distribution of final outcomes, the drawdown distribution, and a verdict. On
synthetic data this says "fragile" — correctly.

**🔮 Forecast** — the per-trade report you asked for: direction, entry, stop,
target, reward:risk, position size (vol-targeted / Kelly / fixed), the reasoning,
and a robustness read. One card per trade.

**🚀 Execution** — the paper trader. Hit "Run one rebalance cycle" and it turns the
current signals into risk-sized orders in a simulated account, then shows your
equity, cash, positions, signals, and order log. Paper-only. To use Alpaca paper
keys later, set `ALPACA_KEY` / `ALPACA_SECRET` in your environment.

---

## 4. How to read the numbers (quick reference)

- **Sharpe** — return per unit of risk. >1 is good, >2 is excellent (and rare/
  suspicious on daily data). Negative = losing after risk.
- **Max drawdown** — worst peak-to-trough loss. −50% means it halved at some point.
- **Win rate** — % of trades that made money. High win rate + negative skew (many
  small wins, rare big losses) is normal for carry/vol strategies.
- **P(loss)** in Monte Carlo — chance of ending below where you started, across
  resampled paths. High = fragile.

---

## 5. Honest caveats (good interview talking points)

- Simple technical strategies (RSI, breakout) are maximally crowded; treat their
  edge as small and cost-sensitive.
- Vol-arb and merger-arb have **negative skew** — many small wins, occasional big
  losses. That's inherent; size them small.
- The market-making strategy is a *simulation* — real MM needs infrastructure this
  repo doesn't have. It's flagged as not deployable.
- Every "the fancy version doesn't beat the simple one on synthetic data" result
  is kept and explained, not hidden. That intellectual honesty is the point.

---

## 6. Where it goes next

- Swap `HF_FEED=yfinance` (or Alpaca paper keys) for real data.
- Real event/macro feeds to replace the synthetic generators in `core/altdata.py`.
- Dashboard polish (best done in Claude Code, which can iterate on the UI live).
- See `CLAUDE.md` for the full context and roadmap.

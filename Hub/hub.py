#hub.py

# QUANT PROJECTS HUB
# Central launcher for all projects and tools.
# Run with: streamlit run hub.py

import streamlit as st # type: ignore
import subprocess
import sys
import os
from pathlib import Path

st.set_page_config(
    page_title  = "Quant Hub",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #030608;
    color: #C0D8E8;
}
h1, h2, h3 { font-family: 'Share Tech Mono', monospace !important; }
h1 { color: #00FF88 !important; font-size: 2.2rem !important; letter-spacing: 0.05em; }
h2 { color: #00CCFF !important; font-size: 1.1rem !important; letter-spacing: 0.08em; }
h3 { color: #888 !important; font-size: 0.8rem !important; }

/* card grid */
.card {
    background: #0A1018;
    border: 1px solid #0E2030;
    border-radius: 6px;
    padding: 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}
.card:hover { border-color: #00FF88; }

/* status badges */
.badge-live    { background: #003820; color: #00FF88; padding: 2px 8px; border-radius: 3px; font-size: 0.65rem; font-family: 'Share Tech Mono'; }
.badge-wip     { background: #302000; color: #FFAA00; padding: 2px 8px; border-radius: 3px; font-size: 0.65rem; font-family: 'Share Tech Mono'; }
.badge-new     { background: #200030; color: #CC88FF; padding: 2px 8px; border-radius: 3px; font-size: 0.65rem; font-family: 'Share Tech Mono'; }

/* section dividers */
.section-header {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #0E4030;
    border-bottom: 1px solid #0E2030;
    padding-bottom: 6px;
    margin: 28px 0 16px 0;
    text-transform: uppercase;
}
hr { border-color: #0E2030 !important; }
[data-testid="stSidebar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown("# ⚡ QUANT PROJECTS HUB")
st.markdown(
    "<span style='font-family:Share Tech Mono;color:#336655;font-size:0.8rem;'>"
    "PRANSHU MUNDADA  ·  PYTHON QUANT STACK  ·  LIVE MARKET DATA  ·  16+ PROJECTS"
    "</span>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ── helper: open-in-terminal button ───────────────────────────────────────────
def project_card(
    icon:     str,
    name:     str,
    desc:     str,
    path:     str,      # folder name in the repo
    port:     int,
    status:   str = "live",   # live / wip / new
    tags:     list = None,
):
    badge = {
        "live": "<span class='badge-live'>● LIVE</span>",
        "wip":  "<span class='badge-wip'>⏳ WIP</span>",
        "new":  "<span class='badge-new'>★ NEW</span>",
    }.get(status, "")

    tag_html = " ".join(
        f"<span style='background:#0A1820;color:#336655;padding:1px 6px;border-radius:2px;font-size:0.6rem;font-family:Share Tech Mono;'>{t}</span>"
        for t in (tags or [])
    )

    st.markdown(f"""
<div class='card'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
    <div>
      <span style='font-size:1.3rem;'>{icon}</span>
      <span style='font-family:Share Tech Mono;font-size:0.95rem;color:#E0F0FF;margin-left:8px;'>{name}</span>
      {badge}
    </div>
    <div style='font-family:Share Tech Mono;color:#0E4030;font-size:0.65rem;'>:{port}</div>
  </div>
  <div style='color:#6A8898;font-size:0.8rem;margin:6px 0 8px 0;'>{desc}</div>
  <div>{tag_html}</div>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        if st.button(f"▶ Launch", key=f"launch_{port}"):
            run = "python main.py" if path == "Equity-Factor" else f"streamlit run app.py --server.port {port}"
            st.info(f"Run in terminal from the repo root:\n```\ncd '{path}'\n{run}\n```")
    with col2:
        st.markdown(
            f"<a href='http://localhost:{port}' target='_blank'>"
            f"<button style='background:#0A1820;color:#00FF88;border:1px solid #0E4030;border-radius:4px;padding:5px 14px;cursor:pointer;font-family:Share Tech Mono;font-size:0.75rem;'>↗ Open</button>"
            f"</a>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — NEW TOOLS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>▸ NEW  ·  TRADING & MACRO INTELLIGENCE</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    project_card(
        icon="📡", name="Active Trading Dashboard",
        desc="ICH+CCI signals, PSAR, RSI, EMA ribbon, 3-candle sniper, engulfing patterns. Your Mark I–IV rules coded as live signals. TP/SL calculator. Works on FX, crypto, commodities, indices.",
        path="Active-Trading", port=8510, status="new",
        tags=["ichimoku","CCI","PSAR","RSI","candlestick","FX","real-time"],
    )
    project_card(
        icon="📚", name="Encyclopedia & Calculator",
        desc="Every formula, indicator definition, and concept from all projects explained from scratch. Black-Scholes, Kelly Criterion, position sizing, Greeks, Ichimoku rules, ICT concepts, stochastic calculus.",
        path="Encyclopedia", port=8512, status="new",
        tags=["formulas","definitions","calculator","education"],
    )

with col2:
    project_card(
        icon="🌍", name="Macro Intelligence Dashboard",
        desc="FRED real yields, VIX term structure, CME FedWatch probabilities, DXY, credit spreads, put/call ratio, yield curve, global liquidity. Everything finance.thomas recommends.",
        path="Macro-Dashboard", port=8511, status="new",
        tags=["FRED","VIX","FedWatch","DXY","credit","macro"],
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SYSTEMATIC / QUANT RESEARCH
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>▸ SYSTEMATIC QUANT  ·  STRATEGIES & RESEARCH</div>", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    project_card(icon="🤖", name="Unified Signal Bot",
        desc="Ensemble of TSMOM, cross-sect momentum, vol regime, SMA trend, and sentiment signals. Vol-targeted position sizing. Drawdown stop. Alpaca paper trading execution.",
        path="Signal-Bot", port=8520, status="live",
        tags=["TSMOM","ensemble","vol-targeting","Alpaca"])
    project_card(icon="📡", name="CAPM / FF3 Regression",
        desc="CAPM and Fama-French 3-factor regressions. Alpha, beta, R², rolling estimates. Return decomposition. Multi-ticker comparison.",
        path="CAPM-Factor-Regression", port=8522, status="live",
        tags=["alpha","beta","Fama-French","OLS"])
    project_card(icon="⚡", name="vectorbt Backtest",
        desc="Professional momentum backtest with transaction costs, slippage, position sizing. Rolling Sharpe, underwater curve, parameter sweep.",
        path="Full-Backtest-with-Vectorbt", port=8524, status="live",
        tags=["vectorbt","momentum","backtest","costs"])
    project_card(icon="🔭", name="alphalens Factor Analysis",
        desc="IC, IC IR, quantile returns, factor decay, turnover. Industry standard Quantopian framework for evaluating alpha factors.",
        path="Alphalens-Factor-Analysis", port=8526, status="live",
        tags=["IC","alphalens","quantile","factor"])

with col4:
    project_card(icon="📈", name="Equity Factor Model",
        desc="Cross-sectional momentum, value, quality factors. Monthly long-short backtesting with Sharpe, Sortino, Calmar, drawdown, hit rate.",
        path="Equity-Factor", port=8521, status="live",
        tags=["momentum","factor","long-short","decile"])
    project_card(icon="⚖️", name="Pairs Trading",
        desc="Engle-Granger and Johansen cointegration. Z-score spread trading. Universe scanner for best cointegrated pairs. Half-life of mean reversion.",
        path="Relative-Value-Pairs-Trading-Model", port=8523, status="live",
        tags=["cointegration","stat-arb","z-score","mean-reversion"])
    project_card(icon="🌍", name="Macro Regime Model",
        desc="PMI + CPI → 4-quadrant Bridgewater classification. Asset allocation by regime. Yield curve recession indicator. FRED data.",
        path="Macro-Regime-Allocation-Model", port=8525, status="live",
        tags=["regime","Bridgewater","FRED","PMI","CPI"])
    project_card(icon="🔬", name="PCA Risk Factor Model",
        desc="PCA on stock universe → latent risk factors. Rolling PCA detects regime shifts. Factor loadings heatmap. Variance attribution.",
        path="Risk-Factor-and-Correlation-Model", port=8527, status="live",
        tags=["PCA","sklearn","factor-loadings","regime"])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RISK & PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>▸ RISK & PORTFOLIO</div>", unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    project_card(icon="🎲", name="Monte Carlo Simulator",
        desc="10,000 portfolio paths via GBM with Ito correction. VaR, CVaR, probability of ruin. Cholesky-correlated multi-asset. Vol target sweep.",
        path="Monte-Carlo-Portfolio-Simulator", port=8530, status="live",
        tags=["VaR","CVaR","GBM","Ito","Cholesky"])
    project_card(icon="📐", name="Efficient Frontier",
        desc="Markowitz mean-variance optimisation. Max Sharpe (tangency portfolio), Min Vol. Capital Market Line. Random feasible set.",
        path="Efficient-Frontier-Mean-Variance-Optimiser", port=8532, status="live",
        tags=["Markowitz","SLSQP","frontier","CML"])
    project_card(icon="💼", name="Portfolio Tracker",
        desc="Live P&L from holdings CSV. Sector breakdown, cumulative return vs SPY, drawdown, Sharpe. Refreshes every 5 minutes.",
        path="Python-Portfolio-Tracker", port=8534, status="live",
        tags=["live","P&L","sector","yfinance"])

with col6:
    project_card(icon="🌡️", name="Correlation Dashboard",
        desc="Rolling pairwise correlation with regime shift detection. Average correlation crisis indicator. Animated heatmap.",
        path="Correlation-and-Heatmap-Dashboard", port=8531, status="live",
        tags=["correlation","regime","heatmap","crisis"])
    project_card(icon="📊", name="TSMOM + Vol Targeting",
        desc="Time-series momentum (MOP 2012) with GARCH vol estimation and volatility targeting. Position = signal × (target_vol / realised_vol). Man AHL methodology.",
        path="Time-Series-Momentum-Model-(TSMOM)", port=8533, status="live",
        tags=["TSMOM","GARCH","vol-targeting","CTA"])

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRICING & FUNDAMENTALS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>▸ PRICING & FUNDAMENTALS</div>", unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    project_card(icon="📈", name="Options Pricer",
        desc="Black-Scholes-Merton. All 8 Greeks (Delta, Gamma, Theta, Vega, Rho, Vanna, Charm, Volga). Newton-Raphson IV solver. Strategy builder (Iron Condor, Butterfly, etc.).",
        path="Options-Pricer", port=8540, status="live",
        tags=["Black-Scholes","Greeks","IV","strategy"])
    project_card(icon="🧠", name="NLP Sentiment Signal",
        desc="FinBERT headline classification. IC evaluation. Keyword model: Fed + hike → SPY 5-day direction. Yahoo RSS, Reddit WSB, NewsAPI.",
        path="NLP-News-Sentiment", port=8542, status="new",
        tags=["FinBERT","NLP","IC","sentiment","alpha"])

with col8:
    project_card(icon="📑", name="Financial Statement Analyser",
        desc="SEC EDGAR 10-K/10-Q data. Piotroski F-Score, Altman Z-Score, revenue growth, margin trends. LLM MD&A analysis (Claude). Peer comparison.",
        path="Financial-Statement-Analyser", port=8541, status="new",
        tags=["SEC","EDGAR","Piotroski","Altman","LLM"])

# ═══════════════════════════════════════════════════════════════════════════════
# QUICK REFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("<div class='section-header'>▸ QUICK LAUNCH  ·  COPY-PASTE COMMANDS</div>", unsafe_allow_html=True)

with st.expander("Show all launch commands"):
    commands = """
# from the repo root, with your venv active
(cd "Active-Trading" && streamlit run app.py --server.port 8510) &
(cd "Encyclopedia" && streamlit run app.py --server.port 8512) &
(cd "Macro-Dashboard" && streamlit run app.py --server.port 8511) &
(cd "Signal-Bot" && streamlit run app.py --server.port 8520) &
(cd "CAPM-Factor-Regression" && streamlit run app.py --server.port 8522) &
(cd "Full-Backtest-with-Vectorbt" && streamlit run app.py --server.port 8524) &
(cd "Alphalens-Factor-Analysis" && streamlit run app.py --server.port 8526) &
(cd "Equity-Factor" && python main.py)   # command-line project, no dashboard
(cd "Relative-Value-Pairs-Trading-Model" && streamlit run app.py --server.port 8523) &
(cd "Macro-Regime-Allocation-Model" && streamlit run app.py --server.port 8525) &
(cd "Risk-Factor-and-Correlation-Model" && streamlit run app.py --server.port 8527) &
(cd "Monte-Carlo-Portfolio-Simulator" && streamlit run app.py --server.port 8530) &
(cd "Efficient-Frontier-Mean-Variance-Optimiser" && streamlit run app.py --server.port 8532) &
(cd "Python-Portfolio-Tracker" && streamlit run app.py --server.port 8534) &
(cd "Correlation-and-Heatmap-Dashboard" && streamlit run app.py --server.port 8531) &
(cd "Time-Series-Momentum-Model-(TSMOM)" && streamlit run app.py --server.port 8533) &
(cd "Options-Pricer" && streamlit run app.py --server.port 8540) &
(cd "NLP-News-Sentiment" && streamlit run app.py --server.port 8542) &
(cd "Financial-Statement-Analyser" && streamlit run app.py --server.port 8541) &
"""
    st.code(commands, language="bash")

with st.expander("Launch script — run all at once"):
    st.code("""#!/bin/bash
# launch_all.sh: start every dashboard, then the Hub
# Usage (from the repo root): chmod +x launch_all.sh && ./launch_all.sh

ROOT="$(cd "$(dirname "$0")" && pwd)"
[ -f "$ROOT/venv/bin/activate" ] && source "$ROOT/venv/bin/activate"

while IFS="|" read -r dir port; do
  (cd "$ROOT/$dir" && streamlit run app.py --server.port "$port" --server.headless true) &
  echo "Started: $dir on :$port"
done <<'EOF'
Active-Trading|8510
Encyclopedia|8512
Macro-Dashboard|8511
Signal-Bot|8520
CAPM-Factor-Regression|8522
Full-Backtest-with-Vectorbt|8524
Alphalens-Factor-Analysis|8526
Relative-Value-Pairs-Trading-Model|8523
Macro-Regime-Allocation-Model|8525
Risk-Factor-and-Correlation-Model|8527
Monte-Carlo-Portfolio-Simulator|8530
Efficient-Frontier-Mean-Variance-Optimiser|8532
Python-Portfolio-Tracker|8534
Correlation-and-Heatmap-Dashboard|8531
Time-Series-Momentum-Model-(TSMOM)|8533
Options-Pricer|8540
NLP-News-Sentiment|8542
Financial-Statement-Analyser|8541
EOF

echo "Hub: http://localhost:8550"
cd "$ROOT/Hub" && streamlit run hub.py --server.port 8550
""", language="bash")

# footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-family:Share Tech Mono;font-size:0.65rem;color:#0E3020;'>"
    "PRANSHU MUNDADA  ·  QUANT PROJECTS  ·  BUILT WITH PYTHON + STREAMLIT  ·  DATA: YAHOO FINANCE · FRED · SEC EDGAR"
    "</div>",
    unsafe_allow_html=True,
)
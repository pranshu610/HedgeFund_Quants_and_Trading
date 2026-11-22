# Quant Projects

Python projects in systematic trading, factor research, risk modelling and portfolio construction. I built them to apply what I study (BSc Finance, Durham; CFA Level I candidate) to real market data.

Most projects run as an interactive Streamlit dashboard, and many also have a command-line interface. Data comes from free public sources: Yahoo Finance, FRED, SEC EDGAR, Ken French's Data Library and NewsAPI.


---

## Flagship: HF-Hub

**[HF-Hub](./HF-Hub)** is a multi-strategy research and paper-trading platform built the way a multi-manager fund is organised. Strategies only output a target position; a separate risk engine sizes every trade and allocates capital across strategies.

- **16 strategies** across trend, mean reversion, stat arb (including Kalman-filtered pairs), macro, event-driven, volatility and crypto.
- **Cost-aware backtester** with a one-bar execution lag, turnover costs and slippage, plus walk-forward optimisation that reports the out-of-sample score, not the in-sample one.
- **Regime detection** (trend, volatility, volume) that scales gross exposure down as volatility rises.
- **Capital allocation** across strategies: minimum variance, max Sharpe, risk parity, HRP and Kelly, on a Ledoit-Wolf shrunk covariance. HRP is the default.
- **Post-mortem loop** that diagnoses why a strategy lost money and only recommends a fix if it improves out-of-sample results.
- **Paper execution** through an offline broker or Alpaca paper trading.
- **145+ tests**, all passing.

```bash
cd HF-Hub && pip install -r requirements.txt && streamlit run dashboard/app.py
```

---

## Projects

### Systematic strategies
| Project | What it does | Key methods |
|---|---|---|
| [Time-Series Momentum (TSMOM)](./Time-Series-Momentum-Model-%28TSMOM%29) | Replicates Moskowitz, Ooi & Pedersen (2012) on a multi-asset ETF universe, with volatility-targeted position sizing | GARCH(1,1) vol forecasting, vol targeting, leverage caps, vol-target sweep |
| [Relative Value Pairs Trading](./Relative-Value-Pairs-Trading-Model) | Finds cointegrated pairs and trades the mean-reverting spread, with a universe scanner | Engle-Granger, Johansen, OLS hedge ratio, z-score signals, Ornstein-Uhlenbeck half-life |
| [Macro Regime Allocation](./Macro-Regime-Allocation-Model) | Sorts the economy into four growth/inflation regimes and allocates across asset classes | ISM PMI and CPI from FRED, All Weather-style quadrants, regime backtest |
| [Vectorbt Momentum Backtest](./Full-Backtest-with-Vectorbt) | 12-1 cross-sectional momentum backtest net of costs and slippage | vectorbt, inverse-vol sizing, underwater curve, 16-cell lookback × cost sensitivity grid |
| [Unified Signal Bot](./Signal-Bot) | Ensemble of five signals, then a risk layer, then paper-trade execution | Signal blending, vol targeting, drawdown stop, Alpaca paper-trading API |

### Factor research
| Project | What it does | Key methods |
|---|---|---|
| [Equity Factor Model](./Equity-Factor) | Ranks stocks on momentum, value, quality and volatility factors and backtests long-short portfolios (S&P 500 option). Command line. | Decile portfolios, Sharpe / Sortino / Calmar, rolling Sharpe |
| [Alphalens Factor Analysis](./Alphalens-Factor-Analysis) | Evaluates factors the alphalens way | Information Coefficient, IC IR, quantile returns, IC decay, turnover |
| [CAPM / Fama-French Regression](./CAPM-Factor-Regression) | Splits stock returns into alpha and factor exposures | OLS via `(X'X)⁻¹X'y`, rolling 24-month beta and alpha, FF3 attribution |
| [PCA Risk Factor Model](./Risk-Factor-and-Correlation-Model) | Extracts latent risk factors from a stock universe | PCA, factor loadings, rolling PC1 dominance, variance attribution |
| [NLP News Sentiment](./NLP-News-Sentiment) | Turns financial headlines into a sentiment signal and tests whether it predicts returns | FinBERT, z-scored daily sentiment, IC vs 1- and 5-day forward returns, logistic keyword model |

### Risk & portfolio construction
| Project | What it does | Key methods |
|---|---|---|
| [Monte Carlo Simulator](./Monte-Carlo-Portfolio-Simulator) | Simulates 10,000 correlated portfolio paths | GBM with Itô correction, Cholesky correlation, VaR, CVaR, probability of ruin |
| [Efficient Frontier](./Efficient-Frontier-Mean-Variance-Optimiser) | Markowitz mean-variance optimisation | SLSQP, max-Sharpe and min-vol portfolios, Capital Market Line |
| [Correlation Dashboard](./Correlation-and-Heatmap-Dashboard) | Tracks how correlations shift across market regimes | Rolling correlation, crisis-regime flags, dispersion |
| [Options Pricer](./Options-Pricer) | Black-Scholes-Merton pricing engine | First- and second-order Greeks, Newton-Raphson implied vol, strategy payoffs, no-arbitrage bounds |
| [Portfolio Tracker](./Python-Portfolio-Tracker) | Live P&L and risk for a holdings file | Sector exposure, drawdown, beta and alpha vs SPY |

### Fundamentals & tools
| Project | What it does | Key methods |
|---|---|---|
| [Financial Statement Analyser](./Financial-Statement-Analyser) | Pulls 10-K data from SEC EDGAR and scores companies | Piotroski F-Score, Altman Z-Score, margin trends, LLM summary of MD&A |
| [Macro Dashboard](./Macro-Dashboard) | Macro monitor | FRED series, VIX term structure, credit spreads, real yields |
| [Active Trading Dashboard](./Active-Trading) | Charting tool with a technical-indicator library and rule backtests | 15+ indicators (Ichimoku, Supertrend, ADX and others), candlestick patterns |
| [Quant Encyclopedia](./Encyclopedia) | Reference and calculator for formulas, Greeks and risk metrics | Kelly, duration, DCF, Sharpe/Sortino calculators |
| [Hub](./Hub) | One page listing every dashboard with its launch command | — |

### Tests
Run with `python tests.py` in each folder (`pytest tests.py` for the Options Pricer). All use synthetic data, so no internet is needed.

| Project | Tests |
|---|---|
| HF-Hub | 145+ passing |
| CAPM / Fama-French Regression | 17 passing |
| Monte Carlo Simulator | 17 passing |
| Efficient Frontier | 16 passing |
| Options Pricer | 14 passing |

---

## In progress
| Project | Goal |
|---|---|
| Earnings Call Sentiment Analyser | Use an LLM to score management tone on earnings call transcripts by topic (guidance, margins, capex, demand), track how it shifts quarter to quarter, and test whether tone changes predict post-earnings returns. Starting with energy companies. |
| Multi-factor alpha model | Combine momentum, value, quality and low-vol into one IC-weighted composite score |
| Execution cost model | Estimate market impact, spread cost and implementation shortfall |

---

## Stack
| Layer | Tools |
|---|---|
| Data | yfinance, FRED, SEC EDGAR, Ken French Data Library, NewsAPI |
| Quant | pandas, NumPy, SciPy, statsmodels, arch, scikit-learn, vectorbt, alphalens |
| NLP | HuggingFace Transformers (FinBERT) |
| Visualisation | Plotly, Matplotlib, Streamlit |
| Execution | Alpaca API (paper trading) |

---

## Running a project
```bash
git clone https://github.com/pranshu610/HedgeFund_Quants_and_Trading.git
cd HedgeFund_Quants_and_Trading

python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt      # everything, from the repo root

cd Monte-Carlo-Portfolio-Simulator   # any project folder
streamlit run app.py
```

To see every dashboard and its launch command on one page:
```bash
cd Hub && streamlit run hub.py
```

---

*Pranshu Mundada · BSc Finance, Durham University · [GitHub](https://github.com/pranshu610)*

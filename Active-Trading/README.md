# Active Trading Dashboard

A charting tool for discretionary trading: a clean price chart that you add indicators to, plus position sizing and a simple rule backtest.

## What it does
- **Chart:** candlesticks for equities, ETFs, FX pairs and gold, on daily or weekly bars, with any indicator overlaid.
- **Indicator library** (`signals.py`): SMA, EMA, WMA, RSI, CCI, Williams %R, stochastic, MACD, Bollinger Bands, ATR, VWAP, Supertrend, Parabolic SAR, Ichimoku, MFI, ADX and Fibonacci levels, plus engulfing, doji and hammer candlestick patterns.
- **TP/SL & Position Size tab:** sizes a trade from account size, risk per trade and an ATR-based stop.
- **Active Signals tab:** lists the signals currently firing.
- **Backtest tab:** tests an indicator rule on history.

## Run it
```bash
pip install streamlit yfinance pandas numpy plotly
streamlit run app.py
```
Data: Yahoo Finance via yfinance.

# Macro Dashboard

A one-page macro monitor for checking the backdrop before a trade or a pitch.

## Tabs
- **Yield curve and rates:** 10Y-2Y spread (inversion as a recession warning), 10Y nominal vs real yield, Fed funds.
- **VIX term structure:** VIX9D, VIX and VIX3M, flagged as contango or backwardation.
- **Credit spreads:** high-yield spreads as a risk-appetite gauge.
- **DXY and global liquidity.**
- **Economic indicators:** CPI, unemployment, ISM PMI.
- **Cross-asset:** how rates, the dollar, equities and commodities are moving together.

## Run it
```bash
pip install streamlit yfinance pandas numpy plotly requests
streamlit run app.py
```
Data: FRED (public CSV downloads, no API key needed) and Yahoo Finance.

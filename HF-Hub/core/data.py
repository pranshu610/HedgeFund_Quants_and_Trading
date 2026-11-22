"""
core/data.py
------------
Unified market-data layer. Every strategy and backtest pulls bars through a
DataFeed so the rest of the codebase never cares whether the data came from
yfinance, Alpaca, a crypto exchange, or a synthetic generator.

A "bar" DataFrame is always: index = tz-aware DatetimeIndex, columns =
['open','high','low','close','volume'] (lowercase). Extra columns (e.g. 'vix')
are allowed and passed through.

Live/real feeds require network + keys, so they are thin adapters. The synthetic
feed lets the whole platform run and be tested with zero external dependencies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"DataFeed returned bars missing columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


class DataFeed(ABC):
    """Contract for any source of OHLCV bars."""

    @abstractmethod
    def history(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        ...


class SyntheticFeed(DataFeed):
    """
    Geometric-Brownian-motion price generator with an optional regime switch and
    a synthetic VIX-like series. Deterministic given `seed`. Used for tests, demos,
    and offline development — NOT for real signals.
    """

    def __init__(self, seed: int | None = None, annual_drift: float = 0.08,
                 annual_vol: float = 0.20):
        self.seed = seed
        self.mu = annual_drift
        self.sigma = annual_vol

    def history(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        rng = np.random.default_rng(self.seed if self.seed is not None else hash(symbol) % 2**32)
        idx = pd.bdate_range(start=start, end=end)
        n = len(idx)
        dt = 1 / 252
        # regime-switching vol so mean-reversion / breakout strategies have something to bite on
        regime = np.where(np.sin(np.linspace(0, 12, n)) > 0.3, 1.8, 0.7)
        daily_vol = self.sigma * np.sqrt(dt) * regime
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        shocks = rng.standard_normal(n)
        log_ret = drift + daily_vol * shocks
        close = 100 * np.exp(np.cumsum(log_ret))
        # build OHLC around close
        intraday = np.abs(rng.standard_normal(n)) * daily_vol * close
        high = close + intraday
        low = close - intraday
        open_ = np.r_[close[0], close[:-1]]
        vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
        # synthetic VIX: inversely related to returns, mean ~18, spikes on drawdowns
        vix = 18 + 60 * np.clip(-log_ret, 0, None) * 20 + rng.normal(0, 1.5, n)
        vix = pd.Series(vix, index=idx).rolling(3, min_periods=1).mean().clip(9, 80)
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": vol, "vix": vix.values},
            index=idx,
        )
        return _validate(df)


class FallbackFeed(DataFeed):
    """Tries several feeds in order, returns the first with data. Lets you type ANY
    ticker: Alpaca serves US stocks + major crypto (and is the trading venue), and
    yfinance covers everything else (altcoins like TAO-USD, global stocks, indices,
    FX, commodities). `last_source` records which feed answered."""

    def __init__(self, feeds):
        self.feeds = feeds              # [(name, feed_instance), ...]
        self.last_source = None

    def history(self, symbol, start, end, interval="1d"):
        errors = []
        for name, feed in self.feeds:
            try:
                df = feed.history(symbol, start, end, interval=interval)
                if df is not None and len(df) > 0:
                    self.last_source = name
                    return df
            except Exception as e:
                errors.append(f"{name}: {str(e)[:50]}")
        raise ValueError(f"No feed had data for '{symbol}'. Tried → " + " | ".join(errors))


class YFinanceFeed(DataFeed):
    """Free daily/intraday data, no key. Requires `pip install yfinance` + network."""

    def history(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        try:
            import yfinance as yf  # lazy import so repo runs without it
        except ImportError:
            raise ImportError(
                "yfinance is not installed. Run `pip install yfinance` in the SAME "
                "environment you launch Streamlit from.")
        df = yf.download(symbol, start=start, end=end, interval=interval,
                         auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {symbol}")
        # yfinance sometimes returns MultiIndex columns for single tickers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return _validate(df)


class AlpacaFeed(DataFeed):
    """
    Alpaca market data — free with a (paper) account. Handles both equities and
    crypto, and daily + intraday bars. Keys read from env by config; NEVER hardcode.
    Crypto symbols use the 'BTC/USD' form; equities use 'AAPL'. This adapter maps
    common yfinance-style tickers (BTC-USD) to Alpaca's form automatically.
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None):
        self.api_key = api_key
        self.secret_key = secret_key

    @staticmethod
    def _is_crypto(symbol: str) -> bool:
        # Alpaca crypto pairs use 'BTC/USD'; yfinance-style is 'BTC-USD'. US equities
        # never end in -USD, so a -USD suffix (or a slash) reliably means crypto.
        s = symbol.upper()
        return "/" in s or s.endswith("-USD") or s.endswith("USDT")

    @staticmethod
    def _to_alpaca_crypto(symbol: str) -> str:
        s = symbol.upper().replace("-USD", "/USD")
        return s if "/" in s else f"{s}/USD"

    @staticmethod
    def _timeframe(interval: str):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        table = {
            "1d": TimeFrame.Day, "1h": TimeFrame.Hour, "1m": TimeFrame.Minute,
            "5m": TimeFrame(5, TimeFrameUnit.Minute), "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
        }
        return table.get(interval, TimeFrame.Day)

    @staticmethod
    def _require():
        try:
            import alpaca  # noqa
        except ImportError:
            raise ImportError(
                "alpaca-py is not installed. Run `pip install alpaca-py` in the SAME "
                "environment you launch Streamlit from (e.g. your miniconda env).")

    def history(self, symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
        self._require()
        tf = self._timeframe(interval)
        if self._is_crypto(symbol):
            from alpaca.data.historical import CryptoHistoricalDataClient
            from alpaca.data.requests import CryptoBarsRequest
            client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
            sym = self._to_alpaca_crypto(symbol)
            req = CryptoBarsRequest(symbol_or_symbols=sym, timeframe=tf, start=start, end=end)
            bars = client.get_crypto_bars(req).df
        else:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            client = StockHistoricalDataClient(self.api_key, self.secret_key)
            req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf, start=start, end=end)
            bars = client.get_stock_bars(req).df
        if bars is None or len(bars) == 0:
            raise ValueError(f"Alpaca returned no data for {symbol}")
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.reset_index(level=0, drop=True)
        return _validate(bars)

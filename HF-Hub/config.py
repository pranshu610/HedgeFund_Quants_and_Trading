"""
config.py
---------
Central config. Keys are read from environment variables so you never commit
secrets. Paper-trading keys from Alpaca are free: https://alpaca.markets
"""
import os

# --- data / execution ---
FEED = os.getenv("HF_FEED", "synthetic")          # "synthetic" | "yfinance" | "alpaca"
ALPACA_KEY = os.getenv("ALPACA_KEY", "")
ALPACA_SECRET = os.getenv("ALPACA_SECRET", "")
ALPACA_PAPER = True                                # NEVER flip to live without review

# --- backtest cost model (basis points, per side) ---
COST_BPS = 1.0
SLIPPAGE_BPS = 2.0

# --- risk defaults ---
INITIAL_CAPITAL = 100_000
TARGET_VOL_ANN = 0.15
MAX_LEVERAGE = 3.0
MAX_STRATEGY_WEIGHT = 0.40

# --- default backtest window ---
START = "2015-01-01"
END = "2025-01-01"


def make_feed():
    from core.data import SyntheticFeed, YFinanceFeed, AlpacaFeed, FallbackFeed
    import importlib.util
    _has_yf = importlib.util.find_spec("yfinance") is not None
    if FEED == "yfinance":
        return YFinanceFeed()
    if FEED == "alpaca":
        alpaca = AlpacaFeed(ALPACA_KEY, ALPACA_SECRET)
        # smart fallback: Alpaca for US stocks + major crypto, yfinance for the rest
        # (altcoins like TAO-USD, global/India stocks, indices, FX, commodities)
        if _has_yf:
            return FallbackFeed([("alpaca", alpaca), ("yfinance", YFinanceFeed())])
        return alpaca
    # seed=None -> each symbol gets its own deterministic series (hash of name)
    return SyntheticFeed(seed=None)


def alpaca_ready() -> bool:
    """True if paper keys are present in the environment."""
    return bool(ALPACA_KEY and ALPACA_SECRET)


def _pkg_installed(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


def feed_status() -> dict:
    """Diagnose the selected feed so the UI can guide setup instead of crashing."""
    if FEED == "synthetic":
        return {"feed": "synthetic", "ok": True, "hint": "Offline random data (no setup needed)."}
    if FEED == "yfinance":
        ok = _pkg_installed("yfinance")
        return {"feed": "yfinance", "ok": ok,
                "hint": "Free real data." if ok else "Run: pip install yfinance"}
    if FEED == "alpaca":
        pkg = _pkg_installed("alpaca")
        keys = alpaca_ready()
        ok = pkg and keys
        if not pkg:
            hint = "Run: pip install alpaca-py"
        elif not keys:
            hint = "Set ALPACA_KEY and ALPACA_SECRET env vars (paper keys)."
        else:
            hint = "Alpaca paper data + trading ready."
        return {"feed": "alpaca", "ok": ok, "pkg": pkg, "keys": keys, "hint": hint}
    return {"feed": FEED, "ok": False, "hint": "Unknown feed."}


def default_symbol() -> str:
    """A sensible default ticker for the active feed (DEMO only works on synthetic)."""
    return "DEMO" if FEED == "synthetic" else "AAPL"


def make_broker(capital: float = INITIAL_CAPITAL, use_alpaca: bool = False):
    """SimBroker by default (offline paper). AlpacaBroker only if keys exist AND
    explicitly requested. Always paper — no live path."""
    from live.trader import SimBroker, AlpacaBroker
    if use_alpaca and alpaca_ready():
        return AlpacaBroker(ALPACA_KEY, ALPACA_SECRET)
    return SimBroker(cash=capital)

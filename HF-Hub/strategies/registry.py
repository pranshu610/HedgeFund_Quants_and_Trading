"""
strategies/registry.py
----------------------
Central registry of every strategy in the hub. `REGISTRY` maps a key to the
class; `UNIVERSE` documents the full intended roster (built + planned) so the
dashboard/README can render the whole map and you can see what's live vs TODO.
"""
from strategies.vix_rsi import VixRsiMeanReversion
from strategies.breakout import DonchianBreakout
from strategies.trendline import TrendlineFollow
from strategies.multifactor import MultiFactorEquity
from strategies.pairs_statarb import PairsStatArb
from strategies.kalman_pairs import KalmanPairs
from strategies.equity_ls import ResidualMomentumLS
from strategies.global_macro import GlobalMacro
from strategies.fx_carry import FXCarry
from strategies.vol_arb import VolArb
from strategies.commodity_trend import CommodityTrend
from strategies.crypto_funding import CryptoFunding
from strategies.event_earnings import EarningsDrift
from strategies.merger_arb import MergerArb
from strategies.index_rebal import IndexRebal
from strategies.mm_scalper import MMScalper
from strategies.ichimoku import Ichimoku
from strategies.cci_strategy import CCIStrategy
from strategies.convergence import FixedIncomeArb, IndexArb
from strategies.classic_algos import MACD, BollingerRev, OvernightDrift

# --- LIVE strategies (single-series, backtestable today) ---
REGISTRY = {
    "vix_rsi": VixRsiMeanReversion,
    "breakout": DonchianBreakout,
    "trendline": TrendlineFollow,
    "commodity_trend": CommodityTrend,
    "ichimoku": Ichimoku,
    "cci": CCIStrategy,
    "macd": MACD,
    "bollinger_rev": BollingerRev,
    "overnight_drift": OvernightDrift,
}
# panel/cross-sectional strategies use generate_panel(bars)
PANEL_REGISTRY = {
    "multifactor": MultiFactorEquity,
    "equity_ls": ResidualMomentumLS,
}
# pairs strategies take TWO price series (generate_pair)
PAIR_REGISTRY = {
    "pairs_statarb": PairsStatArb,
    "kalman_pairs": KalmanPairs,
    "fixed_income_arb": FixedIncomeArb,
    "index_arb": IndexArb,
}
# advanced strategies with bespoke inputs (macro dicts / rate panels / event tables)
ADVANCED_REGISTRY = {
    "global_macro": GlobalMacro,     # generate_macro(macro_df, assets_dict)
    "fx_carry": FXCarry,             # generate_carry(spots_df, rates_df)
    "vol_arb": VolArb,               # generate(df) on a synthetic variance book
    "crypto_funding": CryptoFunding, # generate(funding_series)
    "event_earnings": EarningsDrift, # generate(df, events_df)
    "merger_arb": MergerArb,         # book_returns(deals_df, calendar)
    "index_rebal": IndexRebal,       # generate(df, events_df)
    "mm_scalper": MMScalper,         # simulate(df)
}

# --- FULL intended universe: the definitive 16-strategy roster ---
# status: "live" | "planned". horizon: scalp/intraday/swing/position/event
UNIVERSE = [
    # -- LIVE (backtestable today) --
    ("vix_rsi",         "Mean Reversion",  "swing",    ["equities", "etf"],           "live"),
    ("breakout",        "Trend Following", "position", ["futures", "crypto", "fx"],   "live"),
    ("trendline",       "Trend Following", "swing",    ["equities", "crypto", "fx"],  "live"),
    ("multifactor",     "Quant Factor",    "position", ["equities"],                  "live"),
    # -- PLANNED (roadmap, in build order) --
    ("equity_ls",       "Equity L/S",      "swing",    ["equities"],                  "live"),
    ("pairs_statarb",   "Statistical Arb", "intraday", ["equities"],                  "live"),
    ("kalman_pairs",    "Statistical Arb", "intraday", ["equities", "crypto"],        "live"),
    ("fixed_income_arb","Fixed Income Arb","swing",    ["rates", "bonds", "etf"],     "live"),
    ("index_arb",       "Index Arb",       "intraday", ["equities", "etf", "futures"],"live"),
    ("global_macro",    "Macro",           "position", ["fx", "rates", "futures"],    "live"),
    ("fx_carry",        "Macro",           "position", ["fx"],                        "live"),
    ("event_earnings",  "Event Driven",    "event",    ["equities"],                  "live"),
    ("merger_arb",      "Event Driven",    "position", ["equities"],                  "live"),
    ("vol_arb",         "Volatility",      "swing",    ["options"],                   "live"),
    ("index_rebal",     "Flow",            "event",    ["equities", "etf"],           "live"),
    ("commodity_trend", "Trend/Seasonal",  "position", ["commodities", "futures"],    "live"),
    ("ichimoku",        "Trend Following", "swing",    ["equities", "crypto", "fx"],  "live"),
    ("cci",             "Mean Reversion",  "intraday", ["equities", "crypto", "fx"],  "live"),
    ("macd",            "Trend Following", "swing",    ["equities", "crypto", "fx"],  "live"),
    ("bollinger_rev",   "Mean Reversion",  "swing",    ["equities", "fx", "crypto"],  "live"),
    ("overnight_drift", "Event / flow",    "overnight",["equities", "etf"],           "live"),
    ("crypto_funding",  "Crypto Quant",    "intraday", ["crypto"],                    "live"),
    ("mm_scalper",      "Market Making",   "scalp",    ["crypto"],                     "live"),
]
assert len(UNIVERSE) == 23, "roster must stay at 23 strategies"


def get(name: str):
    if name in REGISTRY:
        return REGISTRY[name]
    if name in PANEL_REGISTRY:
        return PANEL_REGISTRY[name]
    if name in PAIR_REGISTRY:
        return PAIR_REGISTRY[name]
    if name in ADVANCED_REGISTRY:
        return ADVANCED_REGISTRY[name]
    raise KeyError(f"Unknown strategy '{name}'. Known: "
                   f"{list(REGISTRY) + list(PANEL_REGISTRY) + list(PAIR_REGISTRY) + list(ADVANCED_REGISTRY)}")

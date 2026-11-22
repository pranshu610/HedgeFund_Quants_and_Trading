"""
live/trader.py
--------------
Paper-trading loop. Turns strategy signals into risk-sized orders and (paper-)
executes them. Two brokers behind one interface:

  SimBroker   — fully offline paper broker (fills at last price, tracks cash /
                positions / equity). Default, no keys, powers the dashboard and
                tests without network.
  AlpacaBroker — thin wrapper over alpaca-py's paper TradingClient. Lazy-imported;
                 needs free paper keys. `config.ALPACA_PAPER` MUST stay True.

The Trader orchestrates one rebalance cycle: for each (strategy, symbol) mandate
it pulls the latest bars, gets the strategy's latest signal, vol-targets the
position, scales by the current regime's gross-exposure, then submits orders to
move the current book to target. Everything is logged for the dashboard.

SAFETY: there is no live-trading code path here. Live execution is deliberately
omitted; flipping Alpaca to a live account is out of scope for this repo.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd

from core.indicators import realized_vol
from core.regime import latest_regime, route
from risk.sizing import vol_target_size
from risk.manager import RiskManager
from strategies.registry import REGISTRY


@dataclass
class Proposal:
    strategy: str
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    price: float
    notional: float
    reason: str
    risk_note: str
    approved: bool


@dataclass
class Order:
    ts: pd.Timestamp
    symbol: str
    side: str            # "buy" | "sell"
    qty: float
    price: float
    strategy: str
    status: str = "filled"


@dataclass
class Position:
    symbol: str
    qty: float = 0.0
    avg_price: float = 0.0


# --------------------------------------------------------------------- brokers
class SimBroker:
    """Offline paper broker. Deterministic, no network."""

    def __init__(self, cash: float = 100_000.0):
        self.start_cash = cash
        self.cash = cash
        self.positions: dict[str, Position] = {}
        self.orders: list[Order] = []
        self._prices: dict[str, float] = {}

    def mark(self, prices: dict[str, float]):
        self._prices.update(prices)

    def last_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def equity(self) -> float:
        mv = sum(p.qty * self._prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + mv

    def position_qty(self, symbol: str) -> float:
        return self.positions.get(symbol, Position(symbol)).qty

    def submit(self, symbol: str, side: str, qty: float, strategy: str, ts) -> Order:
        px = self.last_price(symbol)
        if qty <= 0 or px <= 0:
            return Order(ts, symbol, side, 0, px, strategy, "rejected")
        signed = qty if side == "buy" else -qty
        pos = self.positions.setdefault(symbol, Position(symbol))
        new_qty = pos.qty + signed
        # update average price on adds; realise cash on the trade
        if pos.qty == 0 or np.sign(signed) == np.sign(pos.qty):
            total_cost = pos.avg_price * abs(pos.qty) + px * abs(signed)
            pos.avg_price = total_cost / max(abs(new_qty), 1e-9)
        pos.qty = new_qty
        self.cash -= signed * px
        order = Order(ts, symbol, side, qty, px, strategy)
        self.orders.append(order)
        return order


class AlpacaBroker:
    """Paper TradingClient wrapper (paper=True ALWAYS). Needs alpaca-py + paper keys
    from env. Pulls live quotes from the Alpaca data client so run_once/propose work
    end-to-end against the real paper account."""

    def __init__(self, api_key: str, secret_key: str):
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        self.client = TradingClient(api_key, secret_key, paper=True)   # paper ONLY
        self._data = StockHistoricalDataClient(api_key, secret_key)
        self._api_key, self._secret = api_key, secret_key
        self.orders: list[Order] = []
        self._price_cache: dict[str, float] = {}

    def mark(self, prices: dict):
        self._price_cache.update(prices)

    def last_price(self, symbol: str) -> float:
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            q = self._data.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
            px = float(q[symbol].ask_price or q[symbol].bid_price)
            self._price_cache[symbol] = px
            return px
        except Exception:
            return self._price_cache.get(symbol, 0.0)

    def equity(self) -> float:
        try:
            return float(self.client.get_account().equity)
        except Exception:
            return 0.0

    @property
    def cash(self) -> float:
        try:
            return float(self.client.get_account().cash)
        except Exception:
            return 0.0

    @property
    def positions(self) -> dict:
        out = {}
        try:
            for p in self.client.get_all_positions():
                out[p.symbol] = Position(p.symbol, float(p.qty), float(p.avg_entry_price))
        except Exception:
            pass
        return out

    def position_qty(self, symbol: str) -> float:
        try:
            return float(self.client.get_open_position(symbol).qty)
        except Exception:
            return 0.0

    def submit(self, symbol: str, side: str, qty: float, strategy: str, ts) -> Order:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        try:
            req = MarketOrderRequest(
                symbol=symbol, qty=round(abs(qty), 4),
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY)
            o = self.client.submit_order(req)
            order = Order(ts, symbol, side, abs(qty), self.last_price(symbol), strategy, str(o.status))
        except Exception as e:
            order = Order(ts, symbol, side, abs(qty), 0.0, strategy, f"rejected: {str(e)[:40]}")
        self.orders.append(order)
        return order


# --------------------------------------------------------------------- trader
@dataclass
class Mandate:
    strategy_key: str
    symbol: str


class Trader:
    """One rebalance cycle turns signals → risk-sized target book → orders."""

    def __init__(self, broker, feed, mandates: list[Mandate], capital: float = 100_000,
                 target_vol: float = 0.15, max_leverage: float = 2.0,
                 lookback_start: str = "2015-01-01", lookback_end: str = "2025-01-01"):
        self.broker = broker
        self.feed = feed
        self.mandates = mandates
        self.capital = capital
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.start, self.end = lookback_start, lookback_end
        self.last_signals: list[dict] = []
        self.risk = RiskManager(capital=capital, max_gross=max_leverage)

    def _target_shares(self, m: Mandate, df: pd.DataFrame, gross: float, n: int) -> tuple[float, dict]:
        cls = REGISTRY[m.strategy_key]
        strat = cls()
        sig = strat.latest_signal(df, m.symbol)
        price = float(df["close"].iloc[-1])
        rv = float(realized_vol(df["close"], 20).iloc[-1])
        lev = vol_target_size(rv, self.target_vol, self.max_leverage)
        base_w = 1.0 / max(n, 1)
        notional = self.capital * base_w * gross * lev * sig.direction
        shares = notional / price if price > 0 else 0.0
        info = dict(strategy=cls.meta.name, symbol=m.symbol, direction=sig.direction,
                    conviction=round(sig.strength, 2), price=round(price, 2),
                    target_shares=round(shares, 2), leverage=round(lev, 2),
                    reason=sig.reason)
        return shares, info

    def run_once(self, asof: pd.Timestamp | None = None) -> dict:
        """Pull data, compute targets, diff vs book, submit orders. Returns a report.
        Symbols that fail to load are skipped (not fatal)."""
        # load every mandate's data; drop any that error (e.g. bad ticker on a feed)
        dfs, prices, good = {}, {}, []
        errors = []
        for m in self.mandates:
            try:
                d = self.feed.history(m.symbol, self.start, self.end)
                if len(d) < 60:
                    raise ValueError("insufficient history")
                dfs[m.symbol] = d
                prices[m.symbol] = float(d["close"].iloc[-1])
                good.append(m)
            except Exception as e:
                errors.append(f"{m.symbol}: {e}")
        if not good:
            return {"regime": "n/a", "gross_exposure": 0.0, "orders": [],
                    "signals": [], "equity": self.broker.equity() if hasattr(self.broker, "equity") else 0.0,
                    "errors": errors}

        reg = latest_regime(dfs[good[0].symbol])
        gross = route(reg)["gross_exposure"]
        ts = asof or dfs[good[0].symbol].index[-1]
        if hasattr(self.broker, "mark"):
            self.broker.mark(prices)

        submitted, self.last_signals = [], []
        n = len(good)
        # NET target exposure per symbol across all strategies trading it, then diff
        # ONCE per symbol. Without this, several strategies on one ticker each diff
        # against the shared broker position and thrash against each other.
        net_target: dict[str, float] = {}
        who: dict[str, list] = {}
        for m in good:
            target, info = self._target_shares(m, dfs[m.symbol], gross, n)
            self.last_signals.append(info)
            net_target[m.symbol] = net_target.get(m.symbol, 0.0) + target
            who.setdefault(m.symbol, []).append(info["strategy"])
        for sym, target in net_target.items():
            current = self.broker.position_qty(sym)
            delta = target - current
            if abs(delta * prices[sym]) < 0.01 * self.capital / max(n, 1):
                continue
            side = "buy" if delta > 0 else "sell"
            tag = " + ".join(sorted(set(who[sym])))[:60]
            submitted.append(self.broker.submit(sym, side, abs(delta), tag, ts))

        return {"regime": reg.label(), "gross_exposure": gross,
                "orders": submitted, "signals": self.last_signals,
                "equity": self.broker.equity(), "errors": errors}

    def propose(self, asof: pd.Timestamp | None = None) -> list[Proposal]:
        """ANALYST MODE: compute the trades it WANTS to do, risk-check them, but do
        NOT execute. The PM reviews these 'heads-up' proposals and confirms."""
        proposals = []
        dfs, prices, good = {}, {}, []
        for m in self.mandates:
            try:
                d = self.feed.history(m.symbol, self.start, self.end)
                if len(d) < 60:
                    continue
                dfs[m.symbol] = d
                prices[m.symbol] = float(d["close"].iloc[-1])
                good.append(m)
            except Exception:
                continue
        if not good:
            return []
        if hasattr(self.broker, "mark"):
            self.broker.mark(prices)
        reg = latest_regime(dfs[good[0].symbol])
        gross = route(reg)["gross_exposure"]
        cur_gross = sum(abs(getattr(p, "qty", 0) * prices.get(s, 0))
                        for s, p in getattr(self.broker, "positions", {}).items())
        n = len(good)
        net_target: dict[str, float] = {}
        why: dict[str, list] = {}
        names: dict[str, list] = {}
        for m in good:
            target, info = self._target_shares(m, dfs[m.symbol], gross, n)
            net_target[m.symbol] = net_target.get(m.symbol, 0.0) + target
            if abs(target) > 1e-9:
                why.setdefault(m.symbol, []).append(f"{info['strategy']}: {info['reason']}")
                names.setdefault(m.symbol, []).append(info["strategy"])
        for sym, target in net_target.items():
            current = self.broker.position_qty(sym)
            delta = target - current
            px = prices[sym]
            if abs(delta * px) < 0.01 * self.capital / max(n, 1):
                continue
            side = "buy" if delta > 0 else "sell"
            notional = abs(delta) * px
            strat_names = names.get(sym, ["net"])
            flags = {"short_vol"} if any("vol arb" in s_.lower() for s_ in strat_names) else set()
            dec = self.risk.check_order(sym, notional * (1 if side == "buy" else -1),
                                        cur_gross, is_entry=(current == 0), strategy_flags=flags)
            approved_qty = abs(dec.qty) / px if px > 0 else 0.0
            n_agree = len(strat_names)
            reason = (f"**{n_agree} strategy(ies) net to this position.** "
                      + " | ".join(why.get(sym, ["no active signal"])))
            proposals.append(Proposal(
                strategy=" + ".join(sorted(set(strat_names)))[:60], symbol=sym, side=side,
                qty=round(min(abs(delta), approved_qty), 2), price=round(px, 2),
                notional=round(min(notional, abs(dec.qty)), 2),
                reason=reason, risk_note=dec.reason, approved=dec.approved))
        return proposals

    def execute(self, proposals: list[Proposal], asof: pd.Timestamp | None = None) -> list[Order]:
        """PM MODE: execute the confirmed proposals (paper)."""
        ts = asof or pd.Timestamp.now()
        submitted = []
        for p in proposals:
            if not p.approved or p.qty <= 0:
                continue
            submitted.append(self.broker.submit(p.symbol, p.side, p.qty, p.strategy, ts))
        self.risk.update_equity(self.broker.equity())
        return submitted

    def snapshot(self) -> dict:
        pos = getattr(self.broker, "positions", {})
        return {
            "equity": self.broker.equity() if hasattr(self.broker, "equity") else None,
            "cash": getattr(self.broker, "cash", None),
            "positions": [asdict(p) for p in pos.values()] if pos else [],
            "orders": [asdict(o) for o in getattr(self.broker, "orders", [])][-20:],
            "signals": self.last_signals,
        }


def default_mandates(symbols=None) -> list[Mandate]:
    """Mandates for the tickers YOU are looking at — never a hardcoded list.
    For each symbol we attach the strategies that are actually valid for its asset
    class (an equity never gets commodity seasonality). Pass a list of tickers;
    with none supplied we return an empty book so nothing phantom is proposed."""
    from dashboard.engine import relevant_strategies
    if not symbols:
        return []
    if isinstance(symbols, str):
        symbols = [symbols]
    out = []
    for sym in symbols:
        for key in list(relevant_strategies(sym))[:4]:   # a few valid sleeves per name
            out.append(Mandate(key, sym))
    return out

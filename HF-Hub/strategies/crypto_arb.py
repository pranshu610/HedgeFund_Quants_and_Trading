"""
strategies/crypto_arb.py
------------------------
Execution-grade crypto arbitrage engine.

Most "arb scanners" compare last-traded prices across venues, find a 0.4% gap,
and declare free money. That gap is almost always fake. It dies to four things,
and this engine models all four:

  1. DEPTH      — you don't trade at the top of book. Size walks the ladder, so the
                  effective price is a volume-weighted average, not the best quote.
  2. FEES       — taker fees both legs (typically 2-10bps each), plus withdrawal /
                  transfer cost for spatial arb.
  3. LATENCY    — between seeing the quote and being filled, the edge decays. We
                  model decay as a function of round-trip latency and pair vol.
  4. INVENTORY  — spatial arb needs capital pre-positioned on BOTH venues, or you
                  eat transfer time (minutes-hours) during which the edge is gone.

Three arb types:
  * SPATIAL     — same asset, two venues. Buy cheap venue, sell rich venue.
  * TRIANGULAR  — one venue, three pairs (A/B, B/C, A/C). Cycle for a risk-free loop.
  * BASIS       — perp vs spot on the same venue (cash-and-carry, funding harvest).

The output is not "spread = 0.4%". It's NET EDGE AFTER EVERYTHING, in bps, plus the
max executable size at that edge, plus a decay-adjusted confidence. If net edge is
negative — which it usually is — the engine says so loudly. That honesty is the
whole point: a scanner that shows gross spreads is a machine for losing money
slowly.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np
import pandas as pd


# --------------------------------------------------------------------- fees
# Realistic taker fee schedule (bps). Override per venue as needed.
DEFAULT_TAKER_BPS = {
    "binance": 10.0, "coinbase": 60.0, "kraken": 26.0, "okx": 10.0,
    "bybit": 10.0, "kucoin": 10.0, "bitstamp": 40.0, "gemini": 40.0,
}
DEFAULT_MAKER_BPS = {k: max(0.0, v - 8.0) for k, v in DEFAULT_TAKER_BPS.items()}

# Typical on-chain / transfer cost in bps of notional for moving inventory.
DEFAULT_TRANSFER_BPS = {"BTC": 3.0, "ETH": 5.0, "SOL": 0.5, "USDT": 1.0, "USDC": 1.0}


@dataclass
class BookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    """Minimal L2 book. bids descending, asks ascending."""
    venue: str
    symbol: str
    bids: list                      # list[BookLevel]
    asks: list
    ts: pd.Timestamp | None = None

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else np.nan

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else np.nan

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.best_ask - self.best_bid) / m * 1e4 if m > 0 else np.nan

    def sweep(self, side: str, qty: float) -> tuple[float, float]:
        """Walk the book for `qty` units. Returns (vwap, filled_qty).

        THIS is why naive scanners are wrong: the price you get for real size is
        never the top-of-book price.
        """
        levels = self.asks if side == "buy" else self.bids
        remaining, notional, filled = qty, 0.0, 0.0
        for lv in levels:
            take = min(remaining, lv.size)
            notional += take * lv.price
            filled += take
            remaining -= take
            if remaining <= 1e-12:
                break
        if filled <= 0:
            return (np.nan, 0.0)
        return (notional / filled, filled)

    def depth_to_bps(self, side: str, bps: float) -> float:
        """How much size sits within `bps` of mid — the honest capacity measure."""
        m = self.mid
        if not np.isfinite(m):
            return 0.0
        limit = m * (1 + bps / 1e4) if side == "buy" else m * (1 - bps / 1e4)
        levels = self.asks if side == "buy" else self.bids
        return float(sum(lv.size for lv in levels
                         if (lv.price <= limit if side == "buy" else lv.price >= limit)))


@dataclass
class ArbOpportunity:
    kind: str                       # spatial | triangular | basis
    symbol: str
    buy_venue: str
    sell_venue: str
    gross_bps: float                # naive top-of-book spread — what bad scanners show
    net_bps: float                  # AFTER depth, fees, transfer, latency decay
    size: float                     # max executable size at that net edge
    notional_usd: float
    fees_bps: float
    slippage_bps: float
    transfer_bps: float
    latency_decay_bps: float
    confidence: float               # 0-1, decays with latency vs pair volatility
    half_life_ms: float             # how long this edge survives, estimated
    legs: list = field(default_factory=list)
    verdict: str = ""
    why: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("legs", None)
        return d


# ------------------------------------------------------------------ engine
class CryptoArbEngine:
    """Scans books for executable arbitrage. Every number is net of reality."""

    def __init__(self, taker_bps: dict | None = None, transfer_bps: dict | None = None,
                 latency_ms: float = 250.0, min_net_bps: float = 1.0,
                 max_notional_usd: float = 50_000.0,
                 competition_base_ms: float = 1500.0,
                 competition_scale_bps: float = 8.0):
        self.taker = {**DEFAULT_TAKER_BPS, **(taker_bps or {})}
        self.transfer = {**DEFAULT_TRANSFER_BPS, **(transfer_bps or {})}
        self.latency_ms = latency_ms
        self.min_net_bps = min_net_bps
        self.max_notional_usd = max_notional_usd
        # competition model: how fast rival bots take a visible edge
        self.competition_base_ms = competition_base_ms
        self.competition_scale_bps = competition_scale_bps

    # -- latency model -------------------------------------------------
    def edge_half_life_ms(self, vol_annual: float, gross_bps: float) -> float:
        """How long an edge of `gross_bps` survives.

        TWO forces kill it, and honest models need both:

        1. DIFFUSION — price drifts and the gap closes on its own.
           t_diff ~ (edge/sigma_ms)^2.
        2. COMPETITION — this is the one that actually matters in crypto. You are
           racing colocated bots. A *wider* gap is MORE visible and gets taken
           FASTER: competitive half-life falls as the edge grows. Modelled as
           t_comp = base / (1 + (edge_bps / scale)^2), so a 5bps gap may live a
           second while a 50bps gap is gone in milliseconds — because everyone
           sees it.

        Effective half-life is the harmonic-style combination (fastest killer wins).
        """
        ms_per_year = 365 * 24 * 3600 * 1000
        sigma_ms = vol_annual / np.sqrt(ms_per_year)
        edge = max(gross_bps, 1e-6) / 1e4
        t_diff = (edge / sigma_ms) ** 2 if sigma_ms > 0 else 60_000.0
        # competition: base survival at low competition, crushed as the edge widens
        t_comp = self.competition_base_ms / (1 + (gross_bps / self.competition_scale_bps) ** 2)
        t = 1.0 / (1.0 / max(t_diff, 1e-6) + 1.0 / max(t_comp, 1e-6))
        return float(np.clip(t, 2.0, 300_000.0))

    def latency_decay(self, vol_annual: float, gross_bps: float) -> tuple[float, float]:
        """Returns (bps lost to latency, confidence in [0,1])."""
        hl = self.edge_half_life_ms(vol_annual, gross_bps)
        surviving = 0.5 ** (self.latency_ms / hl)
        return (gross_bps * (1 - surviving), float(np.clip(surviving, 0.0, 1.0)))

    # -- spatial -------------------------------------------------------
    def scan_spatial(self, books: list, vol_annual: float = 0.6,
                     pre_positioned: bool = True) -> list:
        """Same asset across venues. `pre_positioned=True` means you already hold
        inventory on both sides (a real desk does) so you skip transfer cost."""
        out = []
        for i, b_buy in enumerate(books):
            for j, b_sell in enumerate(books):
                if i == j or b_buy.symbol != b_sell.symbol:
                    continue
                gross = (b_sell.best_bid - b_buy.best_ask) / b_buy.best_ask * 1e4
                if not np.isfinite(gross) or gross <= 0:
                    continue

                base = b_buy.symbol.split("/")[0].split("-")[0].upper()
                fees = self.taker.get(b_buy.venue, 10.0) + self.taker.get(b_sell.venue, 10.0)
                transfer = 0.0 if pre_positioned else self.transfer.get(base, 5.0)
                decay_bps, conf = self.latency_decay(vol_annual, gross)

                # size: the largest qty where the edge still clears costs after depth
                size = self._max_executable(b_buy, b_sell, fees + transfer + decay_bps)
                if size <= 0:
                    # REJECTED — still report it, priced at a nominal clip, so the
                    # scanner explains why the "obvious" gap is not real money.
                    probe = min(sum(l.size for l in b_buy.asks),
                                sum(l.size for l in b_sell.bids)) * 0.10
                    if probe <= 0:
                        continue
                    vb, _fb = b_buy.sweep("buy", probe)
                    vs, _fs = b_sell.sweep("sell", probe)
                    if not (np.isfinite(vb) and np.isfinite(vs)):
                        continue
                    realized = (vs - vb) / vb * 1e4
                    net = realized - fees - transfer - decay_bps
                    out.append(ArbOpportunity(
                        kind="spatial", symbol=b_buy.symbol,
                        buy_venue=b_buy.venue, sell_venue=b_sell.venue,
                        gross_bps=round(gross, 2), net_bps=round(net, 2), size=0.0,
                        notional_usd=0.0, fees_bps=round(fees, 2),
                        slippage_bps=round(gross - realized, 2), transfer_bps=round(transfer, 2),
                        latency_decay_bps=round(decay_bps, 2), confidence=round(conf, 3),
                        half_life_ms=round(self.edge_half_life_ms(vol_annual, gross), 1),
                        legs=[], verdict="❌ not executable",
                        why=(f"Top-of-book shows {gross:.1f}bps — looks like free money. It is not: "
                             f"{fees:.1f}bps fees"
                             + (f" + {transfer:.1f}bps transfer" if transfer else "")
                             + f" + {gross - realized:.1f}bps depth slippage + {decay_bps:.1f}bps "
                               f"latency decay = NET {net:+.1f}bps. No size clears the cost floor.")))
                    continue
                vwap_buy, fb = b_buy.sweep("buy", size)
                vwap_sell, fs = b_sell.sweep("sell", size)
                if not (np.isfinite(vwap_buy) and np.isfinite(vwap_sell)) or min(fb, fs) <= 0:
                    continue
                realized = (vwap_sell - vwap_buy) / vwap_buy * 1e4
                slip = gross - realized
                net = realized - fees - transfer - decay_bps
                notional = size * vwap_buy

                out.append(ArbOpportunity(
                    kind="spatial", symbol=b_buy.symbol,
                    buy_venue=b_buy.venue, sell_venue=b_sell.venue,
                    gross_bps=round(gross, 2), net_bps=round(net, 2), size=round(size, 6),
                    notional_usd=round(notional, 2), fees_bps=round(fees, 2),
                    slippage_bps=round(slip, 2), transfer_bps=round(transfer, 2),
                    latency_decay_bps=round(decay_bps, 2), confidence=round(conf, 3),
                    half_life_ms=round(self.edge_half_life_ms(vol_annual, gross), 1),
                    legs=[{"action": "BUY", "venue": b_buy.venue, "px": round(vwap_buy, 2), "qty": round(size, 6)},
                          {"action": "SELL", "venue": b_sell.venue, "px": round(vwap_sell, 2), "qty": round(size, 6)}],
                    verdict=self._verdict(net, conf),
                    why=(f"Top-of-book gap {gross:.1f}bps. Sweeping {size:.4f} units costs "
                         f"{slip:.1f}bps of depth slippage, {fees:.1f}bps fees"
                         + (f", {transfer:.1f}bps transfer" if transfer else "")
                         + f", {decay_bps:.1f}bps latency decay at {self.latency_ms:.0f}ms "
                           f"(edge half-life {self.edge_half_life_ms(vol_annual, gross):.0f}ms). "
                         + (f"NET {net:+.1f}bps — executable." if net >= self.min_net_bps
                            else f"NET {net:+.1f}bps — NOT executable; the gap is an illusion."))))
        return sorted(out, key=lambda o: o.net_bps, reverse=True)

    def _max_executable(self, b_buy: OrderBook, b_sell: OrderBook, cost_bps: float,
                        steps: int = 24) -> float:
        """Largest size whose depth-adjusted edge still beats costs. Binary-ish search
        over the achievable ladder — capacity is the metric retail scanners never show."""
        cap_qty = min(sum(l.size for l in b_buy.asks), sum(l.size for l in b_sell.bids))
        if cap_qty <= 0:
            return 0.0
        notional_cap = self.max_notional_usd / max(b_buy.best_ask, 1e-9)
        hi = min(cap_qty, notional_cap)
        lo, best = 0.0, 0.0
        for _ in range(steps):
            mid = (lo + hi) / 2
            if mid <= 0:
                break
            vb, fb = b_buy.sweep("buy", mid)
            vs, fs = b_sell.sweep("sell", mid)
            if not (np.isfinite(vb) and np.isfinite(vs)) or min(fb, fs) < mid * 0.999:
                hi = mid
                continue
            edge = (vs - vb) / vb * 1e4
            if edge - cost_bps >= self.min_net_bps:
                best, lo = mid, mid
            else:
                hi = mid
        return best

    # -- triangular ----------------------------------------------------
    def scan_triangular(self, books_by_symbol: dict, venue: str,
                        base: str = "BTC", quote: str = "USDT", bridge: str = "ETH",
                        notional_usd: float = 10_000.0, vol_annual: float = 0.6) -> list:
        """One venue, three pairs. Cycle: quote -> base -> bridge -> quote.
        Returns the loop's net edge after three taker fees and three depth sweeps."""
        p1, p2, p3 = f"{base}/{quote}", f"{bridge}/{base}", f"{bridge}/{quote}"
        need = [p1, p2, p3]
        if any(s not in books_by_symbol for s in need):
            return []
        b1, b2, b3 = (books_by_symbol[s] for s in need)
        fee = self.taker.get(venue, 10.0)
        out = []
        for direction in ("forward", "reverse"):
            try:
                if direction == "forward":
                    q0 = notional_usd
                    qty_base = q0 / b1.best_ask
                    vw1, _ = b1.sweep("buy", qty_base)
                    qty_base = q0 / vw1
                    qty_bridge = qty_base / b2.best_ask
                    vw2, _ = b2.sweep("buy", qty_bridge)
                    qty_bridge = qty_base / vw2
                    vw3, _ = b3.sweep("sell", qty_bridge)
                    q1 = qty_bridge * vw3
                    legs = [{"action": "BUY", "pair": p1, "px": round(vw1, 4)},
                            {"action": "BUY", "pair": p2, "px": round(vw2, 6)},
                            {"action": "SELL", "pair": p3, "px": round(vw3, 4)}]
                else:
                    q0 = notional_usd
                    qty_bridge = q0 / b3.best_ask
                    vw3, _ = b3.sweep("buy", qty_bridge)
                    qty_bridge = q0 / vw3
                    vw2, _ = b2.sweep("sell", qty_bridge)
                    qty_base = qty_bridge * vw2
                    vw1, _ = b1.sweep("sell", qty_base)
                    q1 = qty_base * vw1
                    legs = [{"action": "BUY", "pair": p3, "px": round(vw3, 4)},
                            {"action": "SELL", "pair": p2, "px": round(vw2, 6)},
                            {"action": "SELL", "pair": p1, "px": round(vw1, 4)}]
                if not np.isfinite(q1) or q1 <= 0:
                    continue
                gross = (q1 / q0 - 1) * 1e4
                fees = 3 * fee
                decay_bps, conf = self.latency_decay(vol_annual, max(gross, 0.1))
                net = gross - fees - decay_bps
                out.append(ArbOpportunity(
                    kind="triangular", symbol=f"{quote}→{base}→{bridge}→{quote} ({direction})",
                    buy_venue=venue, sell_venue=venue,
                    gross_bps=round(gross, 2), net_bps=round(net, 2),
                    size=round(notional_usd, 2), notional_usd=round(notional_usd, 2),
                    fees_bps=round(fees, 2), slippage_bps=0.0, transfer_bps=0.0,
                    latency_decay_bps=round(decay_bps, 2), confidence=round(conf, 3),
                    half_life_ms=round(self.edge_half_life_ms(vol_annual, max(gross, 0.1)), 1),
                    legs=legs, verdict=self._verdict(net, conf),
                    why=(f"Loop returns {gross:+.1f}bps gross after depth sweeps. Three taker fees "
                         f"cost {fees:.1f}bps — that alone kills most triangles. "
                         f"NET {net:+.1f}bps.")))
            except Exception:
                continue
        return sorted(out, key=lambda o: o.net_bps, reverse=True)

    # -- basis / cash-and-carry ---------------------------------------
    def scan_basis(self, spot: OrderBook, perp: OrderBook, funding_rate: float,
                   funding_interval_h: float = 8.0, hold_days: float = 7.0,
                   vol_annual: float = 0.6) -> ArbOpportunity | None:
        """Perp vs spot. Edge = basis convergence + funding collected over the hold,
        net of entry AND exit fees on both legs."""
        basis_bps = (perp.mid - spot.mid) / spot.mid * 1e4
        periods = hold_days * 24 / funding_interval_h
        funding_bps = funding_rate * periods * 1e4
        fees = 2 * (self.taker.get(spot.venue, 10.0) + self.taker.get(perp.venue, 10.0))
        size = min(spot.depth_to_bps("buy", 10), perp.depth_to_bps("sell", 10))
        if size <= 0:
            return None
        vb, _ = spot.sweep("buy", size)
        vs, _ = perp.sweep("sell", size)
        realized_basis = (vs - vb) / vb * 1e4
        gross = realized_basis + funding_bps
        net = gross - fees
        conf = float(np.clip(1 - abs(basis_bps) / 500, 0.3, 0.95))
        return ArbOpportunity(
            kind="basis", symbol=spot.symbol, buy_venue=spot.venue, sell_venue=perp.venue,
            gross_bps=round(gross, 2), net_bps=round(net, 2), size=round(size, 6),
            notional_usd=round(size * vb, 2), fees_bps=round(fees, 2),
            slippage_bps=round(basis_bps - realized_basis, 2), transfer_bps=0.0,
            latency_decay_bps=0.0, confidence=round(conf, 3), half_life_ms=float(hold_days * 86400e3),
            legs=[{"action": "BUY SPOT", "venue": spot.venue, "px": round(vb, 2)},
                  {"action": "SHORT PERP", "venue": perp.venue, "px": round(vs, 2)}],
            verdict=self._verdict(net, conf),
            why=(f"Basis {realized_basis:+.1f}bps + funding {funding_bps:+.1f}bps over {hold_days:.0f}d "
                 f"({periods:.0f} payments at {funding_rate*100:.4f}%) = {gross:+.1f}bps gross. "
                 f"Round-trip fees on both legs cost {fees:.1f}bps. NET {net:+.1f}bps. "
                 f"Delta-neutral: no directional crypto exposure."))

    def _verdict(self, net_bps: float, conf: float) -> str:
        if net_bps < 0:
            return "❌ negative after costs"
        if net_bps < self.min_net_bps:
            return "⚠️ below minimum edge"
        if conf < 0.5:
            return "⚠️ edge decays faster than your latency"
        if net_bps >= 5 and conf >= 0.7:
            return "✅ executable"
        return "⚠️ thin but positive"

    # -- convenience ---------------------------------------------------
    def scan_all(self, books: list, vol_annual: float = 0.6,
                 pre_positioned: bool = True) -> pd.DataFrame:
        opps = self.scan_spatial(books, vol_annual, pre_positioned)
        if not opps:
            return pd.DataFrame()
        return pd.DataFrame([o.to_row() for o in opps])


# ------------------------------------------------------- book construction
def book_from_quotes(venue: str, symbol: str, mid: float, spread_bps: float,
                     depth_usd: float, levels: int = 12, decay: float = 0.72,
                     seed: int | None = None) -> OrderBook:
    """Synthesise a realistic L2 book: geometric size decay away from touch, which
    is what real books look like. Used for offline testing and demo."""
    rng = np.random.default_rng(seed)
    half = mid * spread_bps / 2e4
    bids, asks = [], []
    # Real books are THIN at the touch and thicken outward. Top level holds only a
    # small slice of total depth, so any meaningful size walks the ladder and pays
    # slippage — which is exactly what naive scanners ignore.
    total_qty = depth_usd / mid
    weights = np.array([(1.0 - decay) * (decay ** i) for i in range(levels)])
    weights = weights[::-1]                     # thin at touch, thick away
    weights = weights / weights.sum()
    for i in range(levels):
        step = half + mid * (i * spread_bps / 2e4) * 1.4
        sz = total_qty * weights[i] * (1 + 0.12 * rng.standard_normal())
        sz = max(sz, 1e-9)
        bids.append(BookLevel(round(mid - step, 6), round(sz, 8)))
        asks.append(BookLevel(round(mid + step, 6), round(sz, 8)))
    return OrderBook(venue=venue, symbol=symbol, bids=bids, asks=asks,
                     ts=pd.Timestamp.utcnow())

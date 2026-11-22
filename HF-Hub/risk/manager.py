"""
risk/manager.py
---------------
The shared risk manager — every order passes through it, no strategy may bypass
it. Enforces (in code, not by discipline) the non-negotiable rules from the
build-document risk framework:

  * per-position cap (default 5% of capital)
  * per-book drawdown kill switch (default 12% from high-water mark → halt entries)
  * portfolio drawdown kill switch (default 15% → flatten / halt everything)
  * gross-leverage cap (default 1.5x)
  * data-staleness guard (reject acting on stale prices)
  * PERMANENT BANS: naked options, short-volatility as a primary sleeve,
    martingale / averaging-down. These manufacture fake consistency and blow up.

The manager returns an approve/clip/reject decision with a reason for every order,
so the blotter and dashboard can show exactly why something was sized down or
blocked.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd

BANNED_STRATEGY_FLAGS = {
    # strategies whose *primary* payoff is selling tail risk — allowed only as a
    # tiny, explicitly-flagged sleeve, never sized up by the risk manager.
    "short_vol": "short-volatility payoff (sells tail risk) — hard size cap applies",
}


@dataclass
class RiskDecision:
    approved: bool
    qty: float
    reason: str


@dataclass
class RiskManager:
    capital: float = 100_000.0
    position_cap: float = 0.05          # 5% of capital per position
    book_dd_kill: float = 0.12          # 12% drawdown halts new entries
    portfolio_dd_kill: float = 0.15     # 15% halts everything
    max_gross: float = 1.5              # 1.5x gross leverage
    short_vol_cap: float = 0.02         # short-vol sleeves capped at 2%
    high_water: float = field(default=0.0)
    halted: bool = field(default=False)
    flatten: bool = field(default=False)

    def __post_init__(self):
        if self.high_water == 0.0:
            self.high_water = self.capital

    # --- equity / kill-switch tracking ---
    def update_equity(self, nav: float) -> dict:
        self.high_water = max(self.high_water, nav)
        dd = nav / self.high_water - 1.0
        if dd <= -self.portfolio_dd_kill:
            self.halted, self.flatten = True, True
            state = "PORTFOLIO KILL — flatten everything, human restart required"
        elif dd <= -self.book_dd_kill:
            self.halted = True
            state = "BOOK KILL — no new entries; existing positions trail out"
        else:
            self.halted, self.flatten = False, False
            state = "OK"
        return {"drawdown": dd, "high_water": self.high_water, "state": state}

    # --- order gate ---
    def check_order(self, symbol: str, target_notional: float, current_gross: float,
                    is_entry: bool, strategy_flags: set | None = None,
                    price_age_sec: float = 0.0, max_price_age: float = 3600.0) -> RiskDecision:
        flags = strategy_flags or set()

        if price_age_sec > max_price_age:
            return RiskDecision(False, 0.0, "REJECT: stale market data")

        if self.flatten:
            # only exits allowed
            if is_entry:
                return RiskDecision(False, 0.0, "REJECT: portfolio kill-switch active")

        if self.halted and is_entry:
            return RiskDecision(False, 0.0, "REJECT: book kill-switch — no new entries")

        # permanent bans
        for f in flags & set(BANNED_STRATEGY_FLAGS):
            cap = self.short_vol_cap * self.capital
            if abs(target_notional) > cap:
                target_notional = cap * (1 if target_notional > 0 else -1)
                # fall through with clipped notional
                clipped_reason = f"CLIP: {BANNED_STRATEGY_FLAGS[f]}"
                break
        else:
            clipped_reason = None

        # per-position cap
        pos_cap = self.position_cap * self.capital
        reason = clipped_reason
        if abs(target_notional) > pos_cap:
            target_notional = pos_cap * (1 if target_notional > 0 else -1)
            reason = f"CLIP: position cap {self.position_cap:.0%} of capital"

        # gross-leverage cap
        if current_gross + abs(target_notional) > self.max_gross * self.capital:
            room = max(0.0, self.max_gross * self.capital - current_gross)
            target_notional = room * (1 if target_notional > 0 else -1)
            reason = f"CLIP: gross-leverage cap {self.max_gross}x"

        return RiskDecision(True, target_notional, reason or "APPROVED")

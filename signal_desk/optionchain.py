from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Any, Mapping


def _f(v: Any) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


@dataclasses.dataclass
class Leg:
    ltp: float | None = None
    bid: float | None = None
    ask: float | None = None
    oi: float = 0.0
    prev_oi: float | None = None
    volume: float = 0.0
    iv: float | None = None
    delta: float | None = None
    theta: float | None = None

    @property
    def oi_change(self) -> float | None:
        return None if self.prev_oi is None else self.oi - self.prev_oi


@dataclasses.dataclass
class Row:
    strike: float
    ce: Leg
    pe: Leg


@dataclasses.dataclass
class Chain:
    expiry: str
    spot: float
    rows: list[Row]
    fetched_at: dt.datetime | None = None

    def row(self, strike: float) -> Row | None:
        for r in self.rows:
            if abs(r.strike - strike) < 0.01:
                return r
        return None

    @property
    def step(self) -> float:
        ks = sorted(r.strike for r in self.rows)
        diffs = sorted({round(b - a, 2) for a, b in zip(ks, ks[1:]) if b - a > 0})
        return diffs[0] if diffs else 50.0


def _leg(d: Mapping[str, Any] | None) -> Leg:
    if not isinstance(d, Mapping):
        return Leg()
    g = d.get("greeks") if isinstance(d.get("greeks"), Mapping) else {}
    ltp = _f(d.get("last_price"))
    return Leg(
        ltp=ltp if ltp else None,
        bid=_f(d.get("top_bid_price")) or None,
        ask=_f(d.get("top_ask_price")) or None,
        oi=_f(d.get("oi")) or 0.0,
        prev_oi=_f(d.get("previous_oi")),
        volume=_f(d.get("volume")) or 0.0,
        iv=_f(d.get("implied_volatility")),
        delta=_f(g.get("delta")),
        theta=_f(g.get("theta")),
    )


def parse_chain(raw: Mapping[str, Any], expiry: str, fetched_at: dt.datetime | None = None) -> Chain:
    oc = raw.get("oc") or {}
    rows = []
    for k, legs in oc.items():
        strike = _f(k)
        if not strike or strike <= 0 or not isinstance(legs, Mapping):
            continue
        rows.append(Row(strike, _leg(legs.get("ce")), _leg(legs.get("pe"))))
    rows.sort(key=lambda r: r.strike)
    return Chain(expiry=expiry, spot=_f(raw.get("last_price")) or 0.0, rows=rows, fetched_at=fetched_at)


@dataclasses.dataclass
class ChainStats:
    spot: float
    atm: float
    pcr: float | None
    max_pain: float | None
    call_wall: float | None
    call_wall_oi: float
    put_wall: float | None
    put_wall_oi: float
    call_wall_2: float | None
    put_wall_2: float | None
    oi_bias: float | None
    oi_bias_label: str


def nearest_strike(chain: Chain, price: float) -> float:
    return min((r.strike for r in chain.rows), key=lambda k: abs(k - price)) if chain.rows else round(price / 50) * 50


def max_pain(chain: Chain) -> float | None:
    if not chain.rows:
        return None
    best, best_loss = None, None
    for cand in chain.rows:
        s = cand.strike
        loss = 0.0
        for r in chain.rows:
            if s > r.strike:
                loss += (s - r.strike) * r.ce.oi
            elif s < r.strike:
                loss += (r.strike - s) * r.pe.oi
        if best_loss is None or loss < best_loss:
            best, best_loss = s, loss
    return best


def stats(chain: Chain, prev_chain_oi: Mapping[float, tuple[float, float]] | None = None, window: int = 5,
          wall_window: int = 8) -> ChainStats:
    spot = chain.spot
    atm = nearest_strike(chain, spot) if chain.rows else 0.0
    tot_ce = sum(r.ce.oi for r in chain.rows)
    tot_pe = sum(r.pe.oi for r in chain.rows)
    pcr = (tot_pe / tot_ce) if tot_ce > 0 else None

    step = chain.step
    reach = wall_window * step
    above_all = sorted([r for r in chain.rows if r.strike >= spot], key=lambda r: r.ce.oi, reverse=True)
    below_all = sorted([r for r in chain.rows if r.strike <= spot], key=lambda r: r.pe.oi, reverse=True)
    above = [r for r in above_all if r.strike - spot <= reach] or above_all
    below = [r for r in below_all if spot - r.strike <= reach] or below_all
    cw = above[0] if above and above[0].ce.oi > 0 else None
    pw = below[0] if below and below[0].pe.oi > 0 else None

    def _second(near, every, first, attr):
        far = every[0] if every and first is not None and every[0].strike != first.strike and getattr(every[0], attr).oi > 0 else None
        if far:
            return far.strike
        return near[1].strike if len(near) > 1 and getattr(near[1], attr).oi > 0 else None
    cw2 = _second(above, above_all, cw, "ce")
    pw2 = _second(below, below_all, pw, "pe")

    near = [r for r in chain.rows if abs(r.strike - atm) <= window * step + 0.01]
    ce_add = pe_add = 0.0
    have = False
    for r in near:
        ce_chg, pe_chg = r.ce.oi_change, r.pe.oi_change
        if (ce_chg is None or pe_chg is None) and prev_chain_oi and r.strike in prev_chain_oi:
            p_ce, p_pe = prev_chain_oi[r.strike]
            ce_chg, pe_chg = r.ce.oi - p_ce, r.pe.oi - p_pe
        if ce_chg is None or pe_chg is None:
            continue
        have = True
        ce_add += ce_chg
        pe_add += pe_chg
    bias = (pe_add - ce_add) if have else None
    if bias is None:
        label = "n/a"
    else:
        scale = max(abs(ce_add), abs(pe_add), 1.0)
        rel = bias / scale
        label = "bullish (puts being written)" if rel > 0.25 else "bearish (calls being written)" if rel < -0.25 else "neutral"
    return ChainStats(spot, atm, pcr, max_pain(chain), cw.strike if cw else None, cw.ce.oi if cw else 0.0,
                      pw.strike if pw else None, pw.pe.oi if pw else 0.0, cw2, pw2, bias, label)


def pick_strike(chain: Chain, spot: float, side: str) -> float:
    ks = [r.strike for r in chain.rows]
    if not ks:
        step = 50
        return (spot // step) * step if side == "CE" else -(-spot // step) * step
    if side == "CE":
        below = [k for k in ks if k <= spot]
        return max(below) if below else min(ks)
    above = [k for k in ks if k >= spot]
    return min(above) if above else max(ks)


def choose_expiry(expiries: list[str], today: dt.date, avoid_expiry_day: bool = True) -> str | None:
    ds = []
    for e in expiries:
        try:
            ds.append(dt.date.fromisoformat(e[:10]))
        except ValueError:
            continue
    ds = sorted(d for d in ds if d >= today)
    if not ds:
        return None
    if avoid_expiry_day and ds[0] == today and len(ds) > 1:
        return ds[1].isoformat()
    return ds[0].isoformat()


def nearest_expiry(expiries: list[str], today: dt.date) -> str | None:
    return choose_expiry(expiries, today, avoid_expiry_day=False)


def chain_table(chain: Chain, around: float, n: int = 6) -> list[dict]:
    step = chain.step
    rows = [r for r in chain.rows if abs(r.strike - around) <= n * step + 0.01]
    return [{
        "strike": r.strike,
        "ce_oi": r.ce.oi, "ce_chg": r.ce.oi_change, "ce_ltp": r.ce.ltp, "ce_iv": r.ce.iv,
        "pe_oi": r.pe.oi, "pe_chg": r.pe.oi_change, "pe_ltp": r.pe.ltp, "pe_iv": r.pe.iv,
    } for r in rows]

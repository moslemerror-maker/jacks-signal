from __future__ import annotations

import dataclasses
import datetime as dt
import math
import random

from .clock import IST
from .instruments import Instruments, Stock

NIFTY, BNF, VIX = 13, 25, 21


class FakeClock:
    def __init__(self, t: dt.datetime) -> None:
        self.t = t

    def now(self) -> dt.datetime:
        return self.t


def _bars_from_path(day: dt.date, path: list[float], start="09:15") -> list[tuple[dt.datetime, float]]:
    h, m = map(int, start.split(":"))
    t0 = dt.datetime.combine(day, dt.time(h, m), IST)
    return [(t0 + dt.timedelta(minutes=i), p) for i, p in enumerate(path)]


def _interp(points: list[tuple[str, float]], day: dt.date, noise: float, seed: int) -> list[float]:
    rnd = random.Random(seed)
    pts = [(dt.datetime.combine(day, dt.time(*map(int, t.split(":"))), IST), p) for t, p in points]
    out = []
    t = pts[0][0]
    end = pts[-1][0]
    while t <= end:
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= t <= tb:
                frac = (t - ta).total_seconds() / max((tb - ta).total_seconds(), 1)
                out.append(pa + (pb - pa) * frac + rnd.uniform(-noise, noise))
                break
        t += dt.timedelta(minutes=1)
    return out


OCT9_NIFTY = [("09:15", 22315), ("09:30", 22360), ("09:45", 22420), ("10:15", 22490), ("10:30", 22500),
              ("11:00", 22455), ("11:30", 22520), ("12:00", 22470), ("12:30", 22505), ("13:00", 22495),
              ("13:30", 22520), ("13:45", 22528), ("13:58", 22548), ("14:05", 22560), ("14:18", 22575),
              ("14:35", 22545), ("14:50", 22560), ("15:00", 22563), ("15:05", 22542), ("15:15", 22526), ("15:29", 22520)]


@dataclasses.dataclass
class SimStock:
    symbol: str
    sid: int
    prev_close: float
    waypoints: list[tuple[str, float]]
    vol_today: float
    vol_hist: float
    burst_at: str | None = None


class FakeDhan:
    def __init__(self, clock: FakeClock, day: dt.date, nifty_points=None, stocks: list[SimStock] | None = None,
                 fail_auth: bool = False) -> None:
        self.clock, self.day, self.calls, self.fail_auth = clock, day, 0, fail_auth
        self.client_id, self.access_token = "1000000001", "x.y.z"
        npts = nifty_points or OCT9_NIFTY
        self.paths: dict[int, list[tuple[dt.datetime, float, float]]] = {}
        n = _interp(npts, day, 4, 1)
        self.paths[NIFTY] = [(t, p, 0) for t, p in _bars_from_path(day, n)]
        bnf = [p * 2.42 + 40 * math.sin(i / 37) for i, p in enumerate(n)]
        self.paths[BNF] = [(t, p, 0) for t, p in _bars_from_path(day, bnf)]
        vix = [14.1 - 0.0015 * i for i in range(len(n))]
        self.paths[VIX] = [(t, p, 0) for t, p in _bars_from_path(day, vix)]
        self.prev = {NIFTY: 22231.8, BNF: 54515.0, VIX: 14.2}
        self.stocks = stocks if stocks is not None else default_stocks()
        for s in self.stocks:
            path = _interp(s.waypoints, day, s.prev_close * 0.0004, s.sid)
            bars = []
            for t, p in _bars_from_path(day, path):
                v = s.vol_today
                if s.burst_at:
                    b = dt.datetime.combine(day, dt.time(*map(int, s.burst_at.split(":"))), IST)
                    if b <= t < b + dt.timedelta(minutes=15):
                        v *= 4
                bars.append((t, p, v))
            self.paths[s.sid] = bars
            self.prev[s.sid] = s.prev_close

    def set_token(self, t):  # pragma: no cover - parity with DhanApi
        self.access_token = t

    def _check(self):
        self.calls += 1
        if self.fail_auth:
            from .dhan_api import DhanAuthError
            raise DhanAuthError("simulated expired token")

    def _history(self, sid: int, minutes: int) -> list[tuple[dt.datetime, float, float, float, float, float]]:
        rnd = random.Random(sid * 7)
        out = []
        base = self.prev[sid]
        d = self.day - dt.timedelta(days=1)
        days = []
        while len(days) < 6:
            if d.weekday() < 5:
                days.append(d)
            d -= dt.timedelta(days=1)
        stock = next((s for s in self.stocks if s.sid == sid), None)
        for day in sorted(days):
            t = dt.datetime.combine(day, dt.time(9, 15), IST)
            px = base * (1 + rnd.uniform(-0.01, 0.01))
            while t.time() < dt.time(15, 30):
                o = px
                c = o * (1 + rnd.uniform(-0.002, 0.002))
                hi, lo = max(o, c) * 1.0008, min(o, c) * 0.9992
                vol = (stock.vol_hist * minutes) if stock else 0
                out.append((t, o, hi, lo, c, vol))
                px = c
                t += dt.timedelta(minutes=minutes)
            if day == max(days):
                last = out[-1]
                out[-1] = (last[0], last[1], max(last[2], base), min(last[3], base), base, last[5])
        return out

    def _today(self, sid: int, minutes: int):
        now = self.clock.now()
        bars = [b for b in self.paths.get(sid, []) if b[0] <= now - dt.timedelta(seconds=1)]
        out: list[list] = []
        for t, p, v in bars:
            mins = (t.hour * 60 + t.minute - 555) // minutes * minutes + 555
            slot = t.replace(hour=mins // 60, minute=mins % 60, second=0, microsecond=0)
            if not out or out[-1][0] != slot:
                out.append([slot, p, p, p, p, v])
            else:
                b = out[-1]
                b[2], b[3], b[4], b[5] = max(b[2], p), min(b[3], p), p, b[5] + v
        return [tuple(b) for b in out]

    @staticmethod
    def _arrays(rows):
        return {"open": [r[1] for r in rows], "high": [r[2] for r in rows], "low": [r[3] for r in rows],
                "close": [r[4] for r in rows], "volume": [r[5] for r in rows], "timestamp": [r[0].timestamp() for r in rows]}

    def intraday(self, security_id, segment, instrument, from_date, to_date, interval=15):
        self._check()
        sid = int(security_id)
        return self._arrays(self._history(sid, interval) + self._today(sid, interval))

    def daily(self, security_id, segment, instrument, from_date, to_date):
        self._check()
        sid = int(security_id)
        rnd = random.Random(sid)
        rows, d, px = [], self.day - dt.timedelta(days=45), self.prev[sid] * 0.97
        stock = next((s for s in self.stocks if s.sid == sid), None)
        while d < self.day:
            if d.weekday() < 5:
                o = px
                c = o * (1 + rnd.uniform(-0.012, 0.014))
                rows.append((dt.datetime.combine(d, dt.time(0, 0), IST), o, max(o, c) * 1.006, min(o, c) * 0.994, c,
                             (stock.vol_hist if stock else 1) * 375))
                px = c
            d += dt.timedelta(days=1)
        last = rows[-1]
        pc = self.prev[sid]
        rows[-1] = (last[0], last[1], max(last[2], pc * 1.002), min(last[3], pc * 0.99), pc, last[5] * 1.5)
        return self._arrays(rows)

    def ohlc(self, securities):
        self._check()
        out: dict = {}
        for seg, ids in securities.items():
            out[seg] = {}
            for sid in ids:
                rows = self._today(int(sid), 1)
                if not rows:
                    continue
                out[seg][str(sid)] = {"last_price": rows[-1][4], "ohlc": {"open": rows[0][1], "high": max(r[2] for r in rows),
                                                                          "low": min(r[3] for r in rows), "close": self.prev[int(sid)]}}
        return out

    def expiry_list(self, security_id, segment):
        self._check()
        d = self.day
        outs = []
        while len(outs) < 4:
            if d.weekday() == 1:
                outs.append(d.isoformat())
            d += dt.timedelta(days=1)
        return outs

    def option_chain(self, security_id, segment, expiry):
        self._check()
        rows = self._today(NIFTY, 1)
        spot = rows[-1][4] if rows else self.prev[NIFTY]
        days = max((dt.date.fromisoformat(expiry) - self.day).days, 0) + 0.5
        oc = {}
        for k in range(21800, 23250, 50):
            dist = (k - spot) / (spot * 0.13 * math.sqrt(days / 365))
            nd = 0.5 * (1 + math.erf(-dist / math.sqrt(2)))
            tv = spot * 0.13 * math.sqrt(days / 365) * 0.4 * math.exp(-dist * dist / 2)
            ce = max(spot - k, 0) + tv
            pe = max(k - spot, 0) + tv
            ce_oi = 4e6 * math.exp(-((k - 22600) / 120) ** 2) + 2.5e6 * math.exp(-((k - 22700) / 120) ** 2) + 3e5
            pe_oi = 4.5e6 * math.exp(-((k - 22400) / 120) ** 2) + 2e6 * math.exp(-((k - 22300) / 120) ** 2) + 3e5
            oc[f"{k:.6f}"] = {
                "ce": {"last_price": round(ce, 2), "oi": int(ce_oi), "previous_oi": int(ce_oi * 0.8), "volume": 1000,
                       "implied_volatility": 13.0, "greeks": {"delta": round(nd, 3), "theta": -5.0}},
                "pe": {"last_price": round(pe, 2), "oi": int(pe_oi), "previous_oi": int(pe_oi * 0.7), "volume": 1000,
                       "implied_volatility": 13.5, "greeks": {"delta": round(nd - 1, 3), "theta": -5.0}},
            }
        return {"last_price": spot, "oc": oc}


def default_stocks() -> list[SimStock]:
    return [
        SimStock("ALPHA", 101, 1000.0, [("09:15", 1004), ("10:14", 1010), ("10:29", 1024), ("11:30", 1030), ("15:29", 1036)], 2000, 1500, "10:15"),
        SimStock("BETA", 102, 500.0, [("09:15", 501), ("11:00", 503), ("11:14", 504), ("11:29", 507.5), ("15:29", 505)], 900, 1500),
        SimStock("GAMMA", 103, 2000.0, [("09:15", 1995), ("10:44", 1985), ("10:59", 1962), ("12:00", 1950), ("15:29", 1945)], 1500, 1200, "10:45"),
        SimStock("DELTA", 104, 300.0, [("09:15", 300.2), ("15:29", 300.4)], 1000, 1000),
    ]


def sim_instruments(today: dt.date) -> Instruments:
    inst = Instruments(nifty_id=NIFTY, banknifty_id=BNF, vix_id=VIX, nifty_lot=65, source="simulator")
    inst.stocks = [Stock(s.symbol, s.sid, 500) for s in default_stocks()]
    return inst


def run_day(day: dt.date = dt.date(2026, 10, 9), start="08:50", end="15:40", step_s=30, state_dir=None, quiet=False,
            settings=None, setup=None, **fake_kw):
    import tempfile
    from pathlib import Path

    from .config import Settings
    from .engine import Engine
    from .notify import Notifier

    clock = FakeClock(dt.datetime.combine(day, dt.time(*map(int, start.split(":"))), IST))
    api = FakeDhan(clock, day, **fake_kw)
    s = settings or Settings(dhan_client_id="1", dhan_access_token="x", stock_alerts_enabled=True)
    eng = Engine(s, api=api, clock=clock, notifier=Notifier(), instruments_loader=sim_instruments,
                 state_dir=Path(state_dir or tempfile.mkdtemp(prefix="sdsim-")))
    if setup:
        setup(eng)
    stop = dt.datetime.combine(day, dt.time(*map(int, end.split(":"))), IST)
    import time as _t
    while clock.t <= stop:
        eng.tick()
        if eng.profiles_status == "building" or eng.profiles_status.startswith("building "):
            for _ in range(200):
                if eng.profiles_status.startswith("ready") or eng.profiles_status.startswith("stopped"):
                    break
                _t.sleep(0.02)
        clock.t += dt.timedelta(seconds=step_s)
    if not quiet:
        for item in reversed(list(eng.notifier.feed)):
            print(f"[{item['kind']}] {item['text']}\n")
        print(f"API calls: {api.calls}")
    return eng

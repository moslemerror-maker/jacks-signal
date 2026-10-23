from __future__ import annotations

import argparse
import csv
import datetime as dt
import email.utils
import gzip
import hashlib
import io
import json
import math
import pathlib
import sys
import time
import zipfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import NamedTuple, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "data" / "raw" / "nse_fo_sample"
OUT_ROOT = ROOT / "data" / "derived" / "o1_0b_sample_gate"
FETCHER_VERSION = "o1_0b_sample_gate/1"
BASE = "https://nsearchives.nseindia.com/content"
D = dt.date
UTC = dt.timezone.utc
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SESSION_OPEN_IST = dt.time(9, 15)
SESSION_CLOSE_IST = dt.time(15, 30)
NSE_EPOCH = dt.datetime(1980, 1, 1, tzinfo=UTC)
_MON = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"}
LEGACY_TYPE = {"FUTIDX": "IDF", "FUTSTK": "STF", "OPTIDX": "IDO", "OPTSTK": "STO"}
UDIFF_TYPES = ("IDF", "STF", "IDO", "STO")

SAMPLE = [
    ("legacy", D(2003, 6, 2), "schema stability: earliest verified legacy era"),
    ("legacy", D(2010, 6, 1), "schema stability: 2010"),
    ("legacy", D(2015, 1, 5), "2015 NIFTY options day; non-expiry; consecutive pair (OI change)"),
    ("legacy", D(2015, 1, 6), "consecutive pair (OI change)"),
    ("legacy", D(2015, 1, 29), "2015 monthly expiry day"),
    ("legacy", D(2019, 6, 3), "legacy F&O day, weekly-era NIFTY"),
    ("legacy", D(2019, 6, 27), "2019 monthly expiry day"),
    ("legacy", D(2023, 6, 1), "2023 day (legacy era, before UDiFF exists)"),
    ("legacy", D(2023, 6, 28), "2023 monthly expiry day (expiry advanced from the 06-29 holiday)"),
    ("legacy", D(2024, 1, 2), "overlap: first UDiFF day, legacy rendering"),
    ("legacy", D(2024, 3, 28), "overlap: monthly expiry day, legacy rendering"),
    ("legacy", D(2024, 7, 4), "overlap + consecutive pair"),
    ("legacy", D(2024, 7, 5), "overlap: last legacy day"),
    ("udiff", D(2024, 1, 2), "overlap: first UDiFF day"),
    ("udiff", D(2024, 3, 28), "overlap: monthly expiry day"),
    ("udiff", D(2024, 7, 4), "overlap + consecutive pair"),
    ("udiff", D(2024, 7, 5), "overlap: last legacy day"),
    ("udiff", D(2024, 7, 8), "first post-cutover day (UDiFF only); consecutive pair with 2024-07-05"),
    ("udiff", D(2024, 7, 9), "post-cutover"),
    ("udiff", D(2024, 11, 21), "lot-size transition window (NIFTY 25 -> 75 by series)"),
    ("udiff", D(2025, 9, 2), "first Tuesday weekly expiry (expiry-day regime change)"),
    ("udiff", D(2025, 9, 30), "Tuesday monthly expiry"),
    ("udiff", D(2026, 1, 6), "lot-size transition (NIFTY 75 -> 65) weekly expiry"),
    ("udiff", D(2026, 9, 15), "recent Tuesday weekly expiry"),
    ("udiff", D(2026, 9, 17), "recent day; consecutive pair"),
    ("udiff", D(2026, 9, 18), "recent non-expiry day"),
    ("udiff", D(2024, 2, 5), "same date as the earliest contract master"),
    ("udiff", D(2025, 1, 15), "same date as a mid contract master"),
    ("master", D(2024, 2, 5), "earliest contract master"),
    ("master", D(2025, 1, 15), "mid-history contract master"),
    ("master", D(2026, 9, 17), "recent contract master (id stability)"),
    ("master", D(2026, 9, 18), "recent contract master"),
]
OVERLAP_DATES = [D(2024, 1, 2), D(2024, 3, 28), D(2024, 7, 4), D(2024, 7, 5)]
CONSECUTIVE_PAIRS = [("legacy", D(2015, 1, 5), D(2015, 1, 6)), ("legacy", D(2024, 7, 4), D(2024, 7, 5)),
                     ("udiff", D(2024, 7, 5), D(2024, 7, 8)), ("udiff", D(2026, 9, 17), D(2026, 9, 18))]
EXPIRY_SETTLEMENT_DATES = [("legacy", D(2015, 1, 29)), ("legacy", D(2019, 6, 27)), ("legacy", D(2023, 6, 28)), ("udiff", D(2024, 3, 28)),
                           ("udiff", D(2025, 9, 2)), ("udiff", D(2025, 9, 30)), ("udiff", D(2026, 1, 6)), ("udiff", D(2026, 9, 15))]
MASTER_UDIFF_PAIRS = [(D(2024, 2, 5)), (D(2025, 1, 15)), (D(2026, 9, 18))]

GATE_CRITERIA = {
    "G1_SCHEMA_STABLE": "within each family (legacy, UDiFF, contract master) every sampled file has ONE identical header",
    "G2_CONTRACT_KEY_UNIQUE": "no duplicate (underlying, type, expiry, strike, CE/PE) within any sampled file",
    "G3_STRIKE_EXPIRY_EXACT": "every strike is exactly representable in paise (x100 integral) and every expiry parses unambiguously",
    "G4_LEGACY_UDIFF_RECONCILES": "on every overlap date both renderings have the same contract set and ZERO disagreement on price, settle, volume, OI, OI change",
    "G5_ZERO_VOLUME_TAXONOMY_CLOSED": "every row falls into exactly one defined trade state; anomalous states are counted and reported (PASS_WITH_FINDINGS if any)",
    "G6_OI_CHANGE_RECONCILES": "on consecutive-session pairs OI(t) - OI(t-1) equals CHG_IN_OI for >= 99% of contracts (PASS_WITH_FINDINGS below 100%)",
    "G7_EXPIRY_SETTLEMENT_CHARACTERIZED": "the settle price of expiring NIFTY options is compared with intrinsic value on the index close (characterisation, not pass/fail)",
    "G8_IDENTITY_STABLE": "a source instrument id maps to exactly one contract across all sampled UDiFF and master files (no reuse within the sample)",
    "G9_MASTER_CONSISTENT": "the contract master agrees with UDiFF on expiry (decoded), strike (/100), option type, underlying and lot size for every shared id",
    "G10_PROVENANCE_CAPTURED": "every sampled file has a sha256 sidecar and the response headers needed to reason about availability",
    "G11_TIMESTAMPS_CONSISTENT": "the trade date inside every row equals the file's date",
}

DISPOSITIONS = {
    "G1_SCHEMA_STABLE": "Genuine, benign: UDiFF reserved columns are named Rsvd01..04 in early files and Rsvd1..4 from 2024-07 on (the 30 named columns are identical); the legacy header carries "
                        "a constant trailing empty column. Design: format_version = hash of the header; the normalizer keys on NAMED columns and treats Rsvd* as opaque; an unknown NAMED column set is a hard stop.",
    "G8_IDENTITY_STABLE": "Genuine: FinInstrmId tokens are RECYCLED over long horizons (identical across adjacent days: 80,805/80,805). 0.24% of canonical contracts (747 of 310,996 in the UDiFF sample; nine stock underlyings "
                          "(INFERRED: corporate-action adjusted), none in NIFTY) also carry different tokens on different dates. Design: the token is never an identity; it is an effective-dated alias.",
    "G9_MASTER_CONSISTENT": "Genuine: (a) the master published on D already shows dividend-ADJUSTED strikes (GAIL 2024-02-05, CESC 2025-01-15) while the bhavcopy of D shows the strikes in force on D; "
                            "(b) contracts deleted on D appear in the master as deleted-token rows with blanked identity. Design: the master describes the contract set at ~17:50 on D "
                            "(post next-day adjustment), the bhavcopy describes the trading day; the two are separate evidence classes and are never merged silently.",
}


class ContractKey(NamedTuple):
    underlying: str
    dtype: str
    expiry: dt.date
    strike_paise: int
    cp: str | None


class Bar(NamedTuple):
    key: ContractKey
    trade_date: dt.date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    settle: Decimal
    volume: int
    oi: int
    oi_change: int
    turnover: Decimal | None
    source_id: str | None
    lot: int | None
    underlying_price: Decimal | None
    actual_expiry: dt.date | None
    row_hash: str


def dec(s) -> Decimal | None:
    s = ("" if s is None else str(s)).strip()
    if s == "":
        return None
    try:
        return Decimal(s)
    except InvalidOperation:
        return None


def to_int(s) -> int | None:
    d = dec(s)
    if d is None or d != d.to_integral_value():
        return None
    return int(d)


def strike_to_paise(text: str) -> tuple[int | None, bool]:
    d = dec(text)
    if d is None:
        return None, False
    p = d * 100
    return int(p.to_integral_value()), p == p.to_integral_value()


OPTION_TYPES = ("CE", "PE", "CA", "PA")


def norm_cp(raw: str | None) -> str | None:
    r = (raw or "").strip()
    return r if r in OPTION_TYPES else None


def parse_legacy_date(s: str) -> dt.date:
    return dt.datetime.strptime(s.strip(), "%d-%b-%Y").date()


def row_hash(fields: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def classify_trade_state(volume: int, o: Decimal, h: Decimal, l: Decimal, c: Decimal) -> tuple[bool, str]:
    if volume > 0:
        if min(o, h, l, c) <= 0:
            return False, "ANOMALY_TRADED_NONPOSITIVE_PRICE"
        if not (l <= o <= h and l <= c <= h):
            return False, "ANOMALY_TRADED_OHLC_ORDER"
        return True, "TRADED"
    if volume == 0:
        if o == 0 and h == 0 and l == 0 and c == 0:
            return False, "NOT_TRADED_NO_PRICE"
        if o == 0 and h == 0 and l == 0 and c > 0:
            return False, "NOT_TRADED_PRICE_CARRIED"
        return False, "ANOMALY_NOT_TRADED_WITH_OHLC"
    return False, "ANOMALY_NEGATIVE_VOLUME"


def decode_nse_seconds(x) -> dt.datetime:
    return NSE_EPOCH + dt.timedelta(seconds=int(x))


def gcd_step(values: Sequence[int]) -> int | None:
    vs = sorted(set(values))
    if len(vs) < 2:
        return None
    g = 0
    for a, b in zip(vs, vs[1:]):
        g = math.gcd(g, b - a)
    return g


def session_close_utc(session_date: dt.date) -> dt.datetime:
    return dt.datetime.combine(session_date, SESSION_CLOSE_IST, tzinfo=IST).astimezone(UTC)


def next_session_open_utc(session_date: dt.date, sessions: Sequence[dt.date] | None = None) -> tuple[dt.datetime, str]:
    later = [s for s in (sessions or ()) if s > session_date]
    if later:
        nxt, basis = min(later), "OBSERVED_SESSIONS"
    else:
        nxt, basis = session_date + dt.timedelta(1), "WEEKDAY_FALLBACK"
        while nxt.weekday() >= 5:
            nxt += dt.timedelta(1)
    return dt.datetime.combine(nxt, SESSION_OPEN_IST, tzinfo=IST).astimezone(UTC), basis


def availability_for_eod_file(session_date: dt.date, sessions: Sequence[dt.date] | None, last_modified: dt.datetime | None) -> dict:
    close = session_close_utc(session_date)
    nxt, cal = next_session_open_utc(session_date, sessions)
    if last_modified is not None and close <= last_modified < nxt:
        return {"available_at": last_modified, "availability_basis": "OBSERVED_HTTP_LAST_MODIFIED", "evidence": {"last_modified": last_modified.isoformat(), "calendar": cal}}
    if last_modified is None:
        why = "NO_LAST_MODIFIED"
    elif last_modified < close:
        why = "LAST_MODIFIED_BEFORE_SESSION_CLOSE_IMPLAUSIBLE"
    else:
        why = "LAST_MODIFIED_AFTER_NEXT_OPEN_REUPLOAD_OR_LATE"
    return {"available_at": nxt, "availability_basis": "RULE_NEXT_SESSION_OPEN",
            "evidence": {"last_modified": last_modified.isoformat() if last_modified else None, "reason": why, "calendar": cal}}


def parse_last_modified(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        t = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return t.astimezone(UTC) if t.tzinfo else t.replace(tzinfo=UTC)


def _csv_rows(text: str) -> tuple[list[str], list[list[str]]]:
    rows = list(csv.reader(io.StringIO(text)))
    return rows[0], rows[1:]


def read_zip_text(path: pathlib.Path) -> str:
    z = zipfile.ZipFile(path)
    return z.read(z.namelist()[0]).decode("utf-8", "replace")


def read_gz_text(path: pathlib.Path) -> str:
    return gzip.decompress(path.read_bytes()).decode("utf-8", "replace")


def legacy_bars(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[Bar]:
    idx = {h.strip(): i for i, h in enumerate(header) if h.strip()}
    out = []
    for r in rows:
        if not r or len(r) < len(idx):
            continue
        g = lambda name: r[idx[name]].strip()  # noqa: E731
        raw_type = g("INSTRUMENT")
        cp = g("OPTION_TYP")
        sp, _ = strike_to_paise(g("STRIKE_PR"))
        key = ContractKey(g("SYMBOL"), LEGACY_TYPE.get(raw_type, f"OTHER:{raw_type}"), parse_legacy_date(g("EXPIRY_DT")),
                          sp if sp is not None else -1, norm_cp(cp))
        out.append(Bar(key, parse_legacy_date(g("TIMESTAMP")), dec(g("OPEN")) or Decimal(0), dec(g("HIGH")) or Decimal(0), dec(g("LOW")) or Decimal(0),
                       dec(g("CLOSE")) or Decimal(0), dec(g("SETTLE_PR")) or Decimal(0), to_int(g("CONTRACTS")) or 0, to_int(g("OPEN_INT")) or 0,
                       to_int(g("CHG_IN_OI")) or 0, dec(g("VAL_INLAKH")), None, None, None, None, row_hash(r)))
    return out


def udiff_bars(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[Bar]:
    idx = {h.strip(): i for i, h in enumerate(header) if h.strip()}
    out = []
    for r in rows:
        if not r or len(r) < len(idx):
            continue
        g = lambda name: r[idx[name]].strip()  # noqa: E731
        cp = g("OptnTp")
        sp, _ = strike_to_paise(g("StrkPric") or "0")
        act = g("FininstrmActlXpryDt")
        key = ContractKey(g("TckrSymb"), g("FinInstrmTp"), dt.date.fromisoformat(g("XpryDt")), sp if sp is not None else -1, norm_cp(cp))
        out.append(Bar(key, dt.date.fromisoformat(g("TradDt")), dec(g("OpnPric")) or Decimal(0), dec(g("HghPric")) or Decimal(0), dec(g("LwPric")) or Decimal(0),
                       dec(g("ClsPric")) or Decimal(0), dec(g("SttlmPric")) or Decimal(0), to_int(g("TtlTradgVol")) or 0, to_int(g("OpnIntrst")) or 0,
                       to_int(g("ChngInOpnIntrst")) or 0, dec(g("TtlTrfVal")), g("FinInstrmId") or None, to_int(g("NewBrdLotQty")),
                       dec(g("UndrlygPric")), dt.date.fromisoformat(act) if act else None, row_hash(r)))
    return out


def by_key(bars: Sequence[Bar]) -> tuple[dict[ContractKey, Bar], list[ContractKey]]:
    seen: dict[ContractKey, Bar] = {}
    dups = []
    for b in bars:
        if b.key in seen:
            dups.append(b.key)
        seen[b.key] = b
    return seen, dups


def trade_state_counts(bars: Sequence[Bar]) -> dict:
    states = Counter()
    oi_only = 0
    for b in bars:
        _, s = classify_trade_state(b.volume, b.open, b.high, b.low, b.close)
        states[s] += 1
        if b.volume == 0 and b.oi > 0:
            oi_only += 1
    n = len(bars)
    return {"rows": n, "states": dict(states), "traded": states["TRADED"], "not_traded": n - states["TRADED"],
            "zero_volume_pct": round(100 * sum(v for k, v in states.items() if k != "TRADED" and not k.startswith("ANOMALY_TRADED")) / n, 1) if n else None,
            "zero_volume_with_oi": oi_only, "anomalies": {k: v for k, v in states.items() if k.startswith("ANOMALY")}}


def reconcile_renderings(legacy: dict[ContractKey, Bar], udiff: dict[ContractKey, Bar]) -> dict:
    both = set(legacy) & set(udiff)
    diffs = Counter()
    examples = defaultdict(list)
    turnover_max = Decimal(0)
    for k in both:
        a, b = legacy[k], udiff[k]
        for f in ("open", "high", "low", "close", "settle", "volume", "oi", "oi_change"):
            if getattr(a, f) != getattr(b, f):
                diffs[f] += 1
                if len(examples[f]) < 3:
                    examples[f].append((str(k), str(getattr(a, f)), str(getattr(b, f))))
        if a.turnover is not None and b.turnover is not None:
            turnover_max = max(turnover_max, abs(a.turnover - b.turnover / 100000))
    return {"legacy_rows": len(legacy), "udiff_rows": len(udiff), "in_both": len(both), "legacy_only": len(set(legacy) - set(udiff)),
            "udiff_only": len(set(udiff) - set(legacy)), "field_disagreements": dict(diffs), "examples": dict(examples),
            "max_turnover_diff_lakh": str(turnover_max), "reconciles": not diffs and len(both) == len(legacy) == len(udiff)}


def oi_change_reconciliation(prev: dict[ContractKey, Bar], today: dict[ContractKey, Bar]) -> dict:
    exact = new_ok = new_bad = present_bad = 0
    examples = []
    for k, b in today.items():
        p = prev.get(k)
        expected = b.oi - (p.oi if p else 0)
        if expected == b.oi_change:
            if p is None:
                new_ok += 1
            else:
                exact += 1
        else:
            if p is None:
                new_bad += 1
            else:
                present_bad += 1
            if len(examples) < 5:
                examples.append((str(k), b.oi, p.oi if p else None, b.oi_change, expected))
    dropped = [(k, p.oi) for k, p in prev.items() if k not in today and p.oi > 0]
    n = len(today)
    ok = exact + new_ok
    return {"contracts_today": n, "exact_both_days": exact, "exact_new_listing": new_ok, "mismatch_new_listing": new_bad, "mismatch_present_both_days": present_bad,
            "match_pct": round(100 * ok / n, 3) if n else None, "dropped_with_oi": len(dropped), "dropped_oi_total": sum(o for _, o in dropped), "examples": examples}


def expiry_day_settlement(today: dict[ContractKey, Bar], trade_date: dt.date, underlying: str, spot: Decimal) -> dict:
    rows = [(k, b) for k, b in today.items() if k.underlying == underlying and k.dtype == "IDO" and k.expiry == trade_date]
    classes = Counter(classify_expiry_settle(b.settle, b.close, spot, Decimal(k.strike_paise) / 100, k.cp) for k, b in rows)
    return {"expiring_rows": len(rows), "traded": sum(1 for _, b in rows if b.volume > 0), "spot": str(spot), "settle_classes": dict(classes)}


def classify_expiry_settle(settle: Decimal, close: Decimal, spot: Decimal, strike: Decimal, cp: str | None) -> str:
    intrinsic = max(spot - strike, Decimal(0)) if cp in ("CE", "CA") else max(strike - spot, Decimal(0))
    if settle == 0:
        return "ZERO"
    if settle == spot:
        return "UNDERLYING_SETTLEMENT_PRICE"
    if settle == intrinsic:
        return "INTRINSIC_VALUE"
    if settle == close:
        return "EQUALS_LAST_CLOSE"
    return "OTHER"


def id_conflicts(observations: Sequence[tuple[str, ContractKey, str]]) -> dict:
    by_id: dict[str, set] = defaultdict(set)
    by_key: dict[ContractKey, set] = defaultdict(set)
    for _, k, i in observations:
        by_id[i].add(k)
        by_key[k].add(i)
    reused = {i: sorted(map(str, ks))[:3] for i, ks in by_id.items() if len(ks) > 1}
    multi = {str(k): sorted(ids)[:3] for k, ids in by_key.items() if len(ids) > 1}
    return {"ids": len(by_id), "contracts": len(by_key), "ids_mapping_to_several_contracts": len(reused), "contracts_with_several_ids": len(multi),
            "examples_reused": dict(list(reused.items())[:3]), "examples_multi": dict(list(multi.items())[:3])}


def sample_path(family: str, d: dt.date) -> pathlib.Path:
    if family == "legacy":
        return SAMPLE_ROOT / "legacy" / f"fo{d.day:02d}{_MON[d.month]}{d.year}bhav.csv.zip"
    if family == "udiff":
        return SAMPLE_ROOT / "udiff" / f"BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return SAMPLE_ROOT / "master" / f"NSE_FO_contract_{d:%d%m%Y}.csv.gz"


def sample_url(family: str, d: dt.date) -> str:
    if family == "legacy":
        return f"{BASE}/historical/DERIVATIVES/{d.year}/{_MON[d.month]}/fo{d.day:02d}{_MON[d.month]}{d.year}bhav.csv.zip"
    if family == "udiff":
        return f"{BASE}/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    return f"{BASE}/fo/NSE_FO_contract_{d:%d%m%Y}.csv.gz"


def _sidecar(path: pathlib.Path) -> pathlib.Path:
    return path.with_suffix(path.suffix + ".meta.json")


def fetch_one(family: str, d: dt.date, session, *, now=None) -> dict:
    path, url = sample_path(family, d), sample_url(family, d)
    meta = _sidecar(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not meta.exists():
            raise RuntimeError(f"{path.name} exists without a provenance sidecar; refusing to trust or overwrite it")
        side = json.loads(meta.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != side["sha256"]:
            raise RuntimeError(f"{path.name} does not match its recorded sha256; raw files are immutable - investigate, do not overwrite")
        return side
    r = session.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                                  "Referer": "https://www.nseindia.com/all-reports-derivatives"}, timeout=40)
    if r.status_code == 404:
        return {"family": family, "date": d.isoformat(), "source_url": url, "http_status": 404, "available": False}
    r.raise_for_status()
    ct = r.headers.get("Content-Type", "").lower()
    if "html" in ct or len(r.content) < 500:
        raise RuntimeError(f"{url}: unusable body (content-type {ct!r}, {len(r.content)} bytes)")
    text = read_gz_text_bytes(r.content) if family == "master" else read_zip_text_bytes(r.content)
    _csv_rows(text)
    path.write_bytes(r.content)
    side = {"fetcher": FETCHER_VERSION, "family": family, "trade_date": d.isoformat(), "source_url": url,
            "fetched_at": (now or dt.datetime.now(UTC)).isoformat(), "sha256": hashlib.sha256(r.content).hexdigest(), "content_length": len(r.content),
            "http_status": r.status_code, "available": True,
            "response_headers": {k: r.headers.get(k) for k in ("Last-Modified", "ETag", "Date", "Content-Type", "Content-Length", "Server", "Age", "Cache-Control")}}
    meta.write_text(json.dumps(side, indent=2))
    return side


def read_zip_text_bytes(b: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(b))
    return z.read(z.namelist()[0]).decode("utf-8", "replace")


def read_gz_text_bytes(b: bytes) -> str:
    return gzip.decompress(b).decode("utf-8", "replace")


def load(family: str, d: dt.date):
    p = sample_path(family, d)
    if not p.exists():
        return None
    text = read_gz_text(p) if family == "master" else read_zip_text(p)
    header, rows = _csv_rows(text)
    return header, rows


def index_closes(dates: Sequence[dt.date], symbol: str = "NIFTY 50") -> dict[dt.date, Decimal]:
    try:
        sys.path.insert(0, str(ROOT))
        from sqlalchemy import text
        from backend.app.db.session import get_session
        s = get_session()
        try:
            s.execute(text("SET TRANSACTION READ ONLY"))
            rows = s.execute(text("SELECT b.trade_date, b.close FROM market_bars b JOIN instrument_symbols y ON y.instrument_id=b.instrument_id "
                                  "AND y.symbol=:s AND y.effective_to IS NULL WHERE b.series='INDEX' AND b.trade_date = ANY(:d)"), {"s": symbol, "d": list(dates)}).all()
            return {r[0]: Decimal(repr(r[1])) for r in rows}
        finally:
            s.rollback()
            s.close()
    except Exception:
        return {}


def observed_sessions() -> list[dt.date]:
    try:
        sys.path.insert(0, str(ROOT))
        from sqlalchemy import text
        from backend.app.db.session import get_session
        s = get_session()
        try:
            s.execute(text("SET TRANSACTION READ ONLY"))
            return [r[0] for r in s.execute(text("SELECT DISTINCT trade_date FROM market_bars WHERE series='EQ' AND timeframe='1D' ORDER BY 1"))]
        finally:
            s.rollback()
            s.close()
    except Exception:
        return []


def run_gate(*, use_db: bool = True) -> dict:
    report: dict = {"gate_criteria": GATE_CRITERIA, "generated_at": dt.datetime.now(UTC).isoformat(), "files": {}, "results": {}, "findings": []}
    findings = report["findings"]
    sessions = observed_sessions() if use_db else []
    loaded: dict[tuple[str, dt.date], tuple] = {}
    bars: dict[tuple[str, dt.date], list[Bar]] = {}
    headers: dict[str, set] = defaultdict(set)
    meta: dict[tuple[str, dt.date], dict] = {}
    for family, d, why in SAMPLE:
        got = load(family, d)
        if got is None:
            meta[(family, d)] = {"available": False}
            continue
        header, rows = got
        loaded[(family, d)] = got
        headers[family].add(tuple(h.strip() for h in header))
        side = json.loads(_sidecar(sample_path(family, d)).read_text())
        meta[(family, d)] = side
        if family == "legacy":
            bars[(family, d)] = legacy_bars(header, rows)
        elif family == "udiff":
            bars[(family, d)] = udiff_bars(header, rows)

    prov_ok = True
    files = {}
    for (family, d), side in meta.items():
        if not side.get("available"):
            files[f"{family}:{d}"] = {"available": False}
            continue
        lm = parse_last_modified((side.get("response_headers") or {}).get("Last-Modified"))
        av = availability_for_eod_file(d, sessions, lm)
        files[f"{family}:{d}"] = {"sha256": side["sha256"][:16], "bytes": side["content_length"], "last_modified": lm.isoformat() if lm else None,
                                  "last_modified_ist": lm.astimezone(IST).strftime("%Y-%m-%d %H:%M") if lm else None,
                                  "availability_basis": av["availability_basis"], "reason": av["evidence"].get("reason"), "calendar": av["evidence"].get("calendar")}
        if not side.get("sha256") or not side.get("response_headers"):
            prov_ok = False
    report["files"] = files
    report["results"]["G10_PROVENANCE_CAPTURED"] = {"status": "PASS" if prov_ok and files else "FAIL", "files": len(files),
                                                    "availability_basis": dict(Counter(f.get("availability_basis") for f in files.values() if f.get("available", True)))}
    missing = [k for k, v in files.items() if v.get("available") is False]
    if missing:
        findings.append({"id": "SAMPLE_MISSING", "detail": missing})

    g1 = {f: [list(h) for h in hs] for f, hs in headers.items()}
    report["results"]["G1_SCHEMA_STABLE"] = {"status": "PASS" if all(len(h) == 1 for h in headers.values()) else "FAIL",
                                             "distinct_headers": {f: len(h) for f, h in headers.items()}, "headers": {f: g1[f][0] if g1[f] else None for f in g1}}
    dup_total, inexact_total, date_mismatch = 0, 0, 0
    per_file = {}
    for (family, d), bs in bars.items():
        _, dups = by_key(bs)
        inexact = sum(1 for b in bs if b.key.strike_paise < 0)
        raw = loaded[(family, d)]
        idx = {h.strip(): i for i, h in enumerate(raw[0]) if h.strip()}
        col = "STRIKE_PR" if family == "legacy" else "StrkPric"
        inexact += sum(1 for r in raw[1] if r and len(r) >= len(idx) and not strike_to_paise(r[idx[col]] or "0")[1])
        dm = sum(1 for b in bs if b.trade_date != d)
        types = Counter(b.key.dtype for b in bs)
        dup_total += len(dups)
        inexact_total += inexact
        date_mismatch += dm
        per_file[f"{family}:{d}"] = {"rows": len(bs), "types": dict(types), "duplicates": len(dups), "inexact_strikes": inexact, "date_mismatch": dm}
    report["per_file"] = per_file
    report["results"]["G2_CONTRACT_KEY_UNIQUE"] = {"status": "PASS" if dup_total == 0 else "FAIL", "duplicates": dup_total}
    report["results"]["G3_STRIKE_EXPIRY_EXACT"] = {"status": "PASS" if inexact_total == 0 else "FAIL", "inexact_strikes": inexact_total}
    report["results"]["G11_TIMESTAMPS_CONSISTENT"] = {"status": "PASS" if date_mismatch == 0 else "FAIL", "mismatches": date_mismatch}

    rec = {}
    for d in OVERLAP_DATES:
        if ("legacy", d) in bars and ("udiff", d) in bars:
            rec[str(d)] = reconcile_renderings(by_key(bars[("legacy", d)])[0], by_key(bars[("udiff", d)])[0])
    report["reconciliation"] = rec
    report["results"]["G4_LEGACY_UDIFF_RECONCILES"] = {"status": "PASS" if rec and all(r["reconciles"] for r in rec.values()) else "FAIL", "dates": len(rec),
                                                       "contracts_compared": sum(r["in_both"] for r in rec.values())}

    tv = {f"{fam}:{d}": trade_state_counts(bs) for (fam, d), bs in bars.items()}
    report["trade_states"] = tv
    anomalies = sum(sum(v["anomalies"].values()) for v in tv.values())
    anomaly_by_type: Counter = Counter()
    zero_vol_by_type: Counter = Counter()
    rows_by_type: Counter = Counter()
    for (fam, d), bs in bars.items():
        for b in bs:
            rows_by_type[b.key.dtype] += 1
            _, st = classify_trade_state(b.volume, b.open, b.high, b.low, b.close)
            if st.startswith("ANOMALY"):
                anomaly_by_type[(b.key.dtype, st)] += 1
            elif st != "TRADED":
                zero_vol_by_type[(b.key.dtype, st)] += 1
    report["anomalies_by_type"] = {f"{t}:{s}": n for (t, s), n in sorted(anomaly_by_type.items())}
    report["non_traded_by_type"] = {f"{t}:{s}": n for (t, s), n in sorted(zero_vol_by_type.items())}
    report["rows_by_type"] = dict(rows_by_type)
    report["results"]["G5_ZERO_VOLUME_TAXONOMY_CLOSED"] = {"status": "PASS" if anomalies == 0 else "PASS_WITH_FINDINGS", "anomalous_rows": anomalies,
                                                           "anomaly_breakdown": dict(sum((Counter(v["anomalies"]) for v in tv.values()), Counter()))}

    oic = {}
    for fam, d0, d1 in CONSECUTIVE_PAIRS:
        if (fam, d0) in bars and (fam, d1) in bars:
            oic[f"{fam}:{d0}->{d1}"] = oi_change_reconciliation(by_key(bars[(fam, d0)])[0], by_key(bars[(fam, d1)])[0])
    report["oi_change"] = oic
    worst = min((v["match_pct"] for v in oic.values()), default=None)
    report["results"]["G6_OI_CHANGE_RECONCILES"] = {"status": "PASS" if worst == 100 else ("PASS_WITH_FINDINGS" if worst is not None and worst >= 99 else "FAIL"),
                                                    "worst_match_pct": worst, "pairs": len(oic)}

    dates = [d for _, d in EXPIRY_SETTLEMENT_DATES]
    closes = index_closes(dates) if use_db else {}
    es = {}
    for fam, d in EXPIRY_SETTLEMENT_DATES:
        if (fam, d) not in bars:
            continue
        today = by_key(bars[(fam, d)])[0]
        spot = closes.get(d)
        udiff_spot = next((b.underlying_price for k, b in today.items() if k.underlying == "NIFTY" and b.underlying_price), None)
        if spot is not None:
            es[f"{fam}:{d}"] = {**expiry_day_settlement(today, d, "NIFTY", spot), "spot_source": "dev_db_NIFTY_50_close", "udiff_underlying_price_sample": str(udiff_spot) if udiff_spot else None}
        elif udiff_spot is not None:
            es[f"{fam}:{d}"] = {**expiry_day_settlement(today, d, "NIFTY", udiff_spot), "spot_source": "udiff_UndrlygPric"}
    report["expiry_settlement"] = es
    report["results"]["G7_EXPIRY_SETTLEMENT_CHARACTERIZED"] = {"status": "CHARACTERIZED" if es else "NOT_EVALUATED", "dates": len(es)}

    obs = []
    for (fam, d), bs in bars.items():
        if fam == "udiff":
            obs += [(f"udiff:{d}", b.key, b.source_id) for b in bs if b.source_id]
    master = {}
    tombs: dict[dt.date, set] = {}
    mstats = {}
    for d in (x for f, x, _ in SAMPLE if f == "master"):
        if ("master", d) not in loaded:
            continue
        header, rows = loaded[("master", d)]
        idx = {h.strip(): i for i, h in enumerate(header) if h.strip()}
        recs = {}
        tombs[d] = set()
        for r in rows:
            if not r or len(r) < 21:
                continue
            g = lambda n: r[idx[n]].strip()  # noqa: E731
            if g("DelFlg") == "Y":
                tombs[d].add(g("FinInstrmId"))
                continue
            strike = to_int(g("StrkPric"))
            recs[g("FinInstrmId")] = {"underlying": g("TckrSymb"), "type": g("FinInstrmNm"), "expiry": decode_nse_seconds(g("XpryDt")),
                                      "strike_paise": 0 if strike == -1 else strike,
                                      "cp": norm_cp(g("OptnTp")), "lot": to_int(g("NewBrdLotQty")), "min_lot": to_int(g("MinLot")), "tick": to_int(g("BidIntrvl")),
                                      "start": decode_nse_seconds(g("StartDt")) if g("StartDt") not in ("", "0") else None,
                                      "end": decode_nse_seconds(g("EndDt")) if g("EndDt") not in ("", "0") else None,
                                      "maturity": decode_nse_seconds(g("MtrtyDt")) if g("MtrtyDt") not in ("", "0") else None,
                                      "settle_method": g("SttlmMtd"), "contract_name": g("StockNm"), "del_flag": g("DelFlg"), "base_price": g("BasePric")}
        master[d] = recs
        mstats[str(d)] = {"contracts": len(recs), "deleted_token_rows": len(tombs[d]), "types": dict(Counter(v["type"] for v in recs.values())),
                          "settlement_method_by_type": {t: dict(Counter(v["settle_method"] for v in recs.values() if v["type"] == t)) for t in sorted({v["type"] for v in recs.values()})},
                          "tick_values": dict(Counter(v["tick"] for v in recs.values())), "del_flag": dict(Counter(v["del_flag"] for v in recs.values())),
                          "expiry_times_utc": dict(Counter(v["expiry"].strftime("%H:%M:%S") for v in recs.values())),
                          "nifty_option_lots": dict(Counter(v["lot"] for v in recs.values() if v["underlying"] == "NIFTY" and v["type"] == "OPTIDX")),
                          "min_lot_equals_lot": all(v["min_lot"] == v["lot"] for v in recs.values())}
        for i, v in recs.items():
            k = ContractKey(v["underlying"], {"OPTIDX": "IDO", "OPTSTK": "STO", "FUTIDX": "IDF", "FUTSTK": "STF"}.get(v["type"], f"OTHER:{v['type']}"), v["expiry"].date(),
                            v["strike_paise"] if v["strike_paise"] is not None else -1, v["cp"])
            obs.append((f"master:{d}", k, i))
    report["master"] = mstats
    idc = id_conflicts(obs)
    report["identity"] = idc
    report["results"]["G8_IDENTITY_STABLE"] = {"status": "PASS" if idc["ids_mapping_to_several_contracts"] == 0 and idc["contracts_with_several_ids"] == 0 else "FAIL", **idc}

    g9 = {}
    strike_by_underlying: Counter = Counter()
    for d in MASTER_UDIFF_PAIRS:
        if d in master and ("udiff", d) in bars:
            ud = bars[("udiff", d)]
            m = master[d]
            mismatch = Counter()
            matched = 0
            for b in ud:
                mm = m.get(b.source_id)
                if mm is None:
                    mismatch["udiff_id_master_deleted_token" if b.source_id in tombs.get(d, ()) else "udiff_id_not_in_master"] += 1
                    continue
                matched += 1
                if mm["underlying"] != b.key.underlying:
                    mismatch["underlying"] += 1
                if mm["expiry"].date() != b.key.expiry:
                    mismatch["expiry"] += 1
                if mm["strike_paise"] != b.key.strike_paise:
                    mismatch["strike"] += 1
                    strike_by_underlying[(str(d), b.key.underlying)] += 1
                if mm["cp"] != b.key.cp:
                    mismatch["cp"] += 1
                if mm["lot"] != b.lot:
                    mismatch["lot"] += 1
            g9[str(d)] = {"udiff_rows": len(ud), "matched_ids": matched, "mismatches": dict(mismatch), "master_only": len(set(m) - {b.source_id for b in ud})}
    report["master_strike_mismatch_by_underlying"] = {f"{d}:{u}": n for (d, u), n in strike_by_underlying.most_common(20)}
    report["master_vs_udiff"] = g9
    report["results"]["G9_MASTER_CONSISTENT"] = {"status": "PASS" if g9 and all(not v["mismatches"] for v in g9.values()) else "FAIL", "dates": len(g9)}

    uni = {}
    for d in MASTER_UDIFF_PAIRS:
        if d in master and ("udiff", d) in bars:
            listed = {i: v for i, v in master[d].items() if v["underlying"] == "NIFTY" and v["type"] == "OPTIDX"}
            active = {b.source_id: b for b in bars[("udiff", d)] if b.key.underlying == "NIFTY" and b.key.dtype == "IDO"}
            spot = next((b.underlying_price for b in active.values() if b.underlying_price), None)
            unobs = [v for i, v in listed.items() if i not in active]
            obs_ = [v for i, v in listed.items() if i in active]
            def band(v):
                if spot is None or v["strike_paise"] is None:
                    return "n/a"
                r = abs(Decimal(v["strike_paise"]) / 100 - spot) / spot
                return "<=5%" if r <= Decimal("0.05") else ("5-10%" if r <= Decimal("0.10") else ("10-20%" if r <= Decimal("0.20") else ">20%"))
            uni[str(d)] = {"listed_nifty_options": len(listed), "active_observed": len(active), "listed_not_in_bhavcopy": len(unobs), "active_not_in_master": len(set(active) - set(listed)),
                           "spot": str(spot) if spot else None, "listed_but_unobserved_by_moneyness": dict(Counter(band(v) for v in unobs)),
                           "observed_by_moneyness": dict(Counter(band(v) for v in obs_)),
                           "expiries_listed": len({v["expiry"].date() for v in listed.values()}), "expiries_active": len({b.key.expiry for b in active.values()}),
                           "listed_strike_step_paise": gcd_step([v["strike_paise"] for v in listed.values() if v["strike_paise"]]),
                           "active_strike_step_paise": gcd_step([b.key.strike_paise for b in active.values()]),
                           "lots_in_bhavcopy": dict(Counter(b.lot for b in active.values())),
                           "traded_share_of_active_pct": round(100 * sum(1 for b in active.values() if b.volume > 0) / len(active), 1) if active else None}
    report["nifty_universe_evidence"] = uni

    nifty = {}
    for (fam, d), bs in sorted(bars.items(), key=lambda x: (x[0][1], x[0][0])):
        opts = [b for b in bs if b.key.underlying == "NIFTY" and b.key.dtype == "IDO"]
        if not opts:
            continue
        exps = sorted({b.key.expiry for b in opts})
        nifty[f"{fam}:{d}"] = {"option_rows": len(opts), "expiries": len(exps), "nearest_expiries": [str(e) for e in exps[:4]], "distinct_strikes": len({b.key.strike_paise for b in opts}),
                               "strike_step_paise": gcd_step([b.key.strike_paise for b in opts]), "traded_rows": sum(1 for b in opts if b.volume > 0),
                               "lots": dict(Counter(b.lot for b in opts if b.lot)) or None,
                               "actual_expiry_differs": sum(1 for b in opts if b.actual_expiry and b.actual_expiry != b.key.expiry)}
    report["nifty_structure"] = nifty
    ends = {}
    for fam, d in (("udiff", D(2024, 7, 8)), ("legacy", D(2024, 7, 5))):
        if (fam, d) in bars:
            ends[f"{fam}:{d}"] = {"types": dict(Counter(b.key.dtype for b in bars[(fam, d)]))}
    vocab = Counter()
    for (fam, d), (header, rows) in loaded.items():
        if fam == "udiff":
            idx = {h.strip(): i for i, h in enumerate(header) if h.strip()}
            for r in rows:
                if r and len(r) >= len(idx):
                    vocab[("SsnId", r[idx["SsnId"]].strip())] += 1
                    vocab[("Rmks", r[idx["Rmks"]].strip())] += 1
    report["udiff_vocab"] = {f"{k[0]}={k[1]!r}": v for k, v in vocab.most_common(12)}

    report["dispositions"] = DISPOSITIONS
    statuses = {k: v["status"] for k, v in report["results"].items()}
    report["verdict"] = "FAIL" if any(s == "FAIL" for s in statuses.values()) else ("PASS_WITH_FINDINGS" if any(s in ("PASS_WITH_FINDINGS", "NOT_EVALUATED") for s in statuses.values()) else "PASS")
    report["statuses"] = statuses
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args(argv)
    if not args.no_fetch:
        sys.path.insert(0, str(ROOT))
        from backend.app.data.sources import nse_bhavcopy as nb
        session = nb._get_session()
        for family, d, why in SAMPLE:
            side = fetch_one(family, d, session)
            print(f"  {'OK ' if side.get('available', True) else '404'} {family:6} {d}  {why}", flush=True)
            time.sleep(1.0)
    report = run_gate(use_db=not args.no_db)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    out = OUT_ROOT / "report.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nGATE VERDICT: {report['verdict']}")
    for k, v in report["statuses"].items():
        print(f"  {v:20} {k}")
    print(f"\nfull report: {out}")
    return 1 if report["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())

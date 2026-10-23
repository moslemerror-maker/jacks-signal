from __future__ import annotations

import dataclasses
import datetime as dt
from typing import Sequence

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.app.research.intraday_features import market_context, mover_mechanics, price_action, vwap_levels, volatility_volume
from backend.app.research.intraday_features.bar_math import Bar, percentile_rank, today_bars
from backend.app.research.intraday_features.liquidity import rolling_median_daily_traded_value
from backend.app.research.intraday_point_in_time import select_bars_as_of

RELATIVE_VOLUME_WINDOW = 20
REALIZED_VOLATILITY_WINDOW = 20
OPENING_RANGE_BARS = 15
BREAKOUT_WITHIN_BARS = 5
LIQUIDITY_WINDOW_DAYS = 20
MIN_BARS_FOR_CORE_FEATURES = 20
STALE_AFTER_MINUTES = 10

NO_DATA = "NO_DATA"
INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
STALE = "STALE"
OK = "OK"


@dataclasses.dataclass(frozen=True)
class FeatureSnapshot:
    security_id: int
    symbol: str
    as_of_utc: dt.datetime
    trading_date: dt.date | None
    price: float | None
    return_1m: float | None
    return_5m: float | None
    return_15m: float | None
    relative_volume_20: float | None
    realized_volatility_20: float | None
    volatility_expansion_ratio: float | None
    distance_from_vwap: float | None
    session_vwap: float | None
    opening_range_high: float | None
    opening_range_low: float | None
    overnight_gap: float | None
    recent_breakout_up: bool | None
    recent_breakout_down: bool | None
    distance_from_session_open: float | None
    minutes_since_open: int | None
    relative_strength: float | None
    n_bars_today: int
    n_bars_total: int
    latest_bar_utc: dt.datetime | None
    data_quality: str


def _to_bars(views: Sequence) -> list[Bar]:
    return [Bar(v.bar_start_utc, v.trading_date, v.open, v.high, v.low, v.close, v.volume) for v in views]


def bars_up_to(session: Session, *, security_id: int, as_of_utc: dt.datetime, start_utc: dt.datetime | None = None) -> list[Bar]:
    views = select_bars_as_of(session, security_id=security_id, timeframe="1m", as_of_utc=as_of_utc, start_utc=start_utc)
    return _to_bars(views)


def compute_snapshot(session: Session, *, security_id: int, symbol: str, as_of_utc: dt.datetime) -> FeatureSnapshot:
    bars = bars_up_to(session, security_id=security_id, as_of_utc=as_of_utc)

    if not bars:
        return FeatureSnapshot(
            security_id=security_id, symbol=symbol, as_of_utc=as_of_utc, trading_date=None, price=None,
            return_1m=None, return_5m=None, return_15m=None, relative_volume_20=None, realized_volatility_20=None,
            volatility_expansion_ratio=None, distance_from_vwap=None, session_vwap=None, opening_range_high=None,
            opening_range_low=None, overnight_gap=None, recent_breakout_up=None, recent_breakout_down=None,
            distance_from_session_open=None, minutes_since_open=None,
            relative_strength=None, n_bars_today=0, n_bars_total=0, latest_bar_utc=None, data_quality=NO_DATA,
        )

    latest = bars[-1]
    opening_range = vwap_levels.opening_range_high_low(bars, OPENING_RANGE_BARS)
    opening_high, opening_low = opening_range if opening_range else (None, None)
    n_today = len(today_bars(bars))

    relative_volume_20 = volatility_volume.relative_volume(bars, RELATIVE_VOLUME_WINDOW)
    realized_volatility_20 = volatility_volume.realized_volatility(bars, REALIZED_VOLATILITY_WINDOW)

    age_minutes = (as_of_utc - latest.bar_start_utc).total_seconds() / 60.0
    if age_minutes > STALE_AFTER_MINUTES:
        data_quality = STALE
    elif relative_volume_20 is None or realized_volatility_20 is None:
        data_quality = INSUFFICIENT_HISTORY
    else:
        data_quality = OK

    return FeatureSnapshot(
        security_id=security_id, symbol=symbol, as_of_utc=as_of_utc, trading_date=latest.trading_date, price=latest.close,
        return_1m=price_action.return_over_bars(bars, 1), return_5m=price_action.return_over_bars(bars, 5),
        return_15m=price_action.return_over_bars(bars, 15), relative_volume_20=relative_volume_20,
        realized_volatility_20=realized_volatility_20, volatility_expansion_ratio=volatility_volume.volatility_expansion_ratio(bars),
        distance_from_vwap=vwap_levels.distance_from_vwap(bars), session_vwap=vwap_levels.session_vwap(bars),
        opening_range_high=opening_high, opening_range_low=opening_low, overnight_gap=price_action.overnight_gap(bars),
        recent_breakout_up=mover_mechanics.recent_breakout_up(bars, opening_high, within_bars=BREAKOUT_WITHIN_BARS) if opening_high else None,
        recent_breakout_down=mover_mechanics.recent_breakout_down(bars, opening_low, within_bars=BREAKOUT_WITHIN_BARS) if opening_low else None,
        distance_from_session_open=price_action.distance_from_session_open(bars), minutes_since_open=price_action.minutes_since_open(bars),
        relative_strength=None,
        n_bars_today=n_today, n_bars_total=len(bars), latest_bar_utc=latest.bar_start_utc, data_quality=data_quality,
    )


def compute_universe_snapshot(session: Session, *, securities: Sequence[tuple[int, str]], as_of_utc: dt.datetime) -> list[FeatureSnapshot]:
    snapshots = [compute_snapshot(session, security_id=sid, symbol=sym, as_of_utc=as_of_utc) for sid, sym in securities]
    returns_15m = {s.symbol: s.return_15m for s in snapshots if s.return_15m is not None}
    universe_avg = market_context.universe_average_return(returns_15m)
    out = []
    for s in snapshots:
        rs = market_context.relative_strength_vs_universe(s.return_15m, universe_avg) if s.return_15m is not None else None
        out.append(dataclasses.replace(s, relative_strength=rs))
    return out


def cross_sectional_rank_15m(snapshots: Sequence[FeatureSnapshot], symbol: str) -> float | None:
    by_symbol = {s.symbol: s.return_15m for s in snapshots if s.return_15m is not None}
    if symbol not in by_symbol:
        return None
    others = [v for sym, v in by_symbol.items() if sym != symbol]
    return percentile_rank(others, by_symbol[symbol])


def _daily_traded_value_history(session: Session, *, security_ids: Sequence[int], as_of_date: dt.date, lookback_days: int) -> dict[int, list[tuple[dt.date, float]]]:
    rows = session.execute(
        text(
            """
            SELECT security_id, trading_date, SUM(volume * close) AS traded_value
            FROM intraday_equity_bars
            WHERE timeframe = '1m' AND security_id = ANY(:sids) AND trading_date < :as_of_date
            GROUP BY security_id, trading_date
            ORDER BY security_id, trading_date
            """
        ),
        {"sids": list(security_ids), "as_of_date": as_of_date},
    ).fetchall()
    by_symbol: dict[int, list[tuple[dt.date, float]]] = {}
    for sid, trading_date, traded_value in rows:
        by_symbol.setdefault(sid, []).append((trading_date, float(traded_value)))
    return by_symbol


def compute_liquidity_tiers(session: Session, *, security_ids: Sequence[int], as_of_date: dt.date) -> dict[int, str | None]:
    history = _daily_traded_value_history(session, security_ids=security_ids, as_of_date=as_of_date, lookback_days=LIQUIDITY_WINDOW_DAYS)
    rolling_value: dict[int, float | None] = {}
    for sid in security_ids:
        values = [v for _, v in history.get(sid, [])]
        if len(values) < LIQUIDITY_WINDOW_DAYS:
            rolling_value[sid] = None
            continue
        series = rolling_median_daily_traded_value(values + [0.0], window=LIQUIDITY_WINDOW_DAYS)
        rolling_value[sid] = series[-1]

    defined = {sid: v for sid, v in rolling_value.items() if v is not None}
    tiers: dict[int, str | None] = {sid: None for sid in security_ids}
    if not defined:
        return tiers
    ordered = sorted(defined.items(), key=lambda kv: kv[1])
    n = len(ordered)
    for i, (sid, _) in enumerate(ordered):
        pct = (i + 1) / n
        if pct <= 1 / 3:
            tiers[sid] = "low_liquidity"
        elif pct <= 2 / 3:
            tiers[sid] = "mid_liquidity"
        else:
            tiers[sid] = "high_liquidity"
    return tiers

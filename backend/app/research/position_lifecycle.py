from __future__ import annotations

import dataclasses
import datetime
import uuid
from typing import Sequence

from backend.app.research.features.types import Bar


@dataclasses.dataclass(frozen=True)
class NextOpenOutcome:

    entry_date: datetime.date
    entry_price: float
    exit_date: datetime.date
    exit_price: float
    gross_return: float
    holding_sessions: int


@dataclasses.dataclass(frozen=True)
class SignalDay:
    instrument_id: uuid.UUID
    signal_date: datetime.date
    is_long_signal: bool
    next_open_outcome: NextOpenOutcome | None


@dataclasses.dataclass(frozen=True)
class PositionEvent:

    position_id: str
    instrument_id: uuid.UUID
    signal_date: datetime.date
    entry_date: datetime.date
    entry_price: float
    exit_date: datetime.date
    exit_price: float
    holding_sessions: int
    gross_return: float
    transaction_cost: float
    slippage: float
    net_return: float


@dataclasses.dataclass(frozen=True)
class OpenPositionAtCutoff:

    position_id: str
    instrument_id: uuid.UUID
    signal_date: datetime.date
    entry_date: datetime.date
    entry_price: float
    scheduled_exit_date: datetime.date


@dataclasses.dataclass(frozen=True)
class ForegoneSignal:

    instrument_id: uuid.UUID
    signal_date: datetime.date
    reason: str


@dataclasses.dataclass(frozen=True)
class PositionLifecycleResult:
    completed_events: list[PositionEvent]
    open_at_cutoff: list[OpenPositionAtCutoff]
    foregone_signals: list[ForegoneSignal]


@dataclasses.dataclass(frozen=True)
class DataQualityIncident:

    instrument_id: uuid.UUID
    trade_date: datetime.date
    reason: str


class MissingBarError(RuntimeError):

    def __init__(self, incident: DataQualityIncident):
        self.incident = incident
        super().__init__(
            f"data-quality incident: no close bar for instrument_id="
            f"{incident.instrument_id} on {incident.trade_date} "
            f"({incident.reason}) - a position was held this session with "
            "no real price to mark it at. MTM evaluation aborted rather "
            "than forward-filling, zero-filling, or guessing a price."
        )


@dataclasses.dataclass(frozen=True)
class DailyPortfolioRecord:

    trade_date: datetime.date
    gross_return: float
    net_return: float
    cumulative_equity: float
    exposure: float
    num_open_positions: int
    num_entries: int
    num_exits: int
    turnover: float
    cash_fraction: float


@dataclasses.dataclass(frozen=True)
class DailyPortfolioSeriesResult:

    records: list[DailyPortfolioRecord]
    price_basis: str = "UNADJUSTED_RAW_OHLC"
    corporate_action_adjustment: str = "NOT_APPLIED"


def _position_id(instrument_id: uuid.UUID, entry_date: datetime.date) -> str:
    return f"{instrument_id}:{entry_date.isoformat()}"


def simulate_position_lifecycle(
    signal_days: list[SignalDay],
    transaction_cost_bps_per_side: float,
    slippage_bps_per_side: float,
    max_simultaneous_positions: int | None,
) -> PositionLifecycleResult:
    if transaction_cost_bps_per_side < 0:
        raise ValueError("transaction_cost_bps_per_side must be >= 0")
    if slippage_bps_per_side < 0:
        raise ValueError("slippage_bps_per_side must be >= 0")
    if max_simultaneous_positions is not None and max_simultaneous_positions < 1:
        raise ValueError("max_simultaneous_positions must be >= 1 if provided")

    cost_fraction = transaction_cost_bps_per_side / 10000.0
    slippage_fraction = slippage_bps_per_side / 10000.0

    by_date: dict[datetime.date, list[SignalDay]] = {}
    for s in signal_days:
        by_date.setdefault(s.signal_date, []).append(s)

    state: dict[uuid.UUID, dict | None] = {}
    open_count = 0

    completed_events: list[PositionEvent] = []
    foregone_signals: list[ForegoneSignal] = []

    for signal_date in sorted(by_date.keys()):
        day_signals = sorted(by_date[signal_date], key=lambda s: str(s.instrument_id))

        for instrument_id, pos in list(state.items()):
            if pos is None:
                continue
            if pos["exit_date"] == signal_date:
                outcome = pos["outcome"]
                gross_return = outcome.gross_return
                total_cost = 2.0 * cost_fraction
                total_slippage = 2.0 * slippage_fraction
                completed_events.append(
                    PositionEvent(
                        position_id=_position_id(instrument_id, pos["entry_date"]),
                        instrument_id=instrument_id,
                        signal_date=pos["signal_date"],
                        entry_date=pos["entry_date"],
                        entry_price=outcome.entry_price,
                        exit_date=outcome.exit_date,
                        exit_price=outcome.exit_price,
                        holding_sessions=outcome.holding_sessions,
                        gross_return=gross_return,
                        transaction_cost=total_cost,
                        slippage=total_slippage,
                        net_return=gross_return - total_cost - total_slippage,
                    )
                )
                state[instrument_id] = None
                open_count -= 1

        candidates = [
            s
            for s in day_signals
            if s.is_long_signal and state.get(s.instrument_id) is None
        ]
        for s in candidates:
            if s.next_open_outcome is None:
                foregone_signals.append(
                    ForegoneSignal(
                        instrument_id=s.instrument_id,
                        signal_date=s.signal_date,
                        reason="unresolved_next_open_outcome",
                    )
                )
                continue
            if max_simultaneous_positions is not None and open_count >= max_simultaneous_positions:
                foregone_signals.append(
                    ForegoneSignal(
                        instrument_id=s.instrument_id,
                        signal_date=s.signal_date,
                        reason="position_cap",
                    )
                )
                continue
            state[s.instrument_id] = {
                "signal_date": s.signal_date,
                "entry_date": s.next_open_outcome.entry_date,
                "exit_date": s.next_open_outcome.exit_date,
                "outcome": s.next_open_outcome,
            }
            open_count += 1

    open_at_cutoff = [
        OpenPositionAtCutoff(
            position_id=_position_id(instrument_id, pos["entry_date"]),
            instrument_id=instrument_id,
            signal_date=pos["signal_date"],
            entry_date=pos["entry_date"],
            entry_price=pos["outcome"].entry_price,
            scheduled_exit_date=pos["exit_date"],
        )
        for instrument_id, pos in state.items()
        if pos is not None
    ]

    return PositionLifecycleResult(
        completed_events=completed_events,
        open_at_cutoff=open_at_cutoff,
        foregone_signals=foregone_signals,
    )


def event_realization_series_lump_sum_on_exit(
    events: list[PositionEvent], start_date: datetime.date, end_date: datetime.date
) -> dict[datetime.date, float]:
    if end_date < start_date:
        raise ValueError("end_date must be >= start_date")
    series: dict[datetime.date, float] = {}
    d = start_date
    one_day = datetime.timedelta(days=1)
    while d <= end_date:
        series[d] = 0.0
        d += one_day
    for event in events:
        if start_date <= event.exit_date <= end_date:
            series[event.exit_date] = series.get(event.exit_date, 0.0) + event.net_return
    return series


def compute_daily_mark_to_market_portfolio_series(
    lifecycle_result: PositionLifecycleResult,
    bars_by_instrument: dict[uuid.UUID, list[Bar]],
    session_dates: Sequence[datetime.date],
    num_sizing_slots: int,
    max_simultaneous_positions: int | None,
    transaction_cost_bps_per_side: float,
    slippage_bps_per_side: float,
) -> DailyPortfolioSeriesResult:
    if num_sizing_slots < 1:
        raise ValueError("num_sizing_slots must be >= 1")
    if max_simultaneous_positions is None:
        raise ValueError(
            "max_simultaneous_positions=None (unlimited) is not supported by "
            "the fixed-slot MTM implementation - it could silently imply "
            ">100% gross exposure. Provide an explicit positive int; "
            "unlimited position count/leverage needs its own future design."
        )
    if max_simultaneous_positions < 1:
        raise ValueError("max_simultaneous_positions must be >= 1")
    if max_simultaneous_positions > num_sizing_slots:
        raise ValueError(
            f"max_simultaneous_positions ({max_simultaneous_positions}) must be "
            f"<= num_sizing_slots ({num_sizing_slots}) - otherwise fixed "
            "1/num_sizing_slots weighting could exceed 100% gross exposure"
        )
    if transaction_cost_bps_per_side < 0:
        raise ValueError("transaction_cost_bps_per_side must be >= 0")
    if slippage_bps_per_side < 0:
        raise ValueError("slippage_bps_per_side must be >= 0")
    if not session_dates:
        raise ValueError("session_dates must not be empty")
    for i in range(1, len(session_dates)):
        if session_dates[i] <= session_dates[i - 1]:
            raise ValueError(
                "session_dates must be strictly ascending with no duplicates "
                f"(found {session_dates[i - 1]} followed by {session_dates[i]})"
            )
    session_date_set = set(session_dates)
    cutoff = session_dates[-1]

    weight = 1.0 / num_sizing_slots
    per_side_charge = (transaction_cost_bps_per_side + slippage_bps_per_side) / 10000.0

    close_by_instrument_date: dict[uuid.UUID, dict[datetime.date, float]] = {
        instrument_id: {b.trade_date: b.close for b in bars}
        for instrument_id, bars in bars_by_instrument.items()
    }

    positions: list[dict] = []
    for e in lifecycle_result.completed_events:
        positions.append(
            dict(
                instrument_id=e.instrument_id,
                entry_date=e.entry_date,
                entry_price=e.entry_price,
                exit_date=e.exit_date,
                exit_price=e.exit_price,
            )
        )
    for o in lifecycle_result.open_at_cutoff:
        positions.append(
            dict(
                instrument_id=o.instrument_id,
                entry_date=o.entry_date,
                entry_price=o.entry_price,
                exit_date=None,
                exit_price=None,
            )
        )

    for p in positions:
        if p["entry_date"] not in session_date_set:
            raise ValueError(
                f"position entry_date {p['entry_date']} for instrument_id="
                f"{p['instrument_id']} is not present in session_dates - "
                "session_dates must span the same window lifecycle_result "
                "was built from"
            )
        if p["exit_date"] is not None and p["exit_date"] not in session_date_set:
            raise ValueError(
                f"position exit_date {p['exit_date']} for instrument_id="
                f"{p['instrument_id']} is not present in session_dates - "
                "session_dates must span the same window lifecycle_result "
                "was built from"
            )

    entries_by_date: dict[datetime.date, list[dict]] = {}
    exits_by_date: dict[datetime.date, list[dict]] = {}
    for p in positions:
        entries_by_date.setdefault(p["entry_date"], []).append(p)
        if p["exit_date"] is not None:
            exits_by_date.setdefault(p["exit_date"], []).append(p)
    for bucket in (entries_by_date, exits_by_date):
        for day_positions in bucket.values():
            day_positions.sort(key=lambda p: str(p["instrument_id"]))

    active: dict[uuid.UUID, dict] = {}
    records: list[DailyPortfolioRecord] = []
    running_net_sum = 0.0

    for t in session_dates:
        gross = 0.0
        net = 0.0
        num_entries = 0
        num_exits = 0
        turnover = 0.0

        for p in exits_by_date.get(t, []):
            iid = p["instrument_id"]
            st = active.pop(iid)
            leg_gross = weight * (p["exit_price"] - st["last_price"]) / st["entry_price"]
            leg_net = leg_gross - weight * per_side_charge
            gross += leg_gross
            net += leg_net
            num_exits += 1
            turnover += weight

        entered_today: set[uuid.UUID] = set()

        for p in entries_by_date.get(t, []):
            iid = p["instrument_id"]
            entry_price = p["entry_price"]
            close_t = close_by_instrument_date.get(iid, {}).get(t)
            if close_t is None:
                raise MissingBarError(
                    DataQualityIncident(iid, t, "missing_close_on_entry_date")
                )
            leg_gross = weight * (close_t - entry_price) / entry_price
            leg_net = leg_gross - weight * per_side_charge
            gross += leg_gross
            net += leg_net
            active[iid] = dict(entry_price=entry_price, last_price=close_t)
            entered_today.add(iid)
            num_entries += 1
            turnover += weight

        for iid, st in active.items():
            if iid in entered_today:
                continue
            close_t = close_by_instrument_date.get(iid, {}).get(t)
            if close_t is None:
                raise MissingBarError(DataQualityIncident(iid, t, "missing_close_while_held"))
            leg_gross = weight * (close_t - st["last_price"]) / st["entry_price"]
            gross += leg_gross
            net += leg_gross
            st["last_price"] = close_t

        if len(active) > max_simultaneous_positions:
            raise RuntimeError(
                f"observed {len(active)} concurrently open positions on {t}, "
                f"exceeding the declared max_simultaneous_positions="
                f"{max_simultaneous_positions} - lifecycle_result was not "
                "generated with a matching cap; fixed-slot sizing cannot "
                "guarantee bounded exposure under this mismatch"
            )

        running_net_sum += net
        num_open_positions = len(active)
        exposure = num_open_positions * weight
        records.append(
            DailyPortfolioRecord(
                trade_date=t,
                gross_return=gross,
                net_return=net,
                cumulative_equity=1.0 + running_net_sum,
                exposure=exposure,
                num_open_positions=num_open_positions,
                num_entries=num_entries,
                num_exits=num_exits,
                turnover=turnover,
                cash_fraction=1.0 - exposure,
            )
        )

    return DailyPortfolioSeriesResult(records=records)

from __future__ import annotations

import dataclasses
import datetime as dt
from decimal import Decimal, InvalidOperation
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from .availability import IST, UTC
from .pipeline_reliability import FETCH_FAILURE_CATEGORIES

_FETCH_CYCLE_FAILURE_CATEGORIES = FETCH_FAILURE_CATEGORIES + ("CYCLE_CRASH",)
_EXIT_REQUIRED_FIELDS = ("gross_pnl",)


def session_window_utc(session_date: dt.date) -> tuple[dt.datetime, dt.datetime]:
    start = dt.datetime.combine(session_date, dt.time(0, 0), tzinfo=IST).astimezone(UTC)
    end = dt.datetime.combine(session_date + dt.timedelta(days=1), dt.time(0, 0), tzinfo=IST).astimezone(UTC)
    return start, end


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _categorize_reason(reason: str) -> str:
    checks = (
        ("NOT_ESTABLISHED", "predictive probability not established (risk-neutral baseline only)"),
        ("SPOT_PROXY_UNVERIFIED", "spot-proxy forward used"),
        ("below configured threshold", "risk-neutral probability below configured threshold"),
        ("insufficient IV/Greeks", "insufficient IV/Greeks data for a candidate"),
        ("candidate could not be formed", "candidate could not be formed (liquidity/data)"),
        ("no live quotes", "empty snapshot"),
        ("no quote reported an underlying_last_price", "missing underlying price"),
        ("no eligible future expiry", "no eligible future expiry"),
    )
    for needle, category in checks:
        if needle in reason:
            return category
    return "other"


@dataclasses.dataclass(frozen=True, slots=True)
class TradeRecommendationDetail:

    recommendation_id: str
    decision_timestamp: str
    shape: str | None
    selected_expiry: str | None
    forward_resolution_source: str | None
    liquidity_data_quality: str
    reasons: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class EodReport:
    underlying: str
    session_date: dt.date
    window_start_utc: dt.datetime
    window_end_utc: dt.datetime
    generated_at: dt.datetime

    successful_cycles: int
    failed_cycles: int
    total_cycles_inferred: int

    status_counts: dict[str, int]
    shape_counts: dict[str, int]
    model_versions_seen: tuple[str, ...]
    strategy_versions_seen: tuple[str, ...]
    reason_category_counts: dict[str, int]

    degraded_snapshot_count: int
    session_event_counts_by_category: dict[str, int]
    forward_resolution_counts: dict[str, int]

    paper_entry_count: int
    observation_count: int
    exit_count: int
    open_position_count: int
    closed_position_count: int
    incomplete_exit_count: int

    predictive_probability_status: str
    risk_neutral_probability_sample_count: int
    risk_neutral_probability_range: tuple[str, str] | None

    realized_pnl_status: str
    realized_pnl_summary: dict[str, Any] | None

    trade_recommendation_details: tuple[TradeRecommendationDetail, ...]


def build_eod_report(db: Session, *, underlying: str, session_date: dt.date, generated_at: dt.datetime | None = None) -> EodReport:
    window_start, window_end = session_window_utc(session_date)
    now = generated_at if generated_at is not None else dt.datetime.now(UTC)

    recs = db.execute(text(
        "SELECT id, decision_timestamp, status, shape, data_quality, model_version, strategy_version, payload "
        "FROM option_recommendations WHERE underlying=:u AND decision_timestamp >= :s AND decision_timestamp < :e "
        "ORDER BY decision_timestamp"
    ), {"u": underlying, "s": window_start, "e": window_end}).mappings().all()

    session_events = db.execute(text(
        "SELECT category, severity, count(*) AS n FROM option_session_events "
        "WHERE event_timestamp >= :s AND event_timestamp < :e GROUP BY category, severity"
    ), {"s": window_start, "e": window_end}).mappings().all()
    session_event_counts_by_category: dict[str, int] = {}
    for row in session_events:
        session_event_counts_by_category[row["category"]] = session_event_counts_by_category.get(row["category"], 0) + row["n"]
    failed_cycles = sum(n for cat, n in session_event_counts_by_category.items() if cat in _FETCH_CYCLE_FAILURE_CATEGORIES)

    status_counts: dict[str, int] = {}
    shape_counts: dict[str, int] = {}
    model_versions: set[str] = set()
    strategy_versions: set[str] = set()
    reason_category_counts: dict[str, int] = {}
    degraded_snapshot_count = 0
    forward_resolution_counts: dict[str, int] = {}
    predictive_flags: set[str] = set()
    risk_neutral_samples: list[Decimal] = []
    trade_details: list[TradeRecommendationDetail] = []
    rec_ids_by_id: dict[str, dict] = {}

    for r in recs:
        rec_ids_by_id[str(r["id"])] = r
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
        shape_key = r["shape"] if r["shape"] is not None else "N/A"
        shape_counts[shape_key] = shape_counts.get(shape_key, 0) + 1
        if r["model_version"]:
            model_versions.add(r["model_version"])
        if r["strategy_version"]:
            strategy_versions.add(r["strategy_version"])
        if r["data_quality"] != "EOD_RESOLVED":
            degraded_snapshot_count += 1

        payload = r["payload"] or {}
        selected = payload.get("selected_recommendation") or {}
        reasons_seen = selected.get("reasons") or ()
        for reason in reasons_seen:
            cat = _categorize_reason(reason)
            reason_category_counts[cat] = reason_category_counts.get(cat, 0) + 1
        reached_decision_policy = any(
            marker in reason for reason in reasons_seen
            for marker in ("risk-neutral P(net P&L>0 by expiry)", "insufficient IV/Greeks data", "PREDICTIVE_PROBABILITY_STATUS is NOT_ESTABLISHED")
        )
        if reached_decision_policy:
            if any("PREDICTIVE_PROBABILITY_STATUS is NOT_ESTABLISHED" in reason for reason in reasons_seen):
                predictive_flags.add("NOT_ESTABLISHED")
            else:
                predictive_flags.add("ESTABLISHED")

        fwd_source = (payload.get("validation_notes") or {}).get("forward_resolution_source")
        key = fwd_source if fwd_source is not None else "NOT_APPLICABLE"
        forward_resolution_counts[key] = forward_resolution_counts.get(key, 0) + 1

        for entry in payload.get("per_shape_results") or ():
            metrics = (entry.get("recommendation") or {}).get("probability_metrics") or {}
            p = _to_decimal(metrics.get("POSITIVE_NET_PNL_BY_EXPIRY"))
            if p is not None:
                risk_neutral_samples.append(p)

        if r["status"] == "TRADE":
            trade_details.append(TradeRecommendationDetail(
                recommendation_id=str(r["id"]), decision_timestamp=r["decision_timestamp"].isoformat(),
                shape=r["shape"], selected_expiry=(payload.get("validation_notes") or {}).get("expiry_used"),
                forward_resolution_source=fwd_source, liquidity_data_quality=r["data_quality"],
                reasons=tuple(selected.get("reasons") or ()),
            ))

    if not recs:
        predictive_status = "NO_RECOMMENDATIONS_IN_WINDOW"
    elif predictive_flags == {"NOT_ESTABLISHED"}:
        predictive_status = "NOT_ESTABLISHED"
    elif predictive_flags == {"ESTABLISHED"}:
        predictive_status = "ESTABLISHED"
    elif not predictive_flags:
        predictive_status = "NOT_ESTABLISHED"
    else:
        predictive_status = "MIXED"

    risk_neutral_range = (str(min(risk_neutral_samples)), str(max(risk_neutral_samples))) if risk_neutral_samples else None

    events = db.execute(text(
        "SELECT id, recommendation_id, event_type, event_timestamp, payload FROM option_paper_trade_events "
        "WHERE event_timestamp >= :s AND event_timestamp < :e ORDER BY event_timestamp"
    ), {"s": window_start, "e": window_end}).mappings().all()
    paper_entry_count = sum(1 for e in events if e["event_type"] == "PAPER_ENTRY")
    observation_count = sum(1 for e in events if e["event_type"] == "OBSERVATION")
    exit_events = [e for e in events if e["event_type"] == "EXIT"]
    exit_count = len(exit_events)
    entered_recs = {str(e["recommendation_id"]) for e in events if e["event_type"] == "PAPER_ENTRY"}
    exited_recs = {str(e["recommendation_id"]) for e in exit_events}
    open_position_count = len(entered_recs - exited_recs)
    closed_position_count = len(entered_recs & exited_recs)

    valid_exits: list[dict] = []
    incomplete_exit_count = 0
    for e in exit_events:
        p = e["payload"] or {}
        if all(f in p and _to_decimal(p[f]) is not None for f in _EXIT_REQUIRED_FIELDS):
            valid_exits.append(p)
        else:
            incomplete_exit_count += 1

    if valid_exits:
        gross = [_to_decimal(p["gross_pnl"]) for p in valid_exits]
        net_available = [n for n in (_to_decimal(p.get("net_pnl")) for p in valid_exits) if n is not None]
        if not net_available:
            net_pnl_status = "COSTS_NOT_MODELED"
        elif len(net_available) < len(valid_exits):
            net_pnl_status = "PARTIAL"
        else:
            net_pnl_status = "AVAILABLE"
        realized_pnl_status = "AVAILABLE"
        realized_pnl_summary = {
            "closed_trades": len(valid_exits),
            "total_gross_pnl": str(sum(gross, Decimal(0))),
            "positive_outcomes_gross": sum(1 for g in gross if g > 0),
            "negative_outcomes_gross": sum(1 for g in gross if g < 0),
            "flat_outcomes_gross": sum(1 for g in gross if g == 0),
            "net_pnl_status": net_pnl_status,
            "trades_with_net_pnl": len(net_available),
            "total_net_pnl": str(sum(net_available, Decimal(0))) if net_pnl_status == "AVAILABLE" else None,
            "positive_outcomes": sum(1 for n in net_available if n > 0) if net_pnl_status == "AVAILABLE" else None,
            "negative_outcomes": sum(1 for n in net_available if n < 0) if net_pnl_status == "AVAILABLE" else None,
            "flat_outcomes": sum(1 for n in net_available if n == 0) if net_pnl_status == "AVAILABLE" else None,
        }
    else:
        realized_pnl_status = "NOT_ESTABLISHED"
        realized_pnl_summary = None

    return EodReport(
        underlying=underlying, session_date=session_date, window_start_utc=window_start, window_end_utc=window_end,
        generated_at=now, successful_cycles=len(recs), failed_cycles=failed_cycles,
        total_cycles_inferred=len(recs) + failed_cycles, status_counts=status_counts, shape_counts=shape_counts,
        model_versions_seen=tuple(sorted(model_versions)), strategy_versions_seen=tuple(sorted(strategy_versions)),
        reason_category_counts=reason_category_counts, degraded_snapshot_count=degraded_snapshot_count,
        session_event_counts_by_category=session_event_counts_by_category, forward_resolution_counts=forward_resolution_counts,
        paper_entry_count=paper_entry_count, observation_count=observation_count, exit_count=exit_count,
        open_position_count=open_position_count, closed_position_count=closed_position_count,
        incomplete_exit_count=incomplete_exit_count, predictive_probability_status=predictive_status,
        risk_neutral_probability_sample_count=len(risk_neutral_samples), risk_neutral_probability_range=risk_neutral_range,
        realized_pnl_status=realized_pnl_status, realized_pnl_summary=realized_pnl_summary,
        trade_recommendation_details=tuple(trade_details),
    )


def render_markdown(report: EodReport) -> str:
    lines: list[str] = []
    lines.append(f"# AEVORYN EOD Paper Session Report — {report.underlying} — {report.session_date.isoformat()}")
    lines.append("")
    lines.append(f"_Generated {report.generated_at.isoformat()} — window {report.window_start_utc.isoformat()} to {report.window_end_utc.isoformat()} (UTC)._")
    lines.append("")
    lines.append("**This report describes what the paper-decision pipeline did during one session. It does NOT claim "
                 "strategy profitability or predictive edge from one session's data.**")
    lines.append("")

    lines.append("## Session Summary")
    lines.append(f"- Successful cycles (produced a persisted recommendation): {report.successful_cycles}")
    lines.append(f"- Failed cycles (fetch failure / crash, no recommendation produced): {report.failed_cycles}")
    lines.append(f"- Total cycles inferred: {report.total_cycles_inferred}")
    for status in ("TRADE", "WAIT", "NO_TRADE"):
        lines.append(f"- {status}: {report.status_counts.get(status, 0)}")
    lines.append("")

    lines.append("## Data Quality")
    lines.append(f"- Degraded recommendations (data_quality != EOD_RESOLVED): {report.degraded_snapshot_count}")
    if report.session_event_counts_by_category:
        for cat, n in sorted(report.session_event_counts_by_category.items()):
            lines.append(f"- Session event `{cat}`: {n}")
    else:
        lines.append("- No session events recorded (zero, not unavailable — the table was queried and found empty).")
    lines.append("- Forward-resolution mode breakdown:")
    for mode, n in sorted(report.forward_resolution_counts.items()):
        lines.append(f"  - {mode}: {n}")
    lines.append("")

    lines.append("## Recommendation Summary")
    lines.append("Candidate shapes evaluated (count of recommendations where this shape became the cycle's selected decision):")
    for shape, n in sorted(report.shape_counts.items()):
        lines.append(f"- {shape}: {n}")
    lines.append(f"- Model version(s) seen: {', '.join(report.model_versions_seen) or 'N/A'}")
    lines.append(f"- Strategy version(s) seen: {', '.join(report.strategy_versions_seen) or 'N/A'}")
    lines.append("Reason categories (bucketed, not full text):")
    for cat, n in sorted(report.reason_category_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {cat}: {n}")
    lines.append("")

    lines.append("## Paper Activity")
    lines.append(f"- PAPER_ENTRY events: {report.paper_entry_count}")
    lines.append(f"- OBSERVATION events: {report.observation_count}")
    lines.append(f"- EXIT events: {report.exit_count}")
    lines.append(f"- Open positions (entered, no exit found): {report.open_position_count}")
    lines.append(f"- Closed positions (entered and exited): {report.closed_position_count}")
    lines.append(f"- EXIT events with incomplete outcome data (not counted in performance below): {report.incomplete_exit_count}")
    lines.append("")

    lines.append("## Performance (realized paper P&L)")
    if report.realized_pnl_status == "NOT_ESTABLISHED":
        lines.append("**NOT_ESTABLISHED** — no closed paper trade in this session has recognized realized-outcome "
                     "data. This is not a claim of zero P&L; it means the data needed to compute one does not exist "
                     "yet (O2-H Item 7 does not yet implement EXIT event creation).")
    else:
        s = report.realized_pnl_summary
        lines.append(f"- Closed trades with valid outcome data (gross): {s['closed_trades']}")
        lines.append(f"- Total gross P&L: {s['total_gross_pnl']}")
        lines.append(f"- Positive / negative / flat outcomes (gross): {s['positive_outcomes_gross']} / {s['negative_outcomes_gross']} / {s['flat_outcomes_gross']}")
        lines.append(f"- Net P&L status: **{s['net_pnl_status']}** ({s['trades_with_net_pnl']} of {s['closed_trades']} closed trades have a genuinely computed net figure)")
        if s["net_pnl_status"] == "AVAILABLE":
            lines.append(f"- Total net P&L: {s['total_net_pnl']}")
            lines.append(f"- Positive / negative / flat outcomes (net): {s['positive_outcomes']} / {s['negative_outcomes']} / {s['flat_outcomes']}")
        else:
            lines.append("  Costs/slippage are not modeled in the live decision path — net P&L is intentionally left "
                         "unstated rather than approximated as gross or a fabricated zero.")
        lines.append("This is a REALIZED (gross) result from this session's own closed trades only — not a predictive-accuracy claim, and not extrapolated to any future session.")
    lines.append("")

    lines.append("## Probability / Calibration")
    lines.append(f"- Predictive (real-world) probability status recorded this session: **{report.predictive_probability_status}**")
    lines.append("  A validated real-world predictive probability model does not exist in this project yet — this is expected, not a defect.")
    if report.risk_neutral_probability_sample_count:
        lo, hi = report.risk_neutral_probability_range
        lines.append(f"- Risk-neutral (Black-76 no-arbitrage baseline) probability samples: {report.risk_neutral_probability_sample_count}, range [{lo}, {hi}]")
        lines.append("  This is a MODEL BASELINE, not a predictive-accuracy measure — see O2-G's own documentation.")
    else:
        lines.append("- No risk-neutral probability samples were recorded this session (unavailable, not zero-by-assumption).")
    lines.append("")

    lines.append("## TRADE Decision Traceability")
    if not report.trade_recommendation_details:
        lines.append("No TRADE recommendations occurred this session.")
    else:
        for d in report.trade_recommendation_details:
            lines.append(f"- `{d.recommendation_id}` at {d.decision_timestamp} — shape={d.shape}, expiry={d.selected_expiry}, "
                         f"forward={d.forward_resolution_source}, data_quality={d.liquidity_data_quality}")
            for reason in d.reasons:
                lines.append(f"  - {reason}")
    lines.append("")

    return "\n".join(lines)

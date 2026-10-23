from __future__ import annotations

import dataclasses
import datetime as dt
import json
import time
import uuid
from decimal import Decimal
from typing import Any, Callable, Literal, Protocol, Sequence, runtime_checkable

from sqlalchemy import text
from sqlalchemy.orm import Session

from . import decision_policy
from .candidate_generator import CandidateGenerationConfig, CandidateGenerationOutcome, generate_candidates
from .decision_policy import DecisionPolicyThresholds, decide_for_underlying
from .expiry_eligibility import NoEligibleExpiryError, select_eligible_expiry
from .forward import ForwardResolution, resolve_forward_from_live_future, resolve_forward_spot_proxy
from .liquidity import LiquidityAssessment
from .live_quote import LiveQuote
from . import paper_observation
from .pipeline_reliability import classify_fetch_failure
from .probability import Black76RiskNeutralDistribution, SettlementDistribution
from .recommendation import CandidateStrategy, RecommendationResult
from .structures import OptionLeg

SUPPORTED_UNDERLYINGS = ("NIFTY",)

Mode = Literal["SYNTHETIC", "LIVE_PAPER"]
_MODES = ("SYNTHETIC", "LIVE_PAPER")

DEFAULT_POLLING_INTERVAL = dt.timedelta(minutes=5)


@dataclasses.dataclass(frozen=True, slots=True)
class ChainSnapshot:

    source: str
    underlying: str
    observed_at: dt.datetime
    expiry_list_dates: tuple[dt.date, ...]
    quotes: tuple[LiveQuote, ...]
    raw_response_sha256: str | None

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        if not self.source or not self.source.strip():
            raise ValueError("source must be non-empty")


@runtime_checkable
class LiveChainSource(Protocol):

    def fetch_snapshot(self, *, as_of: dt.datetime) -> ChainSnapshot: ...


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticLiveChainSource:

    factory: Callable[[dt.datetime], ChainSnapshot]

    def fetch_snapshot(self, *, as_of: dt.datetime) -> ChainSnapshot:
        return self.factory(as_of)


@dataclasses.dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    mode: Mode
    underlying: str
    source: LiveChainSource
    candidate_config: CandidateGenerationConfig
    decision_thresholds: DecisionPolicyThresholds
    model_version: str
    strategy_version: str

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.underlying not in SUPPORTED_UNDERLYINGS:
            raise ValueError(f"underlying must be one of {SUPPORTED_UNDERLYINGS} in O2-H Item 7, got {self.underlying!r}")


CycleOutcome = Literal["PERSISTED", "FETCH_FAILURE", "PERSISTENCE_FAILURE"]


@dataclasses.dataclass(frozen=True, slots=True)
class CycleResult:
    cycle_started_at: dt.datetime
    cycle_ended_at: dt.datetime
    mode: Mode
    outcome: CycleOutcome
    recommendation_id: uuid.UUID | None
    status: str | None
    was_duplicate: bool
    paper_entry_event_id: uuid.UUID | None
    error: str | None


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, dt.timedelta):
        return obj.total_seconds()
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _dc_to_json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    return _json_safe(dataclasses.asdict(obj))


def _not_modeled(detail: str) -> dict[str, str]:
    return {"status": "NOT_MODELED", "detail": detail}


def _leg_snapshot_json(quote: LiveQuote, liquidity: LiquidityAssessment) -> dict[str, Any]:
    return {
        "contract": _json_safe(dataclasses.asdict(quote.contract)),
        "quote_status": quote.status,
        "quote_observed_at": quote.observed_at.isoformat(),
        "last_price": _json_safe(quote.last_price),
        "bid": _json_safe(quote.bid),
        "ask": _json_safe(quote.ask),
        "volume": quote.volume,
        "open_interest": quote.open_interest,
        "underlying_last_price": _json_safe(quote.underlying_last_price),
        "provider_reported_iv": _json_safe(quote.provider_reported_iv),
        "provider_reported_greeks": {k: _json_safe(v) for k, v in quote.provider_reported_greeks.items()} if quote.provider_reported_greeks else None,
        "provenance_class": quote.provenance_class,
        "liquidity_assessment": {"is_acceptable": liquidity.is_acceptable, "reasons": list(liquidity.reasons)},
    }


def _resolve_forward(snapshot: ChainSnapshot, underlying_last_price: Decimal) -> ForwardResolution:
    futures = [q for q in snapshot.quotes if q.contract.derivative_class in ("IDF", "STF") and q.is_usable]
    if futures:
        return resolve_forward_from_live_future(futures[0])
    return resolve_forward_spot_proxy(underlying_last_price)


def _distribution_factory_for(forward_resolution: ForwardResolution) -> Callable[[CandidateStrategy], SettlementDistribution | None]:
    def factory(candidate: CandidateStrategy) -> SettlementDistribution | None:
        ok_ivs = [r.iv for r in candidate.iv_context if r.status == "OK" and r.iv is not None]
        if not ok_ivs or not candidate.iv_context:
            return None
        avg_iv = sum(ok_ivs, Decimal(0)) / len(ok_ivs)
        t = candidate.iv_context[0].time_to_expiry_years
        if t is None or t <= 0:
            return None
        return Black76RiskNeutralDistribution(forward=forward_resolution.forward, volatility=avg_iv, time_to_expiry_years=t)
    return factory


_STATUS_PRIORITY = {"TRADE": 0, "WAIT": 1, "NO_TRADE": 2}


def _select_cycle_result(shaped_results: Sequence[tuple[str, RecommendationResult]]) -> tuple[str | None, RecommendationResult]:
    shape, result = min(shaped_results, key=lambda pair: _STATUS_PRIORITY[pair[1].status])
    return shape, result


def log_session_event(db: Session, *, event_timestamp: dt.datetime, severity: str, category: str, detail: str) -> None:
    db.execute(text(
        "INSERT INTO option_session_events (id, event_timestamp, severity, category, detail) VALUES (:id, :ts, :sev, :cat, :det)"
    ), {"id": uuid.uuid4(), "ts": event_timestamp, "sev": severity, "cat": category, "det": detail[:4000]})


def _build_payload(
    config: OrchestratorConfig, snapshot: ChainSnapshot, selected_shape: str | None, selected: RecommendationResult,
    per_shape: list[tuple[CandidateGenerationOutcome, RecommendationResult]], notes: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cycle": {
            "mode": config.mode,
            "source": snapshot.source,
            "underlying": snapshot.underlying,
            "observed_at": snapshot.observed_at.isoformat(),
            "raw_response_sha256": snapshot.raw_response_sha256,
            "total_quotes_in_snapshot": len(snapshot.quotes),
            "usable_quotes_in_snapshot": sum(1 for q in snapshot.quotes if q.is_usable),
            "expiry_list_dates": [d.isoformat() for d in snapshot.expiry_list_dates],
        },
        "candidate_config": _dc_to_json_safe(config.candidate_config),
        "decision_thresholds": _dc_to_json_safe(config.decision_thresholds),
        "predictive_probability_status_note": "risk-neutral baseline only - see decision_policy module docstring; NOT a validated real-world predictive probability",
        "predictive_probability_status": decision_policy.PREDICTIVE_PROBABILITY_STATUS,
        "costs_slippage_assumptions": _not_modeled(
            "O2-E cost/slippage primitives (backend/app/options/mtm.py) exist but are not wired into the live "
            "candidate generator or decision policy in this phase - no cost or slippage figure has been computed for this cycle."
        ),
        "risk_and_expected_value_status": {
            "risk": _not_modeled("No risk engine exists yet; RecommendationResult.risk is a narrative placeholder never populated by decision_policy.py in this phase."),
            "expected_value": _not_modeled("EV computation is deferred (decision_policy.py module docstring); O2-G's structure_expected_value exists but is not wired into the live decision path."),
        },
        "selected_shape": selected_shape,
        "selected_recommendation": _dc_to_json_safe(selected),
        "per_shape_results": [
            {"shape": outcome.shape, "rejected_reason": outcome.rejected_reason,
             "candidate": _dc_to_json_safe(outcome.candidate), "recommendation": _dc_to_json_safe(result),
             "decision_snapshot_legs": (
                 [_leg_snapshot_json(q, la) for q, la in zip(outcome.leg_quotes, outcome.leg_liquidity)]
                 if outcome.candidate is not None and outcome.leg_quotes else None
             )}
            for outcome, result in per_shape
        ],
        "validation_notes": notes,
    }


def _persist_and_finish(
    db: Session, *, cycle_started_at: dt.datetime, config: OrchestratorConfig, snapshot: ChainSnapshot,
    selected_shape: str | None, selected: RecommendationResult, per_shape: list[tuple[CandidateGenerationOutcome, RecommendationResult]],
    notes: dict[str, Any],
) -> CycleResult:
    payload = _build_payload(config, snapshot, selected_shape, selected, per_shape, notes)
    existing_id = db.execute(text(
        "SELECT id FROM option_recommendations WHERE underlying=:u AND decision_timestamp=:dt AND source=:s LIMIT 1"
    ), {"u": config.underlying, "dt": selected.timestamp, "s": snapshot.source}).scalar()
    was_duplicate = existing_id is not None
    paper_entry_event_id = None
    if was_duplicate:
        rec_id = existing_id
    else:
        rec_id = uuid.uuid4()
        db.execute(text(
            "INSERT INTO option_recommendations (id, decision_timestamp, underlying, status, shape, data_quality, "
            "model_version, strategy_version, source, payload) VALUES "
            "(:id, :dt, :u, :status, :shape, :dq, :mv, :sv, :src, CAST(:payload AS JSONB))"
        ), {
            "id": rec_id, "dt": selected.timestamp, "u": config.underlying, "status": selected.status,
            "shape": selected_shape, "dq": selected.data_quality, "mv": config.model_version,
            "sv": config.strategy_version, "src": snapshot.source, "payload": json.dumps(payload),
        })
        if selected.status == "TRADE" and selected.candidate is not None:
            selected_outcome = next((o for o, r in per_shape if o.shape == selected_shape), None)
            paper_entry_event_id = _create_paper_entry(db, rec_id, selected, config, selected_outcome)
    db.commit()
    return CycleResult(
        cycle_started_at=cycle_started_at, cycle_ended_at=dt.datetime.now(dt.timezone.utc), mode=config.mode,
        outcome="PERSISTED", recommendation_id=rec_id, status=selected.status, was_duplicate=was_duplicate,
        paper_entry_event_id=paper_entry_event_id, error=None,
    )


def _paper_entry_leg_snapshot(leg: OptionLeg, quote: LiveQuote | None, liquidity: LiquidityAssessment | None) -> dict[str, Any]:
    return {
        "contract": _json_safe(dataclasses.asdict(leg.contract)),
        "side": leg.side,
        "lots": leg.lots,
        "lot_size": leg.lot_size,
        "lot_size_source": leg.lot_size_source,
        "entry_price": _json_safe(leg.entry_price),
        "entry_price_source": leg.entry_price_source,
        "entry_observed_at": leg.entry_observed_at.isoformat() if leg.entry_observed_at else None,
        "provenance_class": leg.provenance_class,
        "entry_ltp": _json_safe(quote.last_price) if quote else None,
        "entry_bid": _json_safe(quote.bid) if quote else None,
        "entry_ask": _json_safe(quote.ask) if quote else None,
        "entry_volume": quote.volume if quote else None,
        "entry_open_interest": quote.open_interest if quote else None,
        "entry_quote_status": quote.status if quote else None,
        "provider_reported_iv": _json_safe(quote.provider_reported_iv) if quote else None,
        "provider_reported_greeks": {k: _json_safe(v) for k, v in quote.provider_reported_greeks.items()} if quote and quote.provider_reported_greeks else None,
        "liquidity_assessment": {"is_acceptable": liquidity.is_acceptable, "reasons": list(liquidity.reasons)} if liquidity else None,
    }


def _create_paper_entry(
    db: Session, recommendation_id: uuid.UUID, result: RecommendationResult, config: OrchestratorConfig,
    outcome: CandidateGenerationOutcome | None,
) -> uuid.UUID:
    event_id = uuid.uuid4()
    candidate = result.candidate
    assert candidate is not None
    legs = candidate.structure.legs
    leg_quotes = outcome.leg_quotes if outcome is not None else ()
    leg_liquidity = outcome.leg_liquidity if outcome is not None else ()
    leg_snapshots = [
        _paper_entry_leg_snapshot(
            leg, leg_quotes[i] if i < len(leg_quotes) else None, leg_liquidity[i] if i < len(leg_liquidity) else None,
        )
        for i, leg in enumerate(legs)
    ]
    payload = {
        "shape": outcome.shape if outcome is not None else None,
        "recommendation_timestamp": result.timestamp.isoformat(),
        "model_version": config.model_version,
        "strategy_version": config.strategy_version,
        "capital_basis": "NOT_MODELED",
        "forward_resolution": _dc_to_json_safe(candidate.forward_resolution),
        "legs": leg_snapshots,
        "candidate": _dc_to_json_safe(candidate),
        "reasons": list(result.reasons),
    }
    db.execute(text(
        "INSERT INTO option_paper_trade_events (id, recommendation_id, event_type, event_timestamp, payload) "
        "VALUES (:id, :rid, 'PAPER_ENTRY', :ts, CAST(:payload AS JSONB))"
    ), {"id": event_id, "rid": recommendation_id, "ts": result.timestamp, "payload": json.dumps(payload)})
    return event_id


def _run_observation_cycle_safely(db: Session, *, underlying: str, snapshot: ChainSnapshot, now: dt.datetime) -> None:
    try:
        paper_observation.run_observation_cycle(db, underlying=underlying, snapshot=snapshot, now=now)
        db.commit()
    except Exception as exc:
        db.rollback()
        try:
            log_session_event(db, event_timestamp=now, severity="ERROR", category="PAPER_OBSERVATION_FAILURE", detail=str(exc))
            db.commit()
        except Exception:
            db.rollback()


def run_one_cycle(*, db: Session, config: OrchestratorConfig, now: dt.datetime) -> CycleResult:
    cycle_started_at = now

    try:
        snapshot = config.source.fetch_snapshot(as_of=now)
    except Exception as exc:
        detail = str(exc)
        log_session_event(db, event_timestamp=now, severity="ERROR", category=classify_fetch_failure(detail), detail=detail)
        db.commit()
        return CycleResult(cycle_started_at=cycle_started_at, cycle_ended_at=dt.datetime.now(dt.timezone.utc), mode=config.mode,
                           outcome="FETCH_FAILURE", recommendation_id=None, status=None, was_duplicate=False,
                           paper_entry_event_id=None, error=str(exc))

    common = dict(model_version=config.model_version, strategy_version=config.strategy_version)

    if not snapshot.quotes:
        result = RecommendationResult(status="NO_TRADE", timestamp=snapshot.observed_at, underlying=config.underlying, candidate=None,
                                      reasons=("no live quotes were returned in this snapshot",), data_quality="UNAVAILABLE_INSUFFICIENT_DATA", **common)
        cycle_result = _persist_and_finish(db, cycle_started_at=cycle_started_at, config=config, snapshot=snapshot, selected_shape=None,
                                           selected=result, per_shape=[], notes={"validation_failure": "empty_snapshot"})
        _run_observation_cycle_safely(db, underlying=config.underlying, snapshot=snapshot, now=now)
        return cycle_result

    underlying_prices = sorted({q.underlying_last_price for q in snapshot.quotes if q.underlying_last_price is not None})
    if not underlying_prices:
        log_session_event(db, event_timestamp=now, severity="WARNING", category="MISSING_UNDERLYING_PRICE",
                           detail="no quote in this snapshot reported an underlying_last_price")
        result = RecommendationResult(status="NO_TRADE", timestamp=snapshot.observed_at, underlying=config.underlying, candidate=None,
                                      reasons=("no quote reported an underlying_last_price - cannot resolve any forward",),
                                      data_quality="UNAVAILABLE_INSUFFICIENT_DATA", **common)
        cycle_result = _persist_and_finish(db, cycle_started_at=cycle_started_at, config=config, snapshot=snapshot, selected_shape=None,
                                           selected=result, per_shape=[], notes={"validation_failure": "missing_underlying_price"})
        _run_observation_cycle_safely(db, underlying=config.underlying, snapshot=snapshot, now=now)
        return cycle_result
    underlying_last_price = underlying_prices[0]

    try:
        expiry = select_eligible_expiry(snapshot.expiry_list_dates, as_of=snapshot.observed_at)
    except NoEligibleExpiryError as exc:
        log_session_event(db, event_timestamp=now, severity="WARNING", category="NO_ELIGIBLE_EXPIRY", detail=str(exc))
        result = RecommendationResult(status="NO_TRADE", timestamp=snapshot.observed_at, underlying=config.underlying, candidate=None,
                                      reasons=(f"no eligible future expiry: {exc}",), data_quality="UNAVAILABLE_INSUFFICIENT_DATA", **common)
        cycle_result = _persist_and_finish(db, cycle_started_at=cycle_started_at, config=config, snapshot=snapshot, selected_shape=None,
                                           selected=result, per_shape=[], notes={"validation_failure": "no_eligible_expiry"})
        _run_observation_cycle_safely(db, underlying=config.underlying, snapshot=snapshot, now=now)
        return cycle_result

    forward_resolution = _resolve_forward(snapshot, underlying_last_price)
    expiry_quotes = [q for q in snapshot.quotes if q.contract.expiry == expiry and q.contract.is_option]
    outcomes = generate_candidates(expiry_quotes, expiry=expiry, forward_resolution=forward_resolution, config=config.candidate_config,
                                   timestamp=snapshot.observed_at, as_of=snapshot.observed_at)
    distribution_factory = _distribution_factory_for(forward_resolution)
    results = decide_for_underlying(outcomes, distribution_factory=distribution_factory, timestamp=snapshot.observed_at,
                                    underlying=config.underlying, thresholds=config.decision_thresholds,
                                    model_version=config.model_version, strategy_version=config.strategy_version)
    shaped = list(zip((o.shape for o in outcomes), results))
    selected_shape, selected = _select_cycle_result(shaped)
    cycle_result = _persist_and_finish(db, cycle_started_at=cycle_started_at, config=config, snapshot=snapshot, selected_shape=selected_shape,
                                       selected=selected, per_shape=list(zip(outcomes, results)),
                                       notes={"expiry_used": expiry.isoformat(), "forward_resolution_source": forward_resolution.source})
    _run_observation_cycle_safely(db, underlying=config.underlying, snapshot=snapshot, now=now)
    return cycle_result


@dataclasses.dataclass(frozen=True, slots=True)
class RunnerConfig:
    polling_interval: dt.timedelta = DEFAULT_POLLING_INTERVAL
    max_cycles: int | None = None
    stop_at: dt.datetime | None = None
    max_consecutive_fetch_failures: int | None = None

    def __post_init__(self) -> None:
        if self.polling_interval <= dt.timedelta(0):
            raise ValueError("polling_interval must be positive")
        if self.max_cycles is not None and self.max_cycles < 1:
            raise ValueError("max_cycles must be >= 1 if given")
        if self.stop_at is not None and self.stop_at.tzinfo is None:
            raise ValueError("stop_at must be timezone-aware")
        if self.max_consecutive_fetch_failures is not None and self.max_consecutive_fetch_failures < 1:
            raise ValueError("max_consecutive_fetch_failures must be >= 1 if given")


def run_session(
    *, session_factory: Callable[[], Session], config: OrchestratorConfig, runner_config: RunnerConfig,
    now_fn: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc), sleep_fn: Callable[[float], None] = time.sleep,
    stop_flag: Callable[[], bool] | None = None,
) -> list[CycleResult]:
    def _stop_at_reached() -> bool:
        return runner_config.stop_at is not None and now_fn() >= runner_config.stop_at

    results: list[CycleResult] = []
    cycle_num = 0
    consecutive_fetch_failures = 0
    while runner_config.max_cycles is None or cycle_num < runner_config.max_cycles:
        if stop_flag is not None and stop_flag():
            break
        if _stop_at_reached():
            break
        if (runner_config.max_consecutive_fetch_failures is not None
                and consecutive_fetch_failures >= runner_config.max_consecutive_fetch_failures):
            break
        db = session_factory()
        now = now_fn()
        try:
            try:
                result = run_one_cycle(db=db, config=config, now=now)
            except Exception as exc:
                db.rollback()
                try:
                    log_session_event(db, event_timestamp=now, severity="ERROR", category="CYCLE_CRASH", detail=str(exc))
                    db.commit()
                except Exception:
                    db.rollback()
                result = CycleResult(cycle_started_at=now, cycle_ended_at=dt.datetime.now(dt.timezone.utc), mode=config.mode,
                                     outcome="PERSISTENCE_FAILURE", recommendation_id=None, status=None, was_duplicate=False,
                                     paper_entry_event_id=None, error=str(exc))
        finally:
            db.close()
        results.append(result)
        consecutive_fetch_failures = consecutive_fetch_failures + 1 if result.outcome == "FETCH_FAILURE" else 0
        cycle_num += 1
        more_to_come = runner_config.max_cycles is None or cycle_num < runner_config.max_cycles
        failure_cap_reached = (runner_config.max_consecutive_fetch_failures is not None
                                and consecutive_fetch_failures >= runner_config.max_consecutive_fetch_failures)
        if more_to_come and not failure_cap_reached and not (stop_flag is not None and stop_flag()) and not _stop_at_reached():
            sleep_fn(runner_config.polling_interval.total_seconds())
    return results

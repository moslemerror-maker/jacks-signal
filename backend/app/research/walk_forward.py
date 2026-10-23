from __future__ import annotations

import dataclasses
import datetime
import enum
import statistics as pystats
import uuid
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Dataset,
    DatasetRow,
    ExperimentDefinition,
    ExperimentMetric,
    ExperimentRun,
    MetricSplit,
    RunPurpose,
    RunStatus,
)
from backend.app.research.experiment_tracking import (
    _capture_dependency_versions,
    _capture_git_provenance,
    _flatten_metrics,
)
from backend.app.research.experiments import _split_metrics, compute_experiment_checksum
from backend.app.research.point_in_time import resolve_label_outcome

AGGREGATE_FOLD_INDEX = -1

DB_FOLD_INDEX_OFFSET = 1


class FoldMode(str, enum.Enum):
    ROLLING = "ROLLING"
    EXPANDING = "EXPANDING"


@dataclasses.dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_start: datetime.date
    train_end: datetime.date
    validation_start: datetime.date
    validation_end: datetime.date
    test_start: datetime.date
    test_end: datetime.date


@dataclasses.dataclass(frozen=True)
class _FoldRow:

    label_resolved: bool
    realized_return: float | None
    features: dict


def generate_walk_forward_folds(
    session_dates: Sequence[datetime.date],
    train_window_sessions: int,
    validation_window_sessions: int,
    test_window_sessions: int,
    step_sessions: int,
    mode: FoldMode,
) -> list[WalkForwardFold]:
    if train_window_sessions < 1 or validation_window_sessions < 1 or test_window_sessions < 1:
        raise ValueError(
            "train_window_sessions, validation_window_sessions, and "
            "test_window_sessions must each be >= 1"
        )
    if step_sessions < 1:
        raise ValueError("step_sessions must be >= 1")
    if not session_dates:
        raise ValueError("session_dates must not be empty")
    for i in range(1, len(session_dates)):
        if session_dates[i] <= session_dates[i - 1]:
            raise ValueError(
                "session_dates must be strictly ascending with no duplicates "
                f"(found {session_dates[i - 1]} followed by {session_dates[i]})"
            )

    folds: list[WalkForwardFold] = []
    fold_index = 0
    n = len(session_dates)
    while True:
        if mode == FoldMode.ROLLING:
            train_start_idx = fold_index * step_sessions
            train_end_idx = train_start_idx + train_window_sessions - 1
        elif mode == FoldMode.EXPANDING:
            train_start_idx = 0
            train_end_idx = train_window_sessions - 1 + fold_index * step_sessions
        else:
            raise ValueError(f"unknown mode {mode!r}")

        validation_start_idx = train_end_idx + 1
        validation_end_idx = validation_start_idx + validation_window_sessions - 1
        test_start_idx = validation_end_idx + 1
        test_end_idx = test_start_idx + test_window_sessions - 1

        if test_end_idx >= n:
            break

        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_start=session_dates[train_start_idx],
                train_end=session_dates[train_end_idx],
                validation_start=session_dates[validation_start_idx],
                validation_end=session_dates[validation_end_idx],
                test_start=session_dates[test_start_idx],
                test_end=session_dates[test_end_idx],
            )
        )
        fold_index += 1

    return folds


def folds_have_overlapping_evaluation_windows(folds: Sequence[WalkForwardFold]) -> bool:
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            a, b = folds[i], folds[j]
            if a.validation_start <= b.test_end and b.validation_start <= a.test_end:
                return True
    return False


def _resolve_fold_rows(
    session: Session,
    dataset_rows: list[DatasetRow],
    label_definition_id: uuid.UUID,
    timeframe: str,
    split_end_date: datetime.date,
) -> list[_FoldRow]:
    result = []
    for row in dataset_rows:
        outcome = resolve_label_outcome(
            session,
            row.instrument_id,
            label_definition_id,
            row.entry_date,
            known_as_of=split_end_date,
            timeframe=timeframe,
        )
        result.append(
            _FoldRow(
                label_resolved=outcome is not None,
                realized_return=outcome.realized_return if outcome is not None else None,
                features=row.features,
            )
        )
    return result


def _aggregate_descriptive_metrics(per_fold_metric_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for row in per_fold_metric_rows:
        if row["metric_name"] != "mean" or row["metric_value"] is None:
            continue
        key = (row["split"], row["category"])
        groups.setdefault(key, []).append(row["metric_value"])

    aggregates = []
    for (split, category), values in groups.items():
        aggregates.append(
            dict(
                split=split,
                category=category,
                metric_name="mean",
                metric_value=pystats.fmean(values),
                sample_size=len(values),
                fold_index=AGGREGATE_FOLD_INDEX,
            )
        )
    return aggregates


def run_walk_forward_evaluation(
    session: Session,
    experiment_definition_id: uuid.UUID,
    dataset_id: uuid.UUID,
    rule_feature_definition_id: uuid.UUID,
    configuration: dict,
    cost_bps_per_trade: float,
    cost_assumptions_description: str,
    train_window_sessions: int,
    validation_window_sessions: int,
    test_window_sessions: int,
    step_sessions: int,
    mode: FoldMode,
    created_by: str,
) -> ExperimentRun:
    definition = session.get(ExperimentDefinition, experiment_definition_id)
    if definition is None:
        raise ValueError(f"unknown experiment_definition_id {experiment_definition_id}")

    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError(f"unknown dataset_id {dataset_id}")

    bound_feature_ids = {dfd.feature_definition_id for dfd in dataset.feature_definitions}
    if rule_feature_definition_id not in bound_feature_ids:
        raise ValueError(
            f"rule_feature_definition_id {rule_feature_definition_id} is not bound "
            f"to dataset {dataset_id}'s feature_definitions"
        )

    executed_at = datetime.datetime.now(datetime.timezone.utc)
    git_commit, git_dirty = _capture_git_provenance()
    dependency_versions = _capture_dependency_versions()

    session_dates = session.scalars(
        select(DatasetRow.entry_date)
        .where(DatasetRow.dataset_id == dataset_id)
        .distinct()
        .order_by(DatasetRow.entry_date)
    ).all()

    cost_assumptions = {
        "cost_bps_per_trade": cost_bps_per_trade,
        "description": cost_assumptions_description,
    }

    if not session_dates:
        folds: list[WalkForwardFold] = []
    else:
        folds = generate_walk_forward_folds(
            session_dates,
            train_window_sessions,
            validation_window_sessions,
            test_window_sessions,
            step_sessions,
            mode,
        )

    if not folds:
        status = RunStatus.FAILED
        failure_reason = (
            "insufficient trading-session history to form even one fold given "
            f"the requested window/step sizes (dataset has {len(session_dates)} "
            "distinct session dates)"
        )
        checksum = None
        walk_forward_scheme = {
            "train_window_sessions": train_window_sessions,
            "validation_window_sessions": validation_window_sessions,
            "test_window_sessions": test_window_sessions,
            "step_sessions": step_sessions,
            "mode": mode.value,
            "n_folds": 0,
            "folds": [],
            "folds_overlap": False,
            "session_dates_count": len(session_dates),
        }
        metric_rows: list[dict] = []
    else:
        overlap = folds_have_overlapping_evaluation_windows(folds)
        walk_forward_scheme = {
            "train_window_sessions": train_window_sessions,
            "validation_window_sessions": validation_window_sessions,
            "test_window_sessions": test_window_sessions,
            "step_sessions": step_sessions,
            "mode": mode.value,
            "n_folds": len(folds),
            "folds_overlap": overlap,
            "session_dates_count": len(session_dates),
            "session_dates_first": session_dates[0].isoformat(),
            "session_dates_last": session_dates[-1].isoformat(),
            "folds": [
                {
                    "fold_index": f.fold_index + DB_FOLD_INDEX_OFFSET,
                    "train_start": f.train_start.isoformat(),
                    "train_end": f.train_end.isoformat(),
                    "validation_start": f.validation_start.isoformat(),
                    "validation_end": f.validation_end.isoformat(),
                    "test_start": f.test_start.isoformat(),
                    "test_end": f.test_end.isoformat(),
                }
                for f in folds
            ],
        }

        all_dataset_rows = session.scalars(
            select(DatasetRow)
            .where(DatasetRow.dataset_id == dataset_id)
            .order_by(DatasetRow.entry_date)
        ).all()

        def _rows_in(start: datetime.date, end: datetime.date) -> list[DatasetRow]:
            return [r for r in all_dataset_rows if start <= r.entry_date <= end]

        metric_rows = []
        fold_results_for_checksum = []
        for fold in folds:
            fold_result = {}
            for split_enum, split_key, window_start, window_end in (
                (MetricSplit.TRAIN, "train", fold.train_start, fold.train_end),
                (MetricSplit.VALIDATION, "validation", fold.validation_start, fold.validation_end),
                (MetricSplit.TEST, "test", fold.test_start, fold.test_end),
            ):
                candidate_rows = _rows_in(window_start, window_end)
                fold_rows = _resolve_fold_rows(
                    session,
                    candidate_rows,
                    dataset.label_definition_id,
                    dataset.timeframe,
                    split_end_date=window_end,
                )
                split_metrics = _split_metrics(
                    fold_rows, rule_feature_definition_id, cost_bps_per_trade
                )
                fold_result[split_key] = split_metrics
                for row in _flatten_metrics(split_enum, split_metrics):
                    row["fold_index"] = fold.fold_index + DB_FOLD_INDEX_OFFSET
                    metric_rows.append(row)
            fold_results_for_checksum.append(fold_result)

        metric_rows.extend(_aggregate_descriptive_metrics(metric_rows))

        status = RunStatus.COMPLETED
        failure_reason = None

        overall_train_start = min(f.train_start for f in folds)
        overall_span_end = max(f.test_end for f in folds)

        config_for_checksum = {
            "experiment_definition_id": str(experiment_definition_id),
            "dataset_id": str(dataset_id),
            "dataset_checksum": dataset.checksum,
            "rule_feature_definition_id": str(rule_feature_definition_id),
            "run_purpose": RunPurpose.EXPLORATION.value,
            "configuration": configuration,
            "cost_assumptions": cost_assumptions,
            "walk_forward_scheme": walk_forward_scheme,
        }
        checksum = compute_experiment_checksum(
            config_for_checksum, {"folds": fold_results_for_checksum}
        )

    run = ExperimentRun(
        experiment_definition_id=experiment_definition_id,
        dataset_id=dataset_id,
        rule_feature_definition_id=rule_feature_definition_id,
        run_purpose=RunPurpose.EXPLORATION,
        git_commit=git_commit,
        git_dirty=git_dirty,
        dependency_versions=dependency_versions,
        configuration=configuration,
        cost_assumptions=cost_assumptions,
        train_start_date=(
            overall_train_start if folds else session_dates[0] if session_dates else datetime.date.min
        ),
        train_end_date=(
            overall_span_end if folds else session_dates[-1] if session_dates else datetime.date.min
        ),
        validation_start_date=None,
        validation_end_date=None,
        test_start_date=None,
        test_end_date=None,
        walk_forward_scheme=walk_forward_scheme,
        executed_at=executed_at,
        status=status,
        failure_reason=failure_reason,
        conclusion=None,
        conclusion_rationale=None,
        checksum=checksum,
        created_by=created_by,
    )
    session.add(run)
    session.flush()

    for metric_kwargs in metric_rows:
        session.add(ExperimentMetric(run_id=run.id, **metric_kwargs))
    if metric_rows:
        session.flush()

    return run

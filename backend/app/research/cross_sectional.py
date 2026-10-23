from __future__ import annotations

import dataclasses
import datetime
import statistics as pystats
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Dataset,
    DatasetRow,
    ExperimentDefinition,
    ExperimentMetric,
    ExperimentRun,
    LabelDefinition,
    MetricCategory,
    MetricSplit,
    RunPurpose,
    RunStatus,
)
from backend.app.research.experiment_tracking import (
    _capture_dependency_versions,
    _capture_git_provenance,
)
from backend.app.research.experiments import _stats, apply_rule, compute_experiment_checksum
from backend.app.research.point_in_time import resolve_label_outcome
from backend.app.research.statistics import (
    MeanTestResult,
    newey_west_mean_test,
    non_overlapping_subsample_mean_test,
)


@dataclasses.dataclass(frozen=True)
class _CrossSectionalRow:
    instrument_id: uuid.UUID
    entry_date: datetime.date
    label_resolved: bool
    realized_return: float | None
    features: dict


@dataclasses.dataclass(frozen=True)
class DailyCrossSectionalSeries:

    signal_conditional: dict[datetime.date, float]
    portfolio: dict[datetime.date, float]
    zero_signal_dates: list[datetime.date]


def compute_daily_cross_sectional_series(
    rows_by_date: dict[datetime.date, list[_CrossSectionalRow]],
    rule_feature_definition_id: uuid.UUID,
    max_simultaneous_positions: int | None = None,
) -> DailyCrossSectionalSeries:
    signal_conditional: dict[datetime.date, float] = {}
    portfolio: dict[datetime.date, float] = {}
    zero_signal_dates: list[datetime.date] = []

    for day in sorted(rows_by_date):
        day_rows = rows_by_date[day]
        active = [
            r
            for r in day_rows
            if r.label_resolved
            and apply_rule(r.features.get(str(rule_feature_definition_id))) == "LONG"
        ]
        if not active:
            zero_signal_dates.append(day)
            portfolio[day] = 0.0
            continue

        signal_conditional[day] = pystats.fmean(r.realized_return for r in active)

        portfolio_positions = active
        if max_simultaneous_positions is not None and len(active) > max_simultaneous_positions:
            portfolio_positions = sorted(active, key=lambda r: str(r.instrument_id))[
                :max_simultaneous_positions
            ]
        portfolio[day] = pystats.fmean(r.realized_return for r in portfolio_positions)

    return DailyCrossSectionalSeries(
        signal_conditional=signal_conditional,
        portfolio=portfolio,
        zero_signal_dates=zero_signal_dates,
    )


def _resolve_cross_sectional_rows(
    session: Session,
    dataset_rows: list[DatasetRow],
    label_definition_id: uuid.UUID,
    timeframe: str,
    known_as_of: datetime.date,
) -> list[_CrossSectionalRow]:
    result = []
    for row in dataset_rows:
        outcome = resolve_label_outcome(
            session,
            row.instrument_id,
            label_definition_id,
            row.entry_date,
            known_as_of=known_as_of,
            timeframe=timeframe,
        )
        result.append(
            _CrossSectionalRow(
                instrument_id=row.instrument_id,
                entry_date=row.entry_date,
                label_resolved=outcome is not None,
                realized_return=outcome.realized_return if outcome is not None else None,
                features=row.features,
            )
        )
    return result


def _mean_test_metric_rows(prefix: str, result: MeanTestResult | None) -> list[dict]:
    if result is None:
        return []
    return [
        dict(metric_name=f"{prefix}_test_statistic", metric_value=result.test_statistic),
        dict(metric_name=f"{prefix}_p_value", metric_value=result.p_value),
        dict(metric_name=f"{prefix}_ci_low", metric_value=result.confidence_interval[0]),
        dict(metric_name=f"{prefix}_ci_high", metric_value=result.confidence_interval[1]),
        dict(metric_name=f"{prefix}_n", metric_value=float(result.n)),
    ]


def run_cross_sectional_evaluation(
    session: Session,
    experiment_definition_id: uuid.UUID,
    dataset_id: uuid.UUID,
    rule_feature_definition_id: uuid.UUID,
    configuration: dict,
    train_start_date: datetime.date,
    train_end_date: datetime.date,
    created_by: str,
    max_simultaneous_positions: int | None = None,
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
    if train_end_date < train_start_date:
        raise ValueError("train_end_date must be >= train_start_date")
    if max_simultaneous_positions is not None and max_simultaneous_positions < 1:
        raise ValueError("max_simultaneous_positions must be >= 1 if provided")

    label_definition = session.get(LabelDefinition, dataset.label_definition_id)
    max_lag = label_definition.horizon_days - 1
    stride = label_definition.horizon_days

    executed_at = datetime.datetime.now(datetime.timezone.utc)
    git_commit, git_dirty = _capture_git_provenance()
    dependency_versions = _capture_dependency_versions()

    dataset_rows = session.scalars(
        select(DatasetRow)
        .where(
            DatasetRow.dataset_id == dataset_id,
            DatasetRow.entry_date >= train_start_date,
            DatasetRow.entry_date <= train_end_date,
        )
        .order_by(DatasetRow.entry_date)
    ).all()

    portfolio_convention = {
        "weighting_scheme": "equal_weight_dynamic_across_active_signals",
        "zero_signal_day_treatment": "cash_zero_return",
        "equal_weighted_across_active_signals": True,
        "unused_capital_in_cash": True,
        "max_simultaneous_positions": max_simultaneous_positions,
    }
    inference_method = {
        "estimand_1": "signal_conditional_daily_mean (descriptive only, never a portfolio claim)",
        "estimand_2": (
            "strategy_portfolio_daily_mean (hypothetical/idealized same-close "
            "return, not realizable execution - Newey-West + non-overlapping "
            "subsample HAC inference)"
        ),
        "newey_west_max_lag": max_lag,
        "non_overlapping_stride": stride,
    }
    configuration = dict(configuration)
    configuration["portfolio_convention"] = portfolio_convention
    configuration["inference_method"] = inference_method

    metric_rows: list[dict] = []

    if not dataset_rows:
        status = RunStatus.FAILED
        failure_reason = (
            "no dataset rows fall within the requested train date range - "
            "likely a mismatched dataset_id or date range"
        )
        checksum = None
    else:
        cross_rows = _resolve_cross_sectional_rows(
            session,
            dataset_rows,
            dataset.label_definition_id,
            dataset.timeframe,
            known_as_of=train_end_date,
        )
        rows_by_date: dict[datetime.date, list[_CrossSectionalRow]] = {}
        for r in cross_rows:
            rows_by_date.setdefault(r.entry_date, []).append(r)

        series = compute_daily_cross_sectional_series(
            rows_by_date, rule_feature_definition_id, max_simultaneous_positions
        )

        signal_values = [series.signal_conditional[d] for d in sorted(series.signal_conditional)]
        signal_stats = _stats(signal_values)

        portfolio_dates_sorted = sorted(series.portfolio)
        portfolio_values = [series.portfolio[d] for d in portfolio_dates_sorted]
        portfolio_stats = _stats(portfolio_values)

        nw_result: MeanTestResult | None = None
        if len(portfolio_values) >= 2 and 0 <= max_lag < len(portfolio_values):
            nw_result = newey_west_mean_test(portfolio_values, max_lag=max_lag)

        subsample_result: MeanTestResult | None = None
        if len(portfolio_values) >= 2:
            try:
                subsample_result = non_overlapping_subsample_mean_test(
                    portfolio_values, stride=stride
                )
            except ValueError:
                subsample_result = None

        for metric_name in ("count", "mean", "std", "win_rate"):
            value = signal_stats[metric_name]
            metric_rows.append(
                dict(
                    split=MetricSplit.TRAIN,
                    category=MetricCategory.DESCRIPTIVE,
                    metric_name=metric_name,
                    metric_value=None if value is None else float(value),
                    sample_size=signal_stats["count"],
                    fold_index=0,
                )
            )
        metric_rows.append(
            dict(
                split=MetricSplit.TRAIN,
                category=MetricCategory.DESCRIPTIVE,
                metric_name="zero_signal_day_count",
                metric_value=float(len(series.zero_signal_dates)),
                sample_size=len(rows_by_date),
                fold_index=0,
            )
        )

        for metric_name in ("count", "mean", "std", "win_rate"):
            value = portfolio_stats[metric_name]
            metric_rows.append(
                dict(
                    split=MetricSplit.TRAIN,
                    category=MetricCategory.HYPOTHETICAL_SIGNAL,
                    metric_name=metric_name,
                    metric_value=None if value is None else float(value),
                    sample_size=portfolio_stats["count"],
                    fold_index=0,
                )
            )

        for row in _mean_test_metric_rows("newey_west", nw_result):
            row.update(
                split=MetricSplit.TRAIN,
                category=MetricCategory.HYPOTHESIS_TEST,
                sample_size=nw_result.n if nw_result else None,
                fold_index=0,
            )
            metric_rows.append(row)
        for row in _mean_test_metric_rows("non_overlapping", subsample_result):
            row.update(
                split=MetricSplit.TRAIN,
                category=MetricCategory.HYPOTHESIS_TEST,
                sample_size=subsample_result.n if subsample_result else None,
                fold_index=0,
            )
            metric_rows.append(row)

        status = RunStatus.COMPLETED
        failure_reason = None

        config_for_checksum = {
            "experiment_definition_id": str(experiment_definition_id),
            "dataset_id": str(dataset_id),
            "dataset_checksum": dataset.checksum,
            "rule_feature_definition_id": str(rule_feature_definition_id),
            "run_purpose": RunPurpose.EXPLORATION.value,
            "configuration": configuration,
            "train_start_date": train_start_date.isoformat(),
            "train_end_date": train_end_date.isoformat(),
        }
        results_for_checksum = {
            "signal_conditional_stats": signal_stats,
            "portfolio_stats": portfolio_stats,
            "newey_west": dataclasses.asdict(nw_result) if nw_result else None,
            "non_overlapping": dataclasses.asdict(subsample_result) if subsample_result else None,
            "zero_signal_date_count": len(series.zero_signal_dates),
        }
        checksum = compute_experiment_checksum(config_for_checksum, results_for_checksum)

    run = ExperimentRun(
        experiment_definition_id=experiment_definition_id,
        dataset_id=dataset_id,
        rule_feature_definition_id=rule_feature_definition_id,
        run_purpose=RunPurpose.EXPLORATION,
        git_commit=git_commit,
        git_dirty=git_dirty,
        dependency_versions=dependency_versions,
        configuration=configuration,
        cost_assumptions={
            "cost_bps_per_trade": 0.0,
            "description": (
                "no transaction cost modeled in this stage - both estimands are "
                "gross, same-close, idealized returns"
            ),
        },
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=None,
        validation_end_date=None,
        test_start_date=None,
        test_end_date=None,
        walk_forward_scheme=None,
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

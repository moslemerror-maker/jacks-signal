from __future__ import annotations

import dataclasses
import datetime
import enum
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
from backend.app.research.cross_sectional import (
    DailyCrossSectionalSeries,
    _resolve_cross_sectional_rows,
    compute_daily_cross_sectional_series,
)
from backend.app.research.experiment_tracking import (
    _capture_dependency_versions,
    _capture_git_provenance,
    check_confirmation_uniqueness,
    count_tested_hypotheses_in_family,
)
from backend.app.research.experiments import _stats, compute_experiment_checksum
from backend.app.research.statistics import (
    MeanTestResult,
    newey_west_mean_test,
    non_overlapping_subsample_mean_test,
)
from backend.app.research.walk_forward import AGGREGATE_FOLD_INDEX

COST_MODEL_CAVEAT = (
    "SIMPLIFIED RESEARCH COST ASSUMPTION, NOT a realistic turnover/execution "
    "cost model: a flat cost_bps is deducted from the portfolio series' "
    "return on every date with an active position. Because a fresh 5-day-"
    "horizon signal is generated on every trading day, consecutive active "
    "days imply OVERLAPPING hypothetical positions that a real portfolio "
    "would not independently re-open on every single day - a correct, "
    "turnover-aware cost model requires tracking actual position "
    "lifecycles, which is NOT built here (future work, not Stage 3.6)."
)


class MultipleTestingMethod(str, enum.Enum):
    BONFERRONI = "BONFERRONI"
    FDR_BH = "FDR_BH"


def compute_adjusted_threshold(
    method: MultipleTestingMethod, alpha: float, n_hypotheses: int
) -> float:
    if n_hypotheses < 1:
        raise ValueError("n_hypotheses must be >= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if method == MultipleTestingMethod.BONFERRONI:
        return alpha / n_hypotheses
    raise NotImplementedError(f"{method} is not yet implemented")


def apply_simplified_flat_cost_to_portfolio_series(
    portfolio_series: dict[datetime.date, float], cost_bps: float
) -> dict[datetime.date, float]:
    cost_fraction = cost_bps / 10000.0
    return {
        day: (value if value == 0.0 else value - cost_fraction)
        for day, value in portfolio_series.items()
    }


def cost_sensitivity_sweep(
    portfolio_series: dict[datetime.date, float], cost_bps_values: list[float]
) -> dict[str, dict]:
    if not cost_bps_values:
        raise ValueError("cost_bps_values must not be empty")
    result = {}
    for cost_bps in cost_bps_values:
        adjusted = apply_simplified_flat_cost_to_portfolio_series(portfolio_series, cost_bps)
        values_in_order = [adjusted[d] for d in sorted(adjusted)]
        result[f"{cost_bps}bps"] = _stats(values_in_order)
    return result


def _mean_test_result_dict(result: MeanTestResult | None) -> dict | None:
    return dataclasses.asdict(result) if result is not None else None


def _split_confirmation_metrics(
    session: Session,
    dataset: Dataset,
    label_definition: LabelDefinition,
    rule_feature_definition_id: uuid.UUID,
    split_start: datetime.date,
    split_end: datetime.date,
    max_simultaneous_positions: int | None,
) -> tuple[dict, list[dict], DailyCrossSectionalSeries]:
    dataset_rows = session.scalars(
        select(DatasetRow)
        .where(
            DatasetRow.dataset_id == dataset.id,
            DatasetRow.entry_date >= split_start,
            DatasetRow.entry_date <= split_end,
        )
        .order_by(DatasetRow.entry_date)
    ).all()

    cross_rows = _resolve_cross_sectional_rows(
        session, dataset_rows, dataset.label_definition_id, dataset.timeframe, known_as_of=split_end
    )
    rows_by_date: dict[datetime.date, list] = {}
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

    max_lag = label_definition.horizon_days - 1
    stride = label_definition.horizon_days

    nw_result: MeanTestResult | None = None
    if len(portfolio_values) >= 2 and 0 <= max_lag < len(portfolio_values):
        nw_result = newey_west_mean_test(portfolio_values, max_lag=max_lag)

    subsample_result: MeanTestResult | None = None
    if len(portfolio_values) >= 2:
        try:
            subsample_result = non_overlapping_subsample_mean_test(portfolio_values, stride=stride)
        except ValueError:
            subsample_result = None

    report_section = {
        "signal_conditional_stats": signal_stats,
        "portfolio_stats": portfolio_stats,
        "newey_west": _mean_test_result_dict(nw_result),
        "non_overlapping": _mean_test_result_dict(subsample_result),
        "zero_signal_date_count": len(series.zero_signal_dates),
        "total_trading_dates": len(rows_by_date),
    }

    metric_rows: list[dict] = []
    for metric_name in ("count", "mean", "std", "win_rate"):
        metric_rows.append(
            dict(
                category=MetricCategory.DESCRIPTIVE,
                metric_name=metric_name,
                metric_value=(
                    None if signal_stats[metric_name] is None else float(signal_stats[metric_name])
                ),
                sample_size=signal_stats["count"],
                fold_index=0,
            )
        )
    for metric_name in ("count", "mean", "std", "win_rate"):
        metric_rows.append(
            dict(
                category=MetricCategory.HYPOTHETICAL_SIGNAL,
                metric_name=metric_name,
                metric_value=(
                    None
                    if portfolio_stats[metric_name] is None
                    else float(portfolio_stats[metric_name])
                ),
                sample_size=portfolio_stats["count"],
                fold_index=0,
            )
        )
    for prefix, result in (("newey_west", nw_result), ("non_overlapping", subsample_result)):
        if result is None:
            continue
        for suffix, value in (
            ("test_statistic", result.test_statistic),
            ("p_value", result.p_value),
            ("ci_low", result.confidence_interval[0]),
            ("ci_high", result.confidence_interval[1]),
            ("n", float(result.n)),
        ):
            metric_rows.append(
                dict(
                    category=MetricCategory.HYPOTHESIS_TEST,
                    metric_name=f"{prefix}_{suffix}",
                    metric_value=value,
                    sample_size=result.n,
                    fold_index=0,
                )
            )

    return report_section, metric_rows, series


def _summarize_walk_forward_consistency(walk_forward_run: ExperimentRun) -> dict:
    fold_means = [
        m.metric_value
        for m in walk_forward_run.metrics
        if m.fold_index != AGGREGATE_FOLD_INDEX
        and m.split == MetricSplit.TEST
        and m.category == MetricCategory.HYPOTHETICAL_SIGNAL
        and m.metric_name == "mean"
        and m.metric_value is not None
    ]
    if not fold_means:
        return {
            "walk_forward_run_id": str(walk_forward_run.id),
            "n_folds_with_data": 0,
            "n_folds_positive": None,
            "fold_mean_min": None,
            "fold_mean_max": None,
            "fold_mean_std": None,
        }
    n_positive = sum(1 for v in fold_means if v > 0)
    return {
        "walk_forward_run_id": str(walk_forward_run.id),
        "n_folds_with_data": len(fold_means),
        "n_folds_positive": n_positive,
        "fold_mean_min": min(fold_means),
        "fold_mean_max": max(fold_means),
        "fold_mean_std": _stats(fold_means)["std"],
    }


def run_confirmation_evaluation(
    session: Session,
    experiment_definition_id: uuid.UUID,
    dataset_id: uuid.UUID,
    rule_feature_definition_id: uuid.UUID,
    configuration: dict,
    train_start_date: datetime.date,
    train_end_date: datetime.date,
    validation_start_date: datetime.date,
    validation_end_date: datetime.date,
    test_start_date: datetime.date,
    test_end_date: datetime.date,
    alpha: float,
    multiple_testing_method: MultipleTestingMethod,
    cost_bps_sweep: list[float],
    created_by: str,
    max_simultaneous_positions: int | None = None,
    walk_forward_run_id: uuid.UUID | None = None,
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
    if train_end_date >= validation_start_date:
        raise ValueError("train_end_date must be strictly before validation_start_date")
    if validation_end_date >= test_start_date:
        raise ValueError("validation_end_date must be strictly before test_start_date")
    if not cost_bps_sweep:
        raise ValueError("cost_bps_sweep must not be empty")

    check_confirmation_uniqueness(session, experiment_definition_id, dataset_id)

    label_definition = session.get(LabelDefinition, dataset.label_definition_id)

    executed_at = datetime.datetime.now(datetime.timezone.utc)
    git_commit, git_dirty = _capture_git_provenance()
    dependency_versions = _capture_dependency_versions()

    n_hypotheses = count_tested_hypotheses_in_family(
        session,
        definition.hypothesis_family_id,
        include_definition_id=experiment_definition_id,
    )
    adjusted_threshold = compute_adjusted_threshold(multiple_testing_method, alpha, n_hypotheses)

    configuration = dict(configuration)
    configuration["inference_method"] = {
        "newey_west_max_lag": label_definition.horizon_days - 1,
        "non_overlapping_stride": label_definition.horizon_days,
        "applied_to": "complete daily portfolio series per split - never per walk-forward fold",
    }
    configuration["multiple_testing"] = {
        "alpha": alpha,
        "method": multiple_testing_method.value,
        "hypothesis_family_id": str(definition.hypothesis_family_id),
        "hypotheses_in_family": n_hypotheses,
        "adjusted_threshold": adjusted_threshold,
    }

    train_section, train_metrics, _train_series = _split_confirmation_metrics(
        session, dataset, label_definition, rule_feature_definition_id,
        train_start_date, train_end_date, max_simultaneous_positions,
    )
    validation_section, validation_metrics, _validation_series = _split_confirmation_metrics(
        session, dataset, label_definition, rule_feature_definition_id,
        validation_start_date, validation_end_date, max_simultaneous_positions,
    )
    test_section, test_metrics, test_series = _split_confirmation_metrics(
        session, dataset, label_definition, rule_feature_definition_id,
        test_start_date, test_end_date, max_simultaneous_positions,
    )

    cost_sensitivity = cost_sensitivity_sweep(test_series.portfolio, cost_bps_sweep)

    walk_forward_consistency = None
    if walk_forward_run_id is not None:
        walk_forward_run = session.get(ExperimentRun, walk_forward_run_id)
        if walk_forward_run is None:
            raise ValueError(f"unknown walk_forward_run_id {walk_forward_run_id}")
        if (
            walk_forward_run.experiment_definition_id != experiment_definition_id
            or walk_forward_run.dataset_id != dataset_id
            or walk_forward_run.run_purpose != RunPurpose.EXPLORATION
            or walk_forward_run.status != RunStatus.COMPLETED
            or walk_forward_run.walk_forward_scheme is None
        ):
            raise ValueError(
                f"walk_forward_run_id {walk_forward_run_id} is not a COMPLETED "
                "EXPLORATION walk-forward run for this same (experiment_definition_id, "
                "dataset_id) pair"
            )
        walk_forward_consistency = _summarize_walk_forward_consistency(walk_forward_run)

    inference_method = {
        "estimand": (
            "strategy_portfolio_daily_mean (hypothetical/idealized same-close "
            "return, not realizable execution - see cost_model_caveat)"
        ),
        "newey_west_max_lag": label_definition.horizon_days - 1,
        "non_overlapping_stride": label_definition.horizon_days,
        "applied_to": "complete daily portfolio series per split - never per walk-forward fold",
    }

    confirmation_report = {
        "inference_method": inference_method,
        "descriptive_performance": {
            "train": train_section["signal_conditional_stats"],
            "validation": validation_section["signal_conditional_stats"],
            "test": test_section["signal_conditional_stats"],
        },
        "portfolio_performance": {
            "train": train_section["portfolio_stats"],
            "validation": validation_section["portfolio_stats"],
            "test": test_section["portfolio_stats"],
        },
        "statistical_inference": {
            "validation": {
                "newey_west": validation_section["newey_west"],
                "non_overlapping": validation_section["non_overlapping"],
            },
            "test": {
                "newey_west": test_section["newey_west"],
                "non_overlapping": test_section["non_overlapping"],
            },
        },
        "multiple_testing": {
            "alpha": alpha,
            "method": multiple_testing_method.value,
            "hypothesis_family_id": str(definition.hypothesis_family_id),
            "hypotheses_in_family": n_hypotheses,
            "adjusted_threshold": adjusted_threshold,
        },
        "cost_sensitivity": cost_sensitivity,
        "cost_model_caveat": COST_MODEL_CAVEAT,
        "walk_forward_consistency": walk_forward_consistency,
        "implementation_shortfall": None,
        "sample_size": {
            "train": {
                "total_trading_dates": train_section["total_trading_dates"],
                "zero_signal_date_count": train_section["zero_signal_date_count"],
            },
            "validation": {
                "total_trading_dates": validation_section["total_trading_dates"],
                "zero_signal_date_count": validation_section["zero_signal_date_count"],
            },
            "test": {
                "total_trading_dates": test_section["total_trading_dates"],
                "zero_signal_date_count": test_section["zero_signal_date_count"],
            },
        },
        "unresolved_limitations": [
            "survivorship-bias caveat applies via the referenced Dataset "
            "(see Dataset.universe_caveat) - the 25-stock basket is a fixed, "
            "current-day-selected universe, not historical NIFTY membership",
            "same-close idealized label (forward_return_5d) - a realistic "
            "execution simulation (next-session-open) is not wired in yet",
            COST_MODEL_CAVEAT,
            "cross-sectional pooling assumes daily aggregation removes "
            "instrument correlation (Method B, Stage 3.4/3.5) - two-way "
            "clustering was not built and remains a documented alternative",
            (
                "walk-forward consistency not evaluated for this confirmation"
                if walk_forward_consistency is None
                else "walk-forward fold consistency is descriptive only - no "
                "inferential claim is attached to fold-to-fold agreement"
            ),
        ],
    }

    status = RunStatus.COMPLETED
    config_for_checksum = {
        "experiment_definition_id": str(experiment_definition_id),
        "dataset_id": str(dataset_id),
        "dataset_checksum": dataset.checksum,
        "rule_feature_definition_id": str(rule_feature_definition_id),
        "run_purpose": RunPurpose.CONFIRMATION.value,
        "configuration": configuration,
        "train_start_date": train_start_date.isoformat(),
        "train_end_date": train_end_date.isoformat(),
        "validation_start_date": validation_start_date.isoformat(),
        "validation_end_date": validation_end_date.isoformat(),
        "test_start_date": test_start_date.isoformat(),
        "test_end_date": test_end_date.isoformat(),
    }
    checksum = compute_experiment_checksum(config_for_checksum, confirmation_report)

    run = ExperimentRun(
        experiment_definition_id=experiment_definition_id,
        dataset_id=dataset_id,
        rule_feature_definition_id=rule_feature_definition_id,
        run_purpose=RunPurpose.CONFIRMATION,
        git_commit=git_commit,
        git_dirty=git_dirty,
        dependency_versions=dependency_versions,
        configuration=configuration,
        cost_assumptions={
            "cost_bps_sweep": cost_bps_sweep,
            "description": COST_MODEL_CAVEAT,
        },
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        walk_forward_scheme=None,
        executed_at=executed_at,
        status=status,
        failure_reason=None,
        conclusion=None,
        conclusion_rationale=None,
        confirmation_report=confirmation_report,
        checksum=checksum,
        created_by=created_by,
    )
    session.add(run)
    session.flush()

    for split_enum, metric_rows in (
        (MetricSplit.TRAIN, train_metrics),
        (MetricSplit.VALIDATION, validation_metrics),
        (MetricSplit.TEST, test_metrics),
    ):
        for kwargs in metric_rows:
            session.add(ExperimentMetric(run_id=run.id, split=split_enum, **kwargs))
    session.flush()

    return run

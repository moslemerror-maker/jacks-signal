from __future__ import annotations

import datetime
import importlib.metadata
import subprocess
import sys
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Dataset,
    DatasetRow,
    ExperimentDefinition,
    ExperimentMetric,
    ExperimentRun,
    HypothesisFamily,
    MetricCategory,
    MetricSplit,
    RunPurpose,
    RunStatus,
)
from backend.app.research.experiments import (
    _split_metrics,
    compute_experiment_checksum,
    decide_conclusion,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEPENDENCY_PACKAGES = ("sqlalchemy", "psycopg", "alembic", "pandas", "pydantic")

_CATEGORY_BY_SPLIT_KEY = {
    "descriptive_label_stats": MetricCategory.DESCRIPTIVE,
    "hypothetical_signal_performance": MetricCategory.HYPOTHETICAL_SIGNAL,
    "realizable_strategy_performance": MetricCategory.REALIZABLE_STRATEGY,
}
_METRIC_NAMES = ("count", "mean", "std", "win_rate")

_PERIOD_ARG_NAMES = (
    "validation_start_date",
    "validation_end_date",
    "test_start_date",
    "test_end_date",
)


def _capture_git_provenance() -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return commit, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(f"could not determine git provenance for this run: {exc}") from exc


def _capture_dependency_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in _DEPENDENCY_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _flatten_metrics(split: MetricSplit, split_metrics: dict) -> list[dict]:
    rows: list[dict] = []
    for key, category in _CATEGORY_BY_SPLIT_KEY.items():
        stats = split_metrics[key]
        sample_size = stats["count"]
        for metric_name in _METRIC_NAMES:
            value = stats[metric_name]
            rows.append(
                dict(
                    split=split,
                    category=category,
                    metric_name=metric_name,
                    metric_value=None if value is None else float(value),
                    sample_size=sample_size,
                )
            )
    return rows


def register_hypothesis_family(
    session: Session, name: str, description: str, created_by: str
) -> HypothesisFamily:
    existing = session.scalar(
        select(HypothesisFamily).where(HypothesisFamily.name == name)
    )
    if existing is not None:
        return existing
    family = HypothesisFamily(name=name, description=description, created_by=created_by)
    session.add(family)
    session.flush()
    return family


def count_tested_hypotheses_in_family(
    session: Session,
    hypothesis_family_id: uuid.UUID,
    include_definition_id: uuid.UUID | None = None,
) -> int:
    base_count = session.scalar(
        select(func.count(func.distinct(ExperimentDefinition.id)))
        .select_from(ExperimentDefinition)
        .join(ExperimentRun, ExperimentRun.experiment_definition_id == ExperimentDefinition.id)
        .where(ExperimentDefinition.hypothesis_family_id == hypothesis_family_id)
    )
    if include_definition_id is None:
        return base_count

    already_has_a_run = (
        session.scalar(
            select(func.count(ExperimentRun.id)).where(
                ExperimentRun.experiment_definition_id == include_definition_id
            )
        )
        > 0
    )
    return base_count if already_has_a_run else base_count + 1


def register_experiment_definition(
    session: Session,
    name: str,
    hypothesis_family_id: uuid.UUID,
    hypothesis: str,
    rule_feature_definition_id: uuid.UUID,
    rule_description: str,
    cost_model_description: str,
    created_by: str,
    version: int = 1,
) -> ExperimentDefinition:
    existing = session.scalar(
        select(ExperimentDefinition).where(
            ExperimentDefinition.name == name, ExperimentDefinition.version == version
        )
    )
    if existing is not None:
        return existing
    definition = ExperimentDefinition(
        name=name,
        version=version,
        hypothesis_family_id=hypothesis_family_id,
        hypothesis=hypothesis,
        rule_feature_definition_id=rule_feature_definition_id,
        rule_description=rule_description,
        cost_model_description=cost_model_description,
        created_by=created_by,
    )
    session.add(definition)
    session.flush()
    return definition


def check_confirmation_uniqueness(
    session: Session, experiment_definition_id: uuid.UUID, dataset_id: uuid.UUID
) -> None:
    existing_confirmation = session.scalar(
        select(ExperimentRun).where(
            ExperimentRun.experiment_definition_id == experiment_definition_id,
            ExperimentRun.dataset_id == dataset_id,
            ExperimentRun.run_purpose == RunPurpose.CONFIRMATION,
        )
    )
    if existing_confirmation is not None:
        raise ValueError(
            f"a CONFIRMATION run already exists for (experiment_definition_id="
            f"{experiment_definition_id}, dataset_id={dataset_id}): "
            f"run_id={existing_confirmation.id}. At most one CONFIRMATION run is "
            "permitted per (definition, dataset) pair (also enforced at the "
            "database level) - register a new ExperimentDefinition version for "
            "a genuinely changed hypothesis instead of retrying this one."
        )


def run_experiment(
    session: Session,
    experiment_definition_id: uuid.UUID,
    dataset_id: uuid.UUID,
    rule_feature_definition_id: uuid.UUID,
    configuration: dict,
    cost_bps_per_trade: float,
    cost_assumptions_description: str,
    run_purpose: RunPurpose,
    train_start_date: datetime.date,
    train_end_date: datetime.date,
    created_by: str,
    validation_start_date: datetime.date | None = None,
    validation_end_date: datetime.date | None = None,
    test_start_date: datetime.date | None = None,
    test_end_date: datetime.date | None = None,
    walk_forward_scheme: dict | None = None,
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

    period_args = dict(
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
    )
    if run_purpose == RunPurpose.CONFIRMATION:
        missing = [name for name in _PERIOD_ARG_NAMES if period_args[name] is None]
        if missing:
            raise ValueError(
                f"CONFIRMATION runs require all of {_PERIOD_ARG_NAMES}; missing: {missing}"
            )
        if train_end_date >= validation_start_date:
            raise ValueError("train_end_date must be strictly before validation_start_date")
        if validation_end_date >= test_start_date:
            raise ValueError("validation_end_date must be strictly before test_start_date")

        check_confirmation_uniqueness(session, experiment_definition_id, dataset_id)
    else:
        present = [name for name in _PERIOD_ARG_NAMES if period_args[name] is not None]
        if present:
            raise ValueError(
                f"EXPLORATION runs must not specify validation/test periods - only "
                f"train/diagnostic data is permitted; got non-None values for: {present}"
            )

    executed_at = datetime.datetime.now(datetime.timezone.utc)
    git_commit, git_dirty = _capture_git_provenance()
    dependency_versions = _capture_dependency_versions()

    rows = session.scalars(
        select(DatasetRow)
        .where(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index)
    ).all()

    def _in_range(row: DatasetRow, start: datetime.date, end: datetime.date) -> bool:
        return start <= row.entry_date <= end

    train_rows = [r for r in rows if _in_range(r, train_start_date, train_end_date)]
    if run_purpose == RunPurpose.CONFIRMATION:
        validation_rows = [
            r for r in rows if _in_range(r, validation_start_date, validation_end_date)
        ]
        test_rows = [r for r in rows if _in_range(r, test_start_date, test_end_date)]
    else:
        validation_rows = []
        test_rows = []

    cost_assumptions = {
        "cost_bps_per_trade": cost_bps_per_trade,
        "description": cost_assumptions_description,
    }

    metric_rows: list[dict] = []
    no_data = (
        not train_rows
        if run_purpose == RunPurpose.EXPLORATION
        else not train_rows and not validation_rows and not test_rows
    )
    if no_data:
        status = RunStatus.FAILED
        failure_reason = (
            "no dataset rows fall within the requested period(s) - likely a "
            "mismatched dataset_id or split dates"
        )
        conclusion = None
        conclusion_rationale = None
        checksum = None
    else:
        results = {
            "train": _split_metrics(train_rows, rule_feature_definition_id, cost_bps_per_trade),
        }
        metric_rows.extend(_flatten_metrics(MetricSplit.TRAIN, results["train"]))

        if run_purpose == RunPurpose.CONFIRMATION:
            results["validation"] = _split_metrics(
                validation_rows, rule_feature_definition_id, cost_bps_per_trade
            )
            results["test"] = _split_metrics(
                test_rows, rule_feature_definition_id, cost_bps_per_trade
            )
            metric_rows.extend(_flatten_metrics(MetricSplit.VALIDATION, results["validation"]))
            metric_rows.extend(_flatten_metrics(MetricSplit.TEST, results["test"]))
            conclusion, conclusion_rationale = decide_conclusion(
                results["validation"], results["test"]
            )
        else:
            conclusion, conclusion_rationale = None, None

        status = RunStatus.COMPLETED
        failure_reason = None

        config_for_checksum = {
            "experiment_definition_id": str(experiment_definition_id),
            "dataset_id": str(dataset_id),
            "dataset_checksum": dataset.checksum,
            "rule_feature_definition_id": str(rule_feature_definition_id),
            "run_purpose": run_purpose.value,
            "configuration": configuration,
            "cost_assumptions": cost_assumptions,
            "train_start_date": train_start_date.isoformat(),
            "train_end_date": train_end_date.isoformat(),
            "validation_start_date": (
                validation_start_date.isoformat() if validation_start_date else None
            ),
            "validation_end_date": (
                validation_end_date.isoformat() if validation_end_date else None
            ),
            "test_start_date": test_start_date.isoformat() if test_start_date else None,
            "test_end_date": test_end_date.isoformat() if test_end_date else None,
            "walk_forward_scheme": walk_forward_scheme,
        }
        checksum = compute_experiment_checksum(config_for_checksum, results)

    run = ExperimentRun(
        experiment_definition_id=experiment_definition_id,
        dataset_id=dataset_id,
        rule_feature_definition_id=rule_feature_definition_id,
        run_purpose=run_purpose,
        git_commit=git_commit,
        git_dirty=git_dirty,
        dependency_versions=dependency_versions,
        configuration=configuration,
        cost_assumptions=cost_assumptions,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        walk_forward_scheme=walk_forward_scheme,
        executed_at=executed_at,
        status=status,
        failure_reason=failure_reason,
        conclusion=conclusion,
        conclusion_rationale=conclusion_rationale,
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

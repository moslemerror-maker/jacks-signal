from __future__ import annotations

import datetime
import hashlib
import json
import statistics
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import Dataset, DatasetRow, Experiment, ExperimentConclusion

RULE_FEATURE_NAME = "close_vs_sma20"

HYPOTHESIS_TEXT = (
    "A simple trend-following signal - long when the 20-day close-vs-SMA "
    "trend feature (close_vs_sma20) is positive, flat otherwise - has "
    "positive mean 5-trading-day forward return, net of a simple assumed "
    "round-trip transaction cost, in both the validation and test periods."
)

RULE_DESCRIPTION_TEXT = (
    "signal = LONG if close_vs_sma20 > 0 else FLAT. Fixed threshold 0.0 "
    "(the feature's own sign boundary), not tuned. Exactly one feature, "
    "no parameter search."
)

COST_MODEL_DESCRIPTION = (
    "A flat round-trip cost (cost_bps_per_trade / 10000) is subtracted "
    "from every LONG trade's realized_return. MODELED: brokerage + STT "
    "+ a crude flat slippage allowance, bundled into one assumed bps "
    "figure. NOT MODELED: market impact, order-size-dependent slippage, "
    "bid-ask spread beyond the flat allowance, financing/margin cost "
    "for the holding period, partial fills, multi-position portfolio "
    "effects, or capital constraints. This is a minimal baseline cost "
    "assumption, not a realistic execution model."
)

MIN_TRADES_FOR_DECISION = 30


def apply_rule(feature_value: float | None) -> str:
    if feature_value is None:
        return "FLAT"
    return "LONG" if feature_value > 0 else "FLAT"


def _stats(values: list[float]) -> dict:
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": None, "std": None, "win_rate": None}
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if n >= 2 else None
    win_rate = sum(1 for v in values if v > 0) / n
    return {"count": n, "mean": mean, "std": std, "win_rate": win_rate}


def _split_metrics(
    rows: list[DatasetRow], rule_feature_definition_id: uuid.UUID, cost_bps_per_trade: float
) -> dict:
    resolved = [r for r in rows if r.label_resolved]
    descriptive_returns = [r.realized_return for r in resolved]

    long_rows = [
        r
        for r in resolved
        if apply_rule(r.features.get(str(rule_feature_definition_id))) == "LONG"
    ]
    gross_returns = [r.realized_return for r in long_rows]
    cost_fraction = cost_bps_per_trade / 10000.0
    net_returns = [r.realized_return - cost_fraction for r in long_rows]

    return {
        "descriptive_label_stats": _stats(descriptive_returns),
        "hypothetical_signal_performance": _stats(gross_returns),
        "realizable_strategy_performance": _stats(net_returns),
    }


def decide_conclusion(
    validation_split: dict, test_split: dict
) -> tuple[ExperimentConclusion, str]:
    v = validation_split["realizable_strategy_performance"]
    t = test_split["realizable_strategy_performance"]

    if v["count"] < MIN_TRADES_FOR_DECISION or t["count"] < MIN_TRADES_FOR_DECISION:
        return (
            ExperimentConclusion.NEEDS_MORE_TESTING,
            f"fewer than {MIN_TRADES_FOR_DECISION} LONG signal-days in validation "
            f"(n={v['count']}) and/or test (n={t['count']}) - insufficient sample "
            "to draw any conclusion.",
        )
    if v["mean"] > 0 and t["mean"] > 0:
        return (
            ExperimentConclusion.ACCEPT,
            f"mean net realized_return positive in both validation "
            f"({v['mean']:.6f}, n={v['count']}) and test ({t['mean']:.6f}, "
            f"n={t['count']}) - proceeds only as a candidate for further, more "
            "rigorous testing, not as a validated trading strategy.",
        )
    if v["mean"] <= 0 and t["mean"] <= 0:
        return (
            ExperimentConclusion.REJECT,
            f"mean net realized_return non-positive in both validation "
            f"({v['mean']:.6f}) and test ({t['mean']:.6f}).",
        )
    return (
        ExperimentConclusion.NEEDS_MORE_TESTING,
        f"mean net realized_return disagrees in sign between validation "
        f"({v['mean']:.6f}) and test ({t['mean']:.6f}).",
    )


def compute_experiment_checksum(config: dict, results: dict) -> str:
    serialized = json.dumps(
        {"config": config, "results": results}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def run_experiment(
    session: Session,
    dataset_id: uuid.UUID,
    rule_feature_definition_id: uuid.UUID,
    name: str,
    hypothesis: str,
    rule_description: str,
    cost_bps_per_trade: float,
    cost_model_description: str,
    train_start_date: datetime.date,
    train_end_date: datetime.date,
    validation_start_date: datetime.date,
    validation_end_date: datetime.date,
    test_start_date: datetime.date,
    test_end_date: datetime.date,
    created_by: str,
) -> Experiment:
    dataset = session.get(Dataset, dataset_id)
    if dataset is None:
        raise ValueError(f"unknown dataset_id {dataset_id}")

    bound_feature_ids = {dfd.feature_definition_id for dfd in dataset.feature_definitions}
    if rule_feature_definition_id not in bound_feature_ids:
        raise ValueError(
            f"rule_feature_definition_id {rule_feature_definition_id} is not bound "
            f"to dataset {dataset_id}'s feature_definitions - build_dataset must "
            "include it in feature_definition_ids before an experiment can use it"
        )

    if train_end_date >= validation_start_date:
        raise ValueError("train_end_date must be strictly before validation_start_date")
    if validation_end_date >= test_start_date:
        raise ValueError("validation_end_date must be strictly before test_start_date")

    rows = session.scalars(
        select(DatasetRow)
        .where(DatasetRow.dataset_id == dataset_id)
        .order_by(DatasetRow.row_index)
    ).all()

    def _in_range(row: DatasetRow, start: datetime.date, end: datetime.date) -> bool:
        return start <= row.entry_date <= end

    train_rows = [r for r in rows if _in_range(r, train_start_date, train_end_date)]
    validation_rows = [
        r for r in rows if _in_range(r, validation_start_date, validation_end_date)
    ]
    test_rows = [r for r in rows if _in_range(r, test_start_date, test_end_date)]

    results = {
        "train": _split_metrics(train_rows, rule_feature_definition_id, cost_bps_per_trade),
        "validation": _split_metrics(
            validation_rows, rule_feature_definition_id, cost_bps_per_trade
        ),
        "test": _split_metrics(test_rows, rule_feature_definition_id, cost_bps_per_trade),
    }

    conclusion, rationale = decide_conclusion(results["validation"], results["test"])

    config = {
        "dataset_id": str(dataset_id),
        "dataset_checksum": dataset.checksum,
        "rule_feature_definition_id": str(rule_feature_definition_id),
        "hypothesis": hypothesis,
        "rule_description": rule_description,
        "cost_bps_per_trade": cost_bps_per_trade,
        "cost_model_description": cost_model_description,
        "train_start_date": train_start_date.isoformat(),
        "train_end_date": train_end_date.isoformat(),
        "validation_start_date": validation_start_date.isoformat(),
        "validation_end_date": validation_end_date.isoformat(),
        "test_start_date": test_start_date.isoformat(),
        "test_end_date": test_end_date.isoformat(),
    }
    checksum = compute_experiment_checksum(config, results)

    experiment = Experiment(
        name=name,
        hypothesis=hypothesis,
        dataset_id=dataset_id,
        rule_feature_definition_id=rule_feature_definition_id,
        rule_description=rule_description,
        cost_bps_per_trade=cost_bps_per_trade,
        cost_model_description=cost_model_description,
        train_start_date=train_start_date,
        train_end_date=train_end_date,
        validation_start_date=validation_start_date,
        validation_end_date=validation_end_date,
        test_start_date=test_start_date,
        test_end_date=test_end_date,
        results=results,
        conclusion=conclusion,
        conclusion_rationale=rationale,
        checksum=checksum,
        created_by=created_by,
    )
    session.add(experiment)
    session.flush()
    return experiment

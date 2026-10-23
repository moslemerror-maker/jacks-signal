from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import FeatureDefinition, FeatureGroup

STAGE_3_FEATURE_DEFINITIONS: list[dict] = [
    dict(
        name="return_5d",
        version=1,
        feature_group=FeatureGroup.PRICE,
        description=(
            "5-trading-day close-to-close return, mainboard EQ-series "
            "bars only: close[t] / close[t-5] - 1."
        ),
        formula_ref="backend.app.research.features.price:compute_return_n_day",
        input_dependencies="market_bars.close (series=EQ)",
        lookback_window=5,
        availability_lag=0,
        timeframe="1D",
    ),
    dict(
        name="close_vs_sma20",
        version=1,
        feature_group=FeatureGroup.TREND,
        description=(
            "Close price relative to its trailing 20-day simple moving "
            "average, mainboard EQ-series bars only: "
            "close[t] / SMA20[t] - 1."
        ),
        formula_ref="backend.app.research.features.trend:compute_close_vs_sma",
        input_dependencies="market_bars.close (series=EQ)",
        lookback_window=20,
        availability_lag=0,
        timeframe="1D",
    ),
    dict(
        name="atr_14",
        version=1,
        feature_group=FeatureGroup.PRICE,
        description=(
            "14-day Average True Range (simple mean of True Range, not "
            "Wilder-smoothed), mainboard EQ-series bars only."
        ),
        formula_ref="backend.app.research.features.price:compute_atr",
        input_dependencies=(
            "market_bars.high, market_bars.low, market_bars.close (series=EQ)"
        ),
        lookback_window=14,
        availability_lag=0,
        timeframe="1D",
    ),
]


def validate_no_lagged_features(definitions: list[dict]) -> None:
    lagged = [d["name"] for d in definitions if d.get("availability_lag", 0) != 0]
    if lagged:
        raise ValueError(
            "feature definitions with non-zero availability_lag are not "
            "usable yet (no most-recently-known-as-of retrieval mode "
            f"exists - see point_in_time.py): {lagged}"
        )


def register_feature_definitions(
    session: Session,
    definitions: list[dict] = STAGE_3_FEATURE_DEFINITIONS,
    created_by: str = "stage3",
) -> dict[str, FeatureDefinition]:
    validate_no_lagged_features(definitions)
    result: dict[str, FeatureDefinition] = {}
    for spec in definitions:
        existing = session.scalar(
            select(FeatureDefinition).where(
                FeatureDefinition.name == spec["name"],
                FeatureDefinition.version == spec["version"],
            )
        )
        if existing is not None:
            result[spec["name"]] = existing
            continue
        fd = FeatureDefinition(created_by=created_by, **spec)
        session.add(fd)
        session.flush()
        result[spec["name"]] = fd
    return result

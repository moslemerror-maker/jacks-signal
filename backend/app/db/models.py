from __future__ import annotations

import datetime
import enum
import uuid

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class AssetType(str, enum.Enum):
    EQUITY = "EQUITY"
    INDEX = "INDEX"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


class OptionType(str, enum.Enum):
    CE = "CE"
    PE = "PE"


class SourceFormat(str, enum.Enum):
    NSE_LEGACY_CSV = "NSE_LEGACY_CSV"
    NSE_UDIFF = "NSE_UDIFF"
    NSE_INDEX_HISTORY = "NSE_INDEX_HISTORY"


class Instrument(Base):

    __tablename__ = "instruments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    exchange: Mapped[str] = mapped_column(String(16), nullable=False, default="NSE")
    asset_type: Mapped[AssetType] = mapped_column(Enum(AssetType), nullable=False)
    sector: Mapped[str | None] = mapped_column(String(64), nullable=True)

    underlying_instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=True
    )
    expiry_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    strike_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    option_type: Mapped[OptionType | None] = mapped_column(Enum(OptionType), nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    symbols: Mapped[list["InstrumentSymbol"]] = relationship(back_populates="instrument")
    isins: Mapped[list["InstrumentIsin"]] = relationship(back_populates="instrument")


class InstrumentSymbol(Base):

    __tablename__ = "instrument_symbols"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    effective_from: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)

    instrument: Mapped["Instrument"] = relationship(back_populates="symbols")

    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "effective_from", name="uq_instrument_symbol_period"
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_instrument_symbol_period_order",
        ),
    )


class InstrumentIsin(Base):

    __tablename__ = "instrument_isins"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    isin: Mapped[str] = mapped_column(String(12), nullable=False, unique=True)
    effective_from: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)

    instrument: Mapped["Instrument"] = relationship(back_populates="isins")

    __table_args__ = (
        CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_instrument_isin_period_order",
        ),
    )


class MarketBar(Base):

    __tablename__ = "market_bars"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    trade_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    series: Mapped[str] = mapped_column(String(8), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    last_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    prev_close: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    turnover: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_trades: Mapped[int | None] = mapped_column(Integer, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_format: Mapped[SourceFormat] = mapped_column(Enum(SourceFormat), nullable=False)
    source_file: Mapped[str] = mapped_column(String(128), nullable=False)
    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "trade_date",
            "series",
            "timeframe",
            name="uq_market_bar_identity",
        ),
        CheckConstraint("high >= low", name="ck_market_bar_high_ge_low"),
        CheckConstraint(
            "open BETWEEN low AND high", name="ck_market_bar_open_in_range"
        ),
        CheckConstraint(
            "close BETWEEN low AND high", name="ck_market_bar_close_in_range"
        ),
    )


class IntradayEquityBar(Base):

    __tablename__ = "intraday_equity_bars"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    provider: Mapped[str] = mapped_column(String(16), nullable=False)
    exchange_segment: Mapped[str] = mapped_column(String(16), nullable=False)
    security_id: Mapped[int] = mapped_column(Integer, nullable=False)
    trading_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)

    bar_start_utc: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    trading_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    raw_timestamp_epoch: Mapped[int] = mapped_column(BigInteger, nullable=False)

    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)

    source_provider: Mapped[str] = mapped_column(String(16), nullable=False)
    source_dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    source_key: Mapped[str] = mapped_column(String(128), nullable=False)
    source_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    normalizer_version: Mapped[str] = mapped_column(String(64), nullable=False)

    ingested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "provider", "exchange_segment", "security_id", "timeframe", "bar_start_utc",
            name="uq_intraday_equity_bar_identity",
        ),
        CheckConstraint("high >= low", name="ck_intraday_equity_bar_high_ge_low"),
        CheckConstraint(
            "open BETWEEN low AND high", name="ck_intraday_equity_bar_open_in_range"
        ),
        CheckConstraint(
            "close BETWEEN low AND high", name="ck_intraday_equity_bar_close_in_range"
        ),
        CheckConstraint(
            "open > 0 AND high > 0 AND low > 0 AND close > 0",
            name="ck_intraday_equity_bar_positive_price",
        ),
        CheckConstraint(
            "volume IS NULL OR volume >= 0", name="ck_intraday_equity_bar_volume_nonneg"
        ),
        Index(
            "ix_intraday_equity_bar_security_date", "security_id", "trading_date"
        ),
    )


class StockCollectionEvent(Base):

    __tablename__ = "stock_collection_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    event_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    trading_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str] = mapped_column(String(48), nullable=False)
    symbols_expected: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbols_succeeded: Mapped[int | None] = mapped_column(Integer, nullable=True)
    symbols_failed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latest_completed_minute: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    bars_persisted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detail: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        Index("ix_stock_collection_event_trading_date", "trading_date"),
        Index("ix_stock_collection_event_category", "category"),
    )


class DefinitionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    RETIRED = "RETIRED"


class FeatureGroup(str, enum.Enum):

    PRICE = "PRICE"
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    VOLUME = "VOLUME"
    STRUCTURE = "STRUCTURE"
    CONTEXT = "CONTEXT"
    DERIVATIVES = "DERIVATIVES"


class FeatureDefinition(Base):

    __tablename__ = "feature_definitions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_group: Mapped[FeatureGroup] = mapped_column(Enum(FeatureGroup), nullable=False)
    description: Mapped[str] = mapped_column(String(512), nullable=False)

    formula_ref: Mapped[str] = mapped_column(String(256), nullable=False)
    input_dependencies: Mapped[str] = mapped_column(String(256), nullable=False)
    lookback_window: Mapped[int | None] = mapped_column(Integer, nullable=True)

    availability_lag: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")

    status: Mapped[DefinitionStatus] = mapped_column(
        Enum(DefinitionStatus), nullable=False, default=DefinitionStatus.ACTIVE
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    values: Mapped[list["FeatureValue"]] = relationship(back_populates="definition")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_feature_definition_name_version"),
    )


class FeatureValue(Base):

    __tablename__ = "feature_values"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    feature_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feature_definitions.id"), nullable=False
    )
    computed_for_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")
    value: Mapped[float] = mapped_column(Float, nullable=False)
    available_at: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    definition: Mapped["FeatureDefinition"] = relationship(back_populates="values")

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "feature_definition_id",
            "computed_for_date",
            "timeframe",
            name="uq_feature_value_identity",
        ),
        CheckConstraint(
            "available_at >= computed_for_date", name="ck_feature_value_available_at_order"
        ),
    )


class LabelDefinition(Base):

    __tablename__ = "label_definitions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    description: Mapped[str] = mapped_column(String(512), nullable=False)
    formula_ref: Mapped[str] = mapped_column(String(256), nullable=False)

    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)

    target_stop_definition: Mapped[str | None] = mapped_column(String(256), nullable=True)

    same_bar_resolution_rule: Mapped[str | None] = mapped_column(String(256), nullable=True)

    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")
    status: Mapped[DefinitionStatus] = mapped_column(
        Enum(DefinitionStatus), nullable=False, default=DefinitionStatus.ACTIVE
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    values: Mapped[list["LabelValue"]] = relationship(back_populates="definition")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_label_definition_name_version"),
    )


class LabelValue(Base):

    __tablename__ = "label_values"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    label_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("label_definitions.id"), nullable=False
    )
    entry_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    horizon_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")

    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_target: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_stop: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    mfe: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_time_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    definition: Mapped["LabelDefinition"] = relationship(back_populates="values")

    __table_args__ = (
        UniqueConstraint(
            "instrument_id",
            "label_definition_id",
            "entry_date",
            "timeframe",
            name="uq_label_value_identity",
        ),
        CheckConstraint(
            "horizon_end_date >= entry_date", name="ck_label_value_horizon_order"
        ),
    )


class Dataset(Base):

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    label_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("label_definitions.id"), nullable=False
    )
    start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False, default="1D")

    known_as_of: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    universe_caveat: Mapped[str] = mapped_column(String(1024), nullable=False)

    source_data_watermark: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    rows_with_any_missing_feature: Mapped[int] = mapped_column(Integer, nullable=False)
    rows_with_unresolved_label: Mapped[int] = mapped_column(Integer, nullable=False)

    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    label_definition: Mapped["LabelDefinition"] = relationship()
    instruments: Mapped[list["DatasetInstrument"]] = relationship(back_populates="dataset")
    feature_definitions: Mapped[list["DatasetFeatureDefinition"]] = relationship(
        back_populates="dataset"
    )
    rows: Mapped[list["DatasetRow"]] = relationship(back_populates="dataset")

    __table_args__ = (
        CheckConstraint("end_date >= start_date", name="ck_dataset_date_range_order"),
    )


class DatasetInstrument(Base):

    __tablename__ = "dataset_instruments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )

    dataset: Mapped["Dataset"] = relationship(back_populates="instruments")

    __table_args__ = (
        UniqueConstraint("dataset_id", "instrument_id", name="uq_dataset_instrument"),
    )


class DatasetFeatureDefinition(Base):

    __tablename__ = "dataset_feature_definitions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
    )
    feature_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feature_definitions.id"), nullable=False
    )

    dataset: Mapped["Dataset"] = relationship(back_populates="feature_definitions")

    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "feature_definition_id", name="uq_dataset_feature_definition"
        ),
    )


class DatasetRow(Base):

    __tablename__ = "dataset_rows"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
    )
    row_index: Mapped[int] = mapped_column(Integer, nullable=False)
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    entry_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    features: Mapped[dict] = mapped_column(JSONB, nullable=False)

    label_resolved: Mapped[bool] = mapped_column(Boolean, nullable=False)
    realized_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    hit_target: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    hit_stop: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    mfe: Mapped[float | None] = mapped_column(Float, nullable=True)
    mae: Mapped[float | None] = mapped_column(Float, nullable=True)
    holding_time_days: Mapped[int | None] = mapped_column(Integer, nullable=True)

    dataset: Mapped["Dataset"] = relationship(back_populates="rows")

    __table_args__ = (
        UniqueConstraint("dataset_id", "row_index", name="uq_dataset_row_index"),
        UniqueConstraint(
            "dataset_id", "instrument_id", "entry_date", name="uq_dataset_row_identity"
        ),
    )


class ExperimentConclusion(str, enum.Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    NEEDS_MORE_TESTING = "NEEDS_MORE_TESTING"


class Experiment(Base):

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    hypothesis: Mapped[str] = mapped_column(String(1024), nullable=False)

    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
    )
    rule_feature_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feature_definitions.id"), nullable=False
    )
    rule_description: Mapped[str] = mapped_column(String(512), nullable=False)

    cost_bps_per_trade: Mapped[float] = mapped_column(Float, nullable=False)
    cost_model_description: Mapped[str] = mapped_column(String(1024), nullable=False)

    train_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    train_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    validation_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    validation_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    test_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    test_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)

    results: Mapped[dict] = mapped_column(JSONB, nullable=False)
    conclusion: Mapped[ExperimentConclusion] = mapped_column(
        Enum(ExperimentConclusion), nullable=False
    )
    conclusion_rationale: Mapped[str] = mapped_column(String(1024), nullable=False)

    checksum: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    dataset: Mapped["Dataset"] = relationship()
    rule_feature_definition: Mapped["FeatureDefinition"] = relationship()

    __table_args__ = (
        CheckConstraint(
            "train_end_date >= train_start_date", name="ck_experiment_train_order"
        ),
        CheckConstraint(
            "validation_end_date >= validation_start_date",
            name="ck_experiment_validation_order",
        ),
        CheckConstraint(
            "test_end_date >= test_start_date", name="ck_experiment_test_order"
        ),
        CheckConstraint(
            "validation_start_date > train_end_date",
            name="ck_experiment_train_before_validation",
        ),
        CheckConstraint(
            "test_start_date > validation_end_date",
            name="ck_experiment_validation_before_test",
        ),
    )


class RunStatus(str, enum.Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class MetricSplit(str, enum.Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"


class MetricCategory(str, enum.Enum):
    DESCRIPTIVE = "DESCRIPTIVE"
    HYPOTHESIS_TEST = "HYPOTHESIS_TEST"
    HYPOTHETICAL_SIGNAL = "HYPOTHETICAL_SIGNAL"
    REALIZABLE_STRATEGY = "REALIZABLE_STRATEGY"


class RunPurpose(str, enum.Enum):

    EXPLORATION = "EXPLORATION"
    CONFIRMATION = "CONFIRMATION"


class HypothesisFamily(Base):

    __tablename__ = "hypothesis_families"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    definitions: Mapped[list["ExperimentDefinition"]] = relationship(
        back_populates="hypothesis_family"
    )


class ExperimentDefinition(Base):

    __tablename__ = "experiment_definitions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    hypothesis_family_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("hypothesis_families.id"), nullable=False
    )
    hypothesis: Mapped[str] = mapped_column(String(1024), nullable=False)
    rule_feature_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feature_definitions.id"), nullable=False
    )
    rule_description: Mapped[str] = mapped_column(String(512), nullable=False)
    cost_model_description: Mapped[str] = mapped_column(String(1024), nullable=False)

    status: Mapped[DefinitionStatus] = mapped_column(
        Enum(DefinitionStatus), nullable=False, default=DefinitionStatus.ACTIVE
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    hypothesis_family: Mapped["HypothesisFamily"] = relationship(back_populates="definitions")
    rule_feature_definition: Mapped["FeatureDefinition"] = relationship()
    runs: Mapped[list["ExperimentRun"]] = relationship(back_populates="experiment_definition")

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_experiment_definition_name_version"),
    )


class ExperimentRun(Base):

    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_definitions.id"), nullable=False
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
    )
    rule_feature_definition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("feature_definitions.id"), nullable=False
    )
    run_purpose: Mapped[RunPurpose] = mapped_column(Enum(RunPurpose), nullable=False)

    git_commit: Mapped[str] = mapped_column(String(40), nullable=False)
    git_dirty: Mapped[bool] = mapped_column(Boolean, nullable=False)
    dependency_versions: Mapped[dict] = mapped_column(JSONB, nullable=False)

    configuration: Mapped[dict] = mapped_column(JSONB, nullable=False)
    cost_assumptions: Mapped[dict] = mapped_column(JSONB, nullable=False)

    train_start_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    train_end_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    validation_start_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    validation_end_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    test_start_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    test_end_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)

    walk_forward_scheme: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    executed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), nullable=False)
    failure_reason: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    conclusion: Mapped[ExperimentConclusion | None] = mapped_column(
        Enum(ExperimentConclusion), nullable=True
    )
    conclusion_rationale: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    confirmation_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    experiment_definition: Mapped["ExperimentDefinition"] = relationship(back_populates="runs")
    dataset: Mapped["Dataset"] = relationship()
    rule_feature_definition: Mapped["FeatureDefinition"] = relationship()
    metrics: Mapped[list["ExperimentMetric"]] = relationship(back_populates="run")

    __table_args__ = (
        CheckConstraint(
            "train_end_date >= train_start_date", name="ck_experiment_run_train_order"
        ),
        CheckConstraint(
            "validation_end_date >= validation_start_date",
            name="ck_experiment_run_validation_order",
        ),
        CheckConstraint(
            "test_end_date >= test_start_date", name="ck_experiment_run_test_order"
        ),
        CheckConstraint(
            "validation_start_date > train_end_date",
            name="ck_experiment_run_train_before_validation",
        ),
        CheckConstraint(
            "test_start_date > validation_end_date",
            name="ck_experiment_run_validation_before_test",
        ),
        CheckConstraint(
            "(status = 'FAILED' AND conclusion IS NULL) OR "
            "(status = 'COMPLETED' AND run_purpose = 'CONFIRMATION' "
            "AND (conclusion IS NOT NULL OR confirmation_report IS NOT NULL)) OR "
            "(status = 'COMPLETED' AND run_purpose = 'EXPLORATION' AND conclusion IS NULL)",
            name="ck_experiment_run_conclusion_matches_status",
        ),
        CheckConstraint(
            "run_purpose <> 'CONFIRMATION' OR "
            "(validation_start_date IS NOT NULL AND validation_end_date IS NOT NULL "
            "AND test_start_date IS NOT NULL AND test_end_date IS NOT NULL)",
            name="ck_experiment_run_confirmation_requires_periods",
        ),
        Index(
            "uq_experiment_run_one_confirmation_per_definition_dataset",
            "experiment_definition_id",
            "dataset_id",
            unique=True,
            postgresql_where=text("run_purpose = 'CONFIRMATION'"),
        ),
    )


class ExperimentMetric(Base):

    __tablename__ = "experiment_metrics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_runs.id"), nullable=False
    )
    split: Mapped[MetricSplit] = mapped_column(Enum(MetricSplit), nullable=False)
    category: Mapped[MetricCategory] = mapped_column(Enum(MetricCategory), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    sample_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fold_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    computed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    run: Mapped["ExperimentRun"] = relationship(back_populates="metrics")

    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "split",
            "category",
            "metric_name",
            "fold_index",
            name="uq_experiment_metric_identity",
        ),
    )


class CorporateActionVerificationStatus(str, enum.Enum):
    UNVERIFIED = "UNVERIFIED"
    MACHINE_PARSED = "MACHINE_PARSED"
    MANUALLY_VERIFIED = "MANUALLY_VERIFIED"
    DISPUTED = "DISPUTED"


class CorporateActionComponentType(str, enum.Enum):
    SPLIT = "SPLIT"
    BONUS = "BONUS"
    RIGHTS = "RIGHTS"
    DIVIDEND = "DIVIDEND"
    MERGER = "MERGER"
    DEMERGER = "DEMERGER"
    OTHER = "OTHER"


class CorporateActionEvent(Base):

    __tablename__ = "corporate_action_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    source_symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    ex_date: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    record_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    source_isin: Mapped[str | None] = mapped_column(String(32), nullable=True)
    raw_subject_text: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_face_val: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_provider: Mapped[str] = mapped_column(String(128), nullable=False)
    source_url: Mapped[str] = mapped_column(String(512), nullable=False)
    source_snapshot_path: Mapped[str] = mapped_column(String(512), nullable=False)
    source_snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_fetched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    announcement_date: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    first_observed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    verification_status: Mapped[CorporateActionVerificationStatus] = mapped_column(
        Enum(CorporateActionVerificationStatus),
        nullable=False,
        default=CorporateActionVerificationStatus.UNVERIFIED,
    )
    verification_notes: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    supersedes: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("corporate_action_events.id"), nullable=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_by: Mapped[str] = mapped_column(String(128), nullable=False)

    components: Mapped[list["CorporateActionComponent"]] = relationship(
        back_populates="event"
    )

    __table_args__ = (
        UniqueConstraint(
            "instrument_id", "ex_date", "raw_subject_text",
            name="uq_corporate_action_event_identity",
        ),
    )


class CorporateActionComponent(Base):

    __tablename__ = "corporate_action_components"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("corporate_action_events.id"), nullable=False
    )
    parse_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    parser_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="stage3.9-phase-b-v1"
    )
    component_order: Mapped[int] = mapped_column(Integer, nullable=False)
    component_type: Mapped[CorporateActionComponentType] = mapped_column(
        Enum(CorporateActionComponentType), nullable=False
    )
    raw_text_fragment: Mapped[str] = mapped_column(String(512), nullable=False)

    face_value_from: Mapped[float | None] = mapped_column(Float, nullable=True)
    face_value_to: Mapped[float | None] = mapped_column(Float, nullable=True)

    ratio_new: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ratio_existing: Mapped[int | None] = mapped_column(Integer, nullable=True)

    cash_amount_per_share: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)

    verification_status: Mapped[CorporateActionVerificationStatus] = mapped_column(
        Enum(CorporateActionVerificationStatus),
        nullable=False,
        default=CorporateActionVerificationStatus.UNVERIFIED,
    )
    verification_notes: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    event: Mapped["CorporateActionEvent"] = relationship(back_populates="components")

    __table_args__ = (
        UniqueConstraint(
            "event_id",
            "parse_revision",
            "component_order",
            name="uq_corporate_action_component_revision_order",
        ),
        CheckConstraint(
            "parse_revision >= 1", name="ck_corporate_action_component_parse_revision"
        ),
    )


class CorporateActionVerificationLog(Base):

    __tablename__ = "corporate_action_verification_log"

    seq: Mapped[int] = mapped_column(
        BigInteger, Identity(always=True), nullable=False, unique=True
    )
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    target_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    target_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    from_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    to_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    logged_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class CorporateActionCoverageStatus(str, enum.Enum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"
    CONFLICTED = "CONFLICTED"


class CorporateActionCoverageAssessment(Base):

    __tablename__ = "corporate_action_coverage_assessments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    window_start: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    window_end: Mapped[datetime.date] = mapped_column(Date, nullable=False)
    status: Mapped[CorporateActionCoverageStatus] = mapped_column(
        Enum(CorporateActionCoverageStatus), nullable=False
    )
    reasoning: Mapped[str] = mapped_column(String(2048), nullable=False)
    sources_considered: Mapped[list] = mapped_column(JSONB, nullable=False)
    assessed_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    assessed_by: Mapped[str] = mapped_column(String(128), nullable=False)

    evidence: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    evidence_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    derived_by: Mapped[str | None] = mapped_column(String(96), nullable=True)

    __table_args__ = (
        CheckConstraint(
            "window_end >= window_start", name="ck_coverage_assessment_window_order"
        ),
    )


class CorporateActionUnresolvedEvent(Base):

    __tablename__ = "corporate_action_unresolved_events"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("corporate_action_coverage_assessments.id"),
        nullable=False,
    )
    event_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("corporate_action_events.id"), nullable=True
    )
    ex_date: Mapped[datetime.date | None] = mapped_column(Date, nullable=True)
    action_class: Mapped[str] = mapped_column(String(24), nullable=False)
    reason_code: Mapped[str] = mapped_column(String(48), nullable=False)
    severity: Mapped[str] = mapped_column(String(24), nullable=False)
    detail: Mapped[dict] = mapped_column(JSONB, nullable=False)
    registered_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "severity IN ('BLOCKS_COMPLETE', 'BLOCKS_ADJUSTMENT', 'INFO')",
            name="ck_unresolved_event_severity",
        ),
    )


class CorporateActionTermVerification(Base):

    __tablename__ = "corporate_action_term_verifications"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False, unique=True)
    component_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("corporate_action_components.id"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    action_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    term_a: Mapped[object] = mapped_column(Numeric(20, 6), nullable=False)
    term_b: Mapped[object] = mapped_column(Numeric(20, 6), nullable=False)
    tier: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    evidence: Mapped[dict] = mapped_column(JSONB, nullable=False)
    evidence_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    attested_by: Mapped[str] = mapped_column(String(128), nullable=False)
    attestation_note: Mapped[str] = mapped_column(Text, nullable=False)
    attested_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    pit_status: Mapped[str] = mapped_column(String(24), nullable=False, server_default="NOT_ESTABLISHED")
    revokes_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("corporate_action_term_verifications.id"), nullable=True
    )


class OptionRecommendation(Base):

    __tablename__ = "option_recommendations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    decision_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    underlying: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    shape: Mapped[str | None] = mapped_column(String(32), nullable=True)
    data_quality: Mapped[str] = mapped_column(String(48), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    strategy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)


class OptionPaperTradeEvent(Base):

    __tablename__ = "option_paper_trade_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    recommendation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("option_recommendations.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    event_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)


class OptionSessionEvent(Base):

    __tablename__ = "option_session_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    recorded_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    event_timestamp: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    category: Mapped[str] = mapped_column(String(48), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)

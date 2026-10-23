from __future__ import annotations

import csv
import dataclasses
import gzip
import hashlib
import io
import re
import zipfile
from typing import Sequence

from ..sources.errors import InvalidContentError


class SchemaMismatch(InvalidContentError):
    pass


@dataclasses.dataclass(frozen=True)
class SchemaSpec:
    schema_id: str
    provider: str
    dataset: str
    container: str
    named_columns: tuple[str, ...]
    opaque: re.Pattern[str] = re.compile(r"^(Rsvd\d+)?$")

    def named_of(self, header: Sequence[str]) -> tuple[str, ...]:
        return tuple(c for c in header if not self.opaque.match(c))


@dataclasses.dataclass(frozen=True)
class ContentInfo:
    schema_id: str
    format_version: str
    header: tuple[str, ...]
    data_rows: int


def format_version_of(header: Sequence[str]) -> str:
    return hashlib.sha256("\x1f".join(header).encode("utf-8")).hexdigest()[:16]


_LEGACY = ("INSTRUMENT", "SYMBOL", "EXPIRY_DT", "STRIKE_PR", "OPTION_TYP", "OPEN", "HIGH", "LOW", "CLOSE", "SETTLE_PR", "CONTRACTS", "VAL_INLAKH", "OPEN_INT", "CHG_IN_OI", "TIMESTAMP")
_UDIFF = (
    "TradDt", "BizDt", "Sgmt", "Src", "FinInstrmTp", "FinInstrmId", "ISIN", "TckrSymb", "SctySrs", "XpryDt", "FininstrmActlXpryDt", "StrkPric", "OptnTp", "FinInstrmNm", "OpnPric",
    "HghPric", "LwPric", "ClsPric", "LastPric", "PrvsClsgPric", "UndrlygPric", "SttlmPric", "OpnIntrst", "ChngInOpnIntrst", "TtlTradgVol", "TtlTrfVal", "TtlNbOfTxsExctd", "SsnId",
    "NewBrdLotQty", "Rmks",
)
_MASTER = (
    "FinInstrmId", "UndrlygFinInstrmId", "FinInstrmNm", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "PrtdToTrad", "MinLot", "NewBrdLotQty",
    "BidIntrvl", "StartDt", "EndDt", "MtrtyDt", "PdctStartDt", "SrsId", "AsstCd", "UndrlygInstrmAsstClss", "StockNm", "SttlmMtd", "BasePric",
    "DelFlg", "ExrcStartDt", "ExrcEndDt", "IssdCptl", "MinPric", "MaxPric", "MktTpAndId", "OffclCorpActnEvtId", "AdmssnTp", "IssePric",
    "TradgStsNrmlMkt", "ElgbltyNrmlMkt", "TradgStsOddLotMkt", "ElgbltyOddLotMkt", "TradgStsSpotMkt", "ElgbltySpotMkt", "TradgStsAuctnMkt",
    "ElgbltyAuctnMkt", "MrgnPctg", "MaxTradQty", "WrngQty", "AdmssnDt", "RmvlDt", "RadmssnDt", "RcrdDt", "NoDlvryStartDt", "NoDlvryEndDt", "SpclExDt",
    "BookClsrStartDt", "BookClsrEndDt", "StockRcrdDt", "TckrSelctn", "OdOffclCorpActnEvtId", "CdtRatg", "Mtg", "IntrstOrDvdd", "RghtsOrBns",
    "MinFillAllOrNn", "AddtlInf", "OptnExrcStyle", "ExrcAllwd", "ExrcRjctAllwd", "PLAllwd", "CorpAdjstmnt", "UndrlygInstrm", "DcmlstnPric",
    "PartnIdr", "UnqPdctIdr", "CpctyGrpId", "SctyId", "Mltplr", "SctyLngNm", "UndrlygMkt", "CtrctTp", "FinInstrmPdctTpCd", "FinInstrmTp",
    "TradToTradInd", "IndxFlg", "Indx", "IndxPrtcptnInd", "ExtrnlDataSrcFlg", "LastUpd", "FinInstrmGnlAttrbts", "AddtlUndrlygAttrbts", "UndrlygAsst",
    "OrgnlXpryDt", "SgmtId", "UnitOfMeasr", "PricQtQty", "PricRgTp", "InitlMrgnTp", "BuyInitlMrgnRate", "MaxSnglTxnQty", "MaxSnglTxnVal", "AsstClss",
    "NearMnthInstrmIdr", "FarMnthInstrmIdr", "PricNmrtr", "Spcfctn", "PricDnmtr", "GnlNmrtr", "GnlDnmtr", "LotNmrtr", "LotDnmtr", "DcmlLctr",
    "SttlmTp", "ParVal", "CorpActnRsn", "FreeFltCptl", "ISIN", "SellInitlMrgnRate", "RatgDtls", "FinInstrmClssfctn", "SpclMrgnTp", "BuySpclMrgnRat",
    "SellSpclMrgnRat", "InitlMrgnSprdBnftFlg", "InstrmEndDtTm", "Ccy", "PdctMnth", "GrpId", "MtchgCrit", "SprdTp", "ValMtd", "SLBMElgblty", "PricMtd",
    "InstrmId", "XpryPrcDt", "MrgnInd", "RglrBuyMrgn", "RglrSellMrgn", "SprdBnftAllwd", "DlvryWght", "DlvryUnit",
)

NSE_FO_BHAVCOPY_LEGACY = SchemaSpec("nse.fo.bhavcopy.legacy/1", "nse_archives", "fo_bhavcopy_legacy", "zip", _LEGACY)
NSE_FO_BHAVCOPY_UDIFF = SchemaSpec("nse.fo.bhavcopy.udiff/1", "nse_archives", "fo_bhavcopy_udiff", "zip", _UDIFF)
NSE_FO_CONTRACT_MASTER = SchemaSpec("nse.fo.contract_master/1", "nse_archives", "fo_contract_master", "gzip", _MASTER)


class SchemaRegistry:
    def __init__(self, specs: Sequence[SchemaSpec] = ()) -> None:
        self._by_dataset: dict[tuple[str, str], list[SchemaSpec]] = {}
        for s in specs:
            self.register(s)

    def register(self, spec: SchemaSpec) -> None:
        ids = [s.schema_id for specs in self._by_dataset.values() for s in specs]
        if spec.schema_id in ids:
            raise ValueError(f"schema {spec.schema_id} is already registered (a changed schema needs a NEW id, never an edit)")
        self._by_dataset.setdefault((spec.provider, spec.dataset), []).append(spec)

    def specs_for(self, provider: str, dataset: str) -> list[SchemaSpec]:
        return list(self._by_dataset.get((provider, dataset), []))

    def classify(self, provider: str, dataset: str, header: Sequence[str]) -> tuple[SchemaSpec, str]:
        h = tuple(c.strip() for c in header)
        specs = self.specs_for(provider, dataset)
        if not specs:
            raise SchemaMismatch(f"no schema registered for {provider}/{dataset}")
        for s in specs:
            if s.named_of(h) == s.named_columns:
                return s, format_version_of(h)
        best = specs[-1]
        named = best.named_of(h)
        missing = [c for c in best.named_columns if c not in named]
        extra = [c for c in named if c not in best.named_columns]
        detail = f"unknown NAMED columns {extra}" if extra else (f"missing columns {missing}" if missing else "columns are reordered")
        raise SchemaMismatch(f"{provider}/{dataset}: header does not match registered schema {best.schema_id}: {detail}")

    def inspect(self, provider: str, dataset: str, content: bytes) -> ContentInfo:
        specs = self.specs_for(provider, dataset)
        if not specs:
            raise SchemaMismatch(f"no schema registered for {provider}/{dataset}")
        text = _decode(content, specs[0].container)
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            raise InvalidContentError("empty file")
        spec, fv = self.classify(provider, dataset, rows[0])
        data = [r for r in rows[1:] if r]
        if not data:
            raise InvalidContentError("header present but no data rows")
        return ContentInfo(spec.schema_id, fv, tuple(c.strip() for c in rows[0]), len(data))


def _decode(content: bytes, container: str) -> str:
    try:
        if container == "zip":
            z = zipfile.ZipFile(io.BytesIO(content))
            names = z.namelist()
            if len(names) != 1:
                raise InvalidContentError(f"expected exactly one member in the zip, found {len(names)}")
            raw = z.read(names[0])
        elif container == "gzip":
            raw = gzip.decompress(content)
        else:
            raw = content
        return raw.decode("utf-8-sig")
    except InvalidContentError:
        raise
    except (zipfile.BadZipFile, OSError, EOFError, UnicodeDecodeError) as exc:
        raise InvalidContentError(f"not a valid {container} payload: {exc}") from exc


def default_registry() -> SchemaRegistry:
    return SchemaRegistry([NSE_FO_BHAVCOPY_LEGACY, NSE_FO_BHAVCOPY_UDIFF, NSE_FO_CONTRACT_MASTER])

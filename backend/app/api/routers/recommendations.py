from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..dependencies import get_db, require_api_key
from ..queries import SUPPORTED_UNDERLYINGS, latest_recommendation
from ..schemas import RecommendationSnapshotResponse

router = APIRouter()


@router.get("/recommendations/latest", response_model=RecommendationSnapshotResponse)
def recommendations_latest(
    underlying: str = Query(default="NIFTY"), db: Session = Depends(get_db), _auth: None = Depends(require_api_key),
) -> RecommendationSnapshotResponse:
    if underlying not in SUPPORTED_UNDERLYINGS:
        raise HTTPException(status_code=400, detail=f"underlying must be one of {SUPPORTED_UNDERLYINGS}, got {underlying!r}")
    snapshot = latest_recommendation(db, underlying=underlying)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"no recommendation found for underlying={underlying!r}")
    return RecommendationSnapshotResponse(**snapshot)

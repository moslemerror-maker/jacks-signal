from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..dependencies import current_time, get_db, require_api_key
from ..queries import SUPPORTED_UNDERLYINGS, check_db_reachable, session_status
from ..schemas import SessionStatusResponse

router = APIRouter()


@router.get("/session/status", response_model=SessionStatusResponse)
def session_status_endpoint(
    underlying: str = Query(default="NIFTY"), db: Session = Depends(get_db), now: dt.datetime = Depends(current_time),
    _auth: None = Depends(require_api_key),
) -> SessionStatusResponse:
    if underlying not in SUPPORTED_UNDERLYINGS:
        raise HTTPException(status_code=400, detail=f"underlying must be one of {SUPPORTED_UNDERLYINGS}, got {underlying!r}")
    if not check_db_reachable(db):
        raise HTTPException(status_code=503, detail="database is not reachable")
    return SessionStatusResponse(**session_status(db, underlying=underlying, now=now))

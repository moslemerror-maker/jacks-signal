from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from sqlalchemy.orm import Session

from ..dependencies import get_db, require_api_key
from ..queries import check_db_reachable
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(response: Response, db: Session = Depends(get_db), _auth: None = Depends(require_api_key)) -> HealthResponse:
    reachable = check_db_reachable(db)
    response.status_code = 200 if reachable else 503
    return HealthResponse(status="OK" if reachable else "DB_UNAVAILABLE", database_reachable=reachable)

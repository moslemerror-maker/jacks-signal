from __future__ import annotations

import datetime as dt
import os
from typing import Generator

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.db.session import get_session


def get_db() -> Generator[Session, None, None]:
    db = get_session()
    try:
        yield db
    finally:
        db.close()


def current_time() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    expected = os.environ.get("MONITORING_API_KEY")
    if expected is None:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing or invalid X-API-Key")

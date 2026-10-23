from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import load_dotenv

from .routers import health, recommendations, session

_DEFAULT_ORIGINS = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173"


def _allowed_origins() -> list[str]:
    load_dotenv()
    raw = os.environ.get("MONITORING_UI_ORIGINS", _DEFAULT_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="AEVORYN Monitoring API",
        description="Read-only, provider-neutral monitoring API over already-persisted option-recommendation state. Never calls Dhan.",
        version="ui-1",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=_allowed_origins(), allow_credentials=False,
        allow_methods=["GET"], allow_headers=["X-API-Key"],
    )
    app.include_router(health.router, prefix="/api", tags=["health"])
    app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])
    app.include_router(session.router, prefix="/api", tags=["session"])
    return app

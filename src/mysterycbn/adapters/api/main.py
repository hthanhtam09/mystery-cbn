"""FastAPI app factory for the async job API (ARCHITECTURE.md §5, adapters/api).

    uvicorn mysterycbn.adapters.api.main:app

Swagger UI at ``/docs``, ReDoc at ``/redoc`` (FastAPI defaults); the engine
itself is untouched by this module -- it only calls ``mysterycbn.app.api.convert``
through ``job_store.JobStore`` (ARCHITECTURE.md's adapters → app dependency).
"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mysterycbn import __version__ as ENGINE_VERSION
from mysterycbn.adapters.api.routes import router
from mysterycbn.adapters.api.schemas import ErrorResponse
from mysterycbn.foundation.errors import (
    CancelledError,
    ConfigError,
    EngineError,
    InputError,
    QualityError,
    StageError,
)

_STATUS_BY_TYPE: tuple[tuple[type[EngineError], int], ...] = (
    (InputError, 400),
    (ConfigError, 422),
    (QualityError, 409),
    (CancelledError, 499),
    (StageError, 500),
)


def _status_for(exc: EngineError) -> int:
    for cls, status in _STATUS_BY_TYPE:
        if isinstance(exc, cls):
            return status
    return 500


def _cors_origins() -> list[str]:
    """Frontend origins allowed to call this API (the frontend is a fully
    separate repo/deployment, per its own decoupling requirement).

    ``MYSTERYCBN_CORS_ORIGINS`` is a comma-separated list; unset defaults to
    the local Next.js dev server. There is no session/cookie auth for this
    API to protect (jobs are addressed by an unguessable uuid4, not a
    cookie), so a permissive default is safe -- production deployments
    should still set this explicitly to their real frontend origin(s).
    """
    raw = os.environ.get("MYSTERYCBN_CORS_ORIGINS")
    if raw is None:
        return ["http://localhost:3000", "http://localhost:3100"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def create_app() -> FastAPI:
    app = FastAPI(
        title="mystery-cbn API",
        version=ENGINE_VERSION,
        description="Region-based mystery color-by-number conversion engine.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.exception_handler(EngineError)
    async def _engine_error_handler(request: Request, exc: EngineError) -> JSONResponse:
        body = ErrorResponse(error_type=type(exc).__name__, message=str(exc))
        return JSONResponse(status_code=_status_for(exc), content=body.model_dump())

    return app


app = create_app()

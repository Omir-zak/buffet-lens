"""FastAPI entry point: app wiring, startup (DB, migrations, agent graph) and error handling."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agents.deps import Deps
from app.agents.graph import build_graph
from app.api.auth import require_api_key
from app.api.compat import router as compat_router
from app.api.errors import ApiError
from app.api.routes import router
from app.config import get_settings
from app.db.database import create_checkpoint_pool, create_pool, init_schema

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup before `yield`, shutdown after."""
    settings = get_settings()
    pool = create_pool(settings.database_url)
    await pool.open()
    await init_schema(pool)

    checkpoint_pool = create_checkpoint_pool(settings.database_url)
    await checkpoint_pool.open()
    checkpointer = AsyncPostgresSaver(checkpoint_pool)
    await checkpointer.setup()  # creates the checkpoint tables if missing

    app.state.pool = pool
    app.state.graph = build_graph(Deps(pool=pool), checkpointer)
    yield
    await checkpoint_pool.close()
    await pool.close()


app = FastAPI(
    title="Buffett-Lens",
    description="Research companion built on Warren Buffett's shareholder letters. Not financial advice.",
    lifespan=lifespan,
)
# require_api_key guards all /v1 endpoints (when an API key is configured).
app.include_router(router, dependencies=[Depends(require_api_key)])
app.include_router(compat_router, dependencies=[Depends(require_api_key)])


@app.exception_handler(ApiError)
async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

"""Application factory and process lifecycle.

The lifespan context manager is the heart of this file. Code above `yield`
runs once when the process starts, code below runs once on shutdown. That
is where the model gets loaded, and it is the reason inference is fast:

    startup   ->  load weights, warm up   (a few seconds, paid once)
    request   ->  forward pass only       (tens of milliseconds, paid always)

If you take one idea from this session, take that one.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import demo, health, predict
from app.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging_config import configure_logging
from app.services.detector import Detector
from app.services.inference import InferenceService
from app.services.jobs import JobStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting %s v%s (env=%s)", settings.app_name, settings.version, settings.env)

    detector = Detector(settings)
    detector.load()  # <-- once per process, not once per request

    app.state.settings = settings
    app.state.detector = detector
    app.state.inference = InferenceService(detector, settings)
    app.state.jobs = JobStore(app.state.inference, settings)

    logger.info("Service ready")
    yield

    logger.info("Shutting down")
    detector.unload()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Object detection over HTTP. YOLOv8n served with FastAPI: "
            "single, batch, JSON and asynchronous job endpoints."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        """Give every request an id and a duration. Costs nothing, saves hours."""
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.1f}"

        if request.url.path not in ("/health/live", "/health/ready"):
            logger.info(
                "%s %s -> %s in %.1f ms [%s]",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
                request_id[:8],
            )
        return response

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(predict.router)

    if settings.enable_demo_routes:
        logger.warning("Demo routes enabled. Do not do this in production.")
        app.include_router(demo.router)

    @app.get("/", include_in_schema=False)
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health/ready",
        }

    return app


# uvicorn app.main:app
app = create_app()

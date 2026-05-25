"""CORS and request logging middleware."""

from __future__ import annotations

import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger


def register_middleware(app: FastAPI) -> None:
    """Set up CORS and request timing."""

    # CORS: wildcard origin but no credentials - safe for public demo tunnel.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_and_time_requests(request: Request, call_next):
        """Log request time."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.3f}s)"
        )
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response

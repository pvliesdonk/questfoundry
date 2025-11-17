"""QuestFoundry WebUI API - Main FastAPI application"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import close_pools, init_pools
from .middleware import AuthMiddleware
from .routers import (
    artifacts_router,
    execution_router,
    projects_router,
    user_settings_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager for startup/shutdown events"""
    # Startup: Initialize connection pools
    init_pools(settings)
    yield
    # Shutdown: Close connection pools
    close_pools()


app = FastAPI(
    title="QuestFoundry WebUI API",
    description="Multi-tenant REST API for QuestFoundry",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for PWA
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add authentication middleware
app.add_middleware(AuthMiddleware)

# Include routers
app.include_router(projects_router)
app.include_router(execution_router)
app.include_router(user_settings_router)
app.include_router(artifacts_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint"""
    return {
        "service": "questfoundry-webui-api",
        "message": "QuestFoundry WebUI API",
        "version": "0.1.0",
        "docs": "/docs",
    }

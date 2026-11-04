"""FastAPI application entry point."""

from fastapi import FastAPI

from backend.api.routes import router
from backend.config import load_settings

settings = load_settings()
app = FastAPI(title="QUANT LAB", version="0.1.0")
app.include_router(router)

"""Environment-backed backend configuration."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime settings with safe, non-secret defaults."""

    environment: str = "development"
    log_level: str = "INFO"


def load_settings() -> Settings:
    """Load supported settings from environment variables."""
    return Settings(
        environment=os.getenv("QUANT_LAB_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )

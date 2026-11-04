"""Tests for package imports and configuration."""

import importlib

import pytest

from backend.config import load_settings


@pytest.mark.parametrize(
    "module",
    [
        "quantlab",
        "quantlab.data",
        "quantlab.analytics",
        "quantlab.strategies",
        "quantlab.backtesting",
        "quantlab.risk",
        "quantlab.portfolio",
        "quantlab.execution",
        "quantlab.experiments",
        "backend.main",
    ],
)
def test_package_imports(module: str) -> None:
    assert importlib.import_module(module)


def test_project_configuration() -> None:
    config = importlib.import_module("quantlab")
    assert config.__version__ == "0.1.0"
    settings = load_settings()
    assert settings.environment == "development"
    assert settings.log_level == "INFO"

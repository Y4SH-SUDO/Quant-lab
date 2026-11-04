# Architecture

The repository is a modular monolith. Packages communicate through explicit
Python interfaces and data structures; there are no microservices or broker
integrations in M0.

- `quantlab.data`: M1 daily historical provider interface and Yahoo adapter
  (unadjusted OHLC with a retained adjusted-close comparison series),
  configurable NSE instrument catalogue, OHLCV normalization/quality checks,
  and local Parquet storage. The ingestion pipeline is daily-only.
- `quantlab.analytics`: mathematical and statistical analysis boundaries.
- `quantlab.strategies`: future strategy definitions; intentionally empty in M0.
- `quantlab.backtesting`: future simulation boundary; intentionally empty in M0.
- `quantlab.validation`: future research validation tools.
- `quantlab.risk` and `quantlab.portfolio`: future risk and allocation boundaries.
- `quantlab.execution`: future paper-execution boundary; no live orders.
- `quantlab.experiments`: future reproducible experiment orchestration.
- `backend`: the FastAPI application boundary, currently exposing only health.
- `tests`: unit, integration, and fixture test boundaries.
- `data`, `research`, `scripts`, and `docs`: local datasets, notebooks/research,
  utilities, and supporting documentation respectively.

Dependency direction is inward: API and future orchestration may depend on
domain packages, while domain packages must not depend on the API. Configuration
and secrets are supplied through environment variables, never source code.

"""HTTP routes for the QUANT LAB backend."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a liveness response for the API."""
    return {"status": "ok", "project": "QUANT LAB"}

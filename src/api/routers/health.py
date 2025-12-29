"""Health check endpoint router."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running."""
    return {"status": "healthy"}


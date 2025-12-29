"""Chat endpoint router."""

from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/chat")
async def chat():
    """Chat endpoint placeholder."""
    return {"message": "Hello, may I help you today?"}


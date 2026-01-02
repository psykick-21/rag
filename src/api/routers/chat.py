"""Chat endpoint router."""

from src.ai.rag.orchestrator import RAGOrchestrator
from src.utils.logger import getLogger
from fastapi import APIRouter, Query
from fastapi import HTTPException

logger = getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/chat")
async def chat(
    query: str = Query(..., description="The user query string"),
    only_latest: bool = Query(False, description="Whether to return only the latest results"),
    debug: bool = Query(False, description="Whether to return debug information")
):
    """Chat endpoint accepting a query string."""
    
    orchestrator = RAGOrchestrator()
    try:
        result = orchestrator.run(query, only_latest, debug)
        return result
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
"""Chat endpoint router."""

from src.ai.rag.retriever import Retriever
from fastapi import APIRouter

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/chat")
async def chat():
    """Chat endpoint placeholder."""
    
    retriever = Retriever()
    retrieval_result = retriever.retrieve("What is the model used in the project?")
    return retrieval_result.chunks

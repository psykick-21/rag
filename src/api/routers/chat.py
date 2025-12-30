"""Chat endpoint router."""

from src.ai.rag.retriever import Retriever
from src.ai.rag.generator import Generator
from src.utils.logger import getLogger
from src.api.utils.confidence import compute_confidence
from fastapi import APIRouter

logger = getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

from fastapi import Query

@router.get("/chat")
async def chat(query: str = Query(..., description="The user query string")):
    """Chat endpoint accepting a query string."""
    
    retriever = Retriever()
    generator = Generator()
    
    retrieval_result = retriever.retrieve(query)
    
    answer = generator.generate_answer(query, retrieval_result.chunks)

    citations = [
        {
            "source": chunk.chunk.source,
            "chunk_index": chunk.chunk.metadata["chunk_index"]
        }
        for chunk in retrieval_result.chunks
    ]

    scores = [chunk.distance for chunk in retrieval_result.chunks]
    confidence = compute_confidence(scores)
    
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence
    }

"""Chat endpoint router."""

from src.ai.rag.retriever import Retriever
from src.ai.rag.generator import Generator
from src.utils.logger import getLogger
from src.api.utils.confidence import compute_confidence
from src.api.utils.retriever_utils import dedupe_retrieved_chunks, filter_top_k_chunks
from src.ai.rag.query_analyzer import generate_sub_queries
from fastapi import APIRouter, Query

logger = getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])


@router.get("/chat")
async def chat(query: str = Query(..., description="The user query string")):
    """Chat endpoint accepting a query string."""
    
    retriever = Retriever()
    generator = Generator()

    sub_queries = generate_sub_queries(query)
    
    retrieval_results = []
    for sub_query in sub_queries:
        retrieval_result = retriever.retrieve(sub_query)
        retrieval_results.extend(retrieval_result.chunks)

    
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

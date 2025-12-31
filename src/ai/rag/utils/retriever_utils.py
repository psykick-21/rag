from typing import List
from src.ai.rag.models import RetrievedDocumentChunk


def dedupe_retrieved_chunks(retrieved_chunks: List[RetrievedDocumentChunk]) -> List[RetrievedDocumentChunk]:
    """Deduplicates the retrieved chunks."""

    seen = {}
    result = []
    for chunk in retrieved_chunks:
        key = (chunk.chunk.source, chunk.chunk.metadata.get("chunk_index"))
        if key not in seen:
            seen[key] = chunk
            result.append(chunk)
    return result


def filter_top_k_chunks(chunks: List[RetrievedDocumentChunk], k=5) -> List[RetrievedDocumentChunk]:
    """Limit the top k chunks based on distances"""

    sorted_chunks = sorted(chunks, key=lambda x: x.distance, reverse=False)
    return sorted_chunks[:k]
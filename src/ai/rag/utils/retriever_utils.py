from typing import List
from src.ai.rag.models import RetrievedDocumentChunk


def dedupe_retrieved_chunks(retrieved_chunks: List[RetrievedDocumentChunk]) -> List[RetrievedDocumentChunk]:
    """Deduplicates the retrieved chunks."""

    seen = set()
    result = []
    for chunk in retrieved_chunks:
        # Get chunk_index safely, defaulting to None if metadata is missing or doesn't have chunk_index
        chunk_index = None
        if chunk.chunk.metadata:
            chunk_index = chunk.chunk.metadata.get("chunk_index")
        key = (chunk.chunk.source, chunk_index)
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


def filter_top_k_chunks(chunks: List[RetrievedDocumentChunk], k=5) -> List[RetrievedDocumentChunk]:
    """Limit the top k chunks based on distances"""

    sorted_chunks = sorted(chunks, key=lambda x: x.distance, reverse=False)
    return sorted_chunks[:k]
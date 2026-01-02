from src.ai.rag.models import RetrievalResult

class DebugUtils:
    """Utility class for debugging."""

    def __init__(self):
        pass

    @staticmethod
    def calc_debug_metrics_for_sub_query(
        sub_query: str,
        retrieval_result: RetrievalResult
    ):
        """Calculates the debug metrics for a sub-query."""

        num_chunks = len(retrieval_result.chunks)
        if num_chunks > 0:
            min_distance = min(chunk.distance for chunk in retrieval_result.chunks) or 999
            max_distance = max(chunk.distance for chunk in retrieval_result.chunks) or 999
            avg_distance = sum(chunk.distance for chunk in retrieval_result.chunks) / num_chunks or 999
        else:
            min_distance = 999
            max_distance = 999
            avg_distance = 999

        return {
            "sub_query": sub_query,
            "num_chunks": len(retrieval_result.chunks),
            "min_distance": min_distance,
            "max_distance": max_distance,
            "avg_distance": avg_distance
        }
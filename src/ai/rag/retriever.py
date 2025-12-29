from src.ai.rag.models import RetrievalResult

class Retriever:
    """
    Responsible only for retrieving relevant document chunks.
    """

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        raise NotImplementedError